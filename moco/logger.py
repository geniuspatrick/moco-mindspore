import logging
import os
import sys
from copy import deepcopy
from typing import Optional
from termcolor import colored

import time
import numpy as np
import mindspore as ms
from mindspore import ops, nn
from mindspore.train.callback import Callback

logger_initialized = {}


class _ColorfulFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None, style="%"):
        super().__init__(fmt, datefmt, style)

    def formatMessage(self, record):
        record = deepcopy(record)  # deepcopy avoid change of original record, which may influencing other handler
        record.asctime = colored(record.asctime, "green")
        record.name = colored(record.name, "blue")
        # record.filename, record.funcName, record.lineno
        if record.levelno == logging.DEBUG:
            record.levelname = colored(record.levelname, "magenta")
        elif record.levelno == logging.INFO:
            record.levelname = colored(record.levelname, "green")
        elif record.levelno == logging.WARNING:
            record.levelname = colored(record.levelname, "yellow", attrs=["blink"])
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            record.levelname = colored(record.levelname, "red", attrs=["blink", "underline"])
        return super().formatMessage(record)


def setup_logger(name: Optional[str] = None, output_dir: Optional[str] = None, rank: int = 0,
                 log_level: int = logging.INFO, color: bool = True) -> logging.Logger:
    """Initialize the logger.
    If the logger has not been initialized, this method will initialize the
    logger by adding one or two handlers, otherwise the initialized logger will
    be directly returned. During initialization, only the logger of the master
    process is added console handler. If ``output_dir`` is specified, all loggers
    will be added file handler.
    Args:
        name (str): Logger name. Defaults to None to setup root logger.
        output_dir (str): The directory to save log.
        rank (int): Process rank in the distributed training. Defaults to 0.
        log_level (int): Verbosity level of the logger. Defaults to ``logging.INFO``.
        color (bool): If True, color the output. Defaults to True.
    Returns:
        logging.Logger: A initialized logger.
    """
    if name in logger_initialized:
        return logger_initialized[name]

    # get root logger if name is None
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    # the messages of this logger will not be propagated to its parent
    logger.propagate = False

    fmt = "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)
    color_formatter = _ColorfulFormatter(fmt=fmt, datefmt=datefmt)

    # create console handler for master process
    if rank == 0:
        console_handler = logging.StreamHandler(stream=sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(color_formatter if color else formatter)
        logger.addHandler(console_handler)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(output_dir, f"rank{rank}.log"))
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger_initialized[name] = logger
    return logger


class TrainMonitor(Callback):
    def __init__(self, per_print_steps):
        super().__init__()
        self.losses = None
        self.step_time = None
        self.epoch_time = None
        self.per_print_steps = per_print_steps
        self.last_print_step = -1
        self._logger = logging.getLogger(self.__class__.__name__)

    def on_train_epoch_begin(self, run_context):
        """Record time at the beginning of epoch."""
        self.losses = []
        self.epoch_time = time.time()
        self.last_print_step = run_context.original_args().cur_step_num - 1

    def on_train_epoch_end(self, run_context):
        """Print training info at the end of epoch."""
        callback_params = run_context.original_args()
        epoch_time = (time.time() - self.epoch_time) * 1000
        per_step_time = epoch_time / callback_params.batch_num
        self._logger.info(f"Epoch time: {epoch_time:.2f} ms, "
                          f"per step time: {per_step_time:.2f} ms, "
                          f"avg loss: {np.mean(self.losses):.4f}")

    def on_train_step_begin(self, run_context):
        """Record time at the beginning of step."""
        self.step_time = time.time()

    def on_train_step_end(self, run_context):
        """Print training info as the end of step."""
        cb_params = run_context.original_args()
        step_time = (time.time() - self.step_time) * 1000
        epoch_num = cb_params.epoch_num
        batch_num = cb_params.batch_num
        step_num = batch_num * epoch_num
        cur_step_num = cb_params.cur_step_num - 1  # make it start from 0
        cur_epoch_num = cb_params.cur_epoch_num - 1  # make it start from 0
        cur_batch_num = cur_step_num % batch_num

        if cb_params.optimizer is not None:
            optimizer = cb_params.optimizer
        else:
            optimizer = cb_params.train_network.network.optimizer
        if optimizer.dynamic_lr:
            lr = optimizer.learning_rate(ms.Tensor(cur_step_num)).reshape(())
        else:
            lr = optimizer.learning_rate

        loss = cb_params.net_outputs
        if isinstance(loss, (tuple, list)):
            loss = loss[0]
        if isinstance(loss, ms.Tensor) and isinstance(loss.asnumpy(), np.ndarray):
            loss = np.mean(loss.asnumpy())
        if isinstance(loss, float) and (np.isnan(loss) or np.isinf(loss)):
            raise ValueError("Invalid loss, terminate training.")
        self.losses.append(loss)

        if (cur_step_num - self.last_print_step) >= self.per_print_steps:
            self.last_print_step = cur_step_num
            epoch_width = len(str(epoch_num))
            batch_width = len(str(batch_num))
            step_width = len(str(step_num))
            self._logger.info(f"Epoch:[{cur_epoch_num:{epoch_width}d}/{epoch_num:{epoch_width}d}], "
                              f"batch:[{cur_batch_num:{batch_width}d}/{batch_num:{batch_width}d}], "
                              f"step:[{cur_step_num:{step_width}d}/{step_num:{step_width}d}], "
                              f"loss:[{loss:.4f}/{np.mean(self.losses):.4f}], "
                              f"lr:{lr.asnumpy():.6f}, "
                              f"time:{step_time:.2f}ms")


class AllReduce(nn.Cell):
    """We have to wrap AllReduce in a Cell. WTF!"""
    def __init__(self):
        super().__init__()
        self.all_reduce = ops.AllReduce()

    def construct(self, x):
        return self.all_reduce(x)


class EvalMonitor(Callback):
    def __init__(self, model, dataset_eval, rank_id=0, device_num=1):
        super().__init__()
        self.model = model
        self.dataset_eval = dataset_eval
        self.rank_id = rank_id
        self.device_num = device_num
        if self.device_num == 1:
            self.all_reduce = lambda x: x
        else:
            self.all_reduce = AllReduce()
        self._logger = logging.getLogger(self.__class__.__name__)

    def on_train_epoch_end(self, run_context):
        cb_params = run_context.original_args()
        self._logger.info("Evaluating...")
        start = time.time()
        result = self.model.eval(self.dataset_eval, dataset_sink_mode=False)
        vs = ms.Tensor(list(result.values()), ms.float32)
        vs = self.all_reduce(vs)
        vs /= self.device_num
        for k, v in zip(result.keys(), vs.asnumpy()):
            result[k] = float(v)
        self._logger.info(result)
        self._logger.info(f"Evaluation elapsed {(time.time() - start) * 1e3}ms.")

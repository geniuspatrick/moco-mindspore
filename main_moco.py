import argparse
import logging
import pprint
import random
import os

import mindspore as ms
from mindspore import context, nn
from mindspore.train import FixedLossScaleManager, Model
from mindspore.communication import init, get_group_size, get_rank
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig

from moco.loader import create_dataset_moco
from moco.resnet import resnet50
from moco.builder import MoCo, WithLossCell, MoCoTrainStep
from moco.lr_scheduler import multi_step_lr, cosine_lr
from moco.logger import setup_logger, TrainMonitor

model_names = ["resnet50"]

parser = argparse.ArgumentParser(description='MindSpore Unsupervised Training')
group = parser.add_argument_group('OpenI')
group.add_argument('--device_target', type=str, default='Ascend')
group.add_argument('--data_url', type=str, default='/cache/data', help='obs path to dataset')
group.add_argument('--train_url', type=str, default='/cache/output', help='obs path to dumped ckpt')

parser.add_argument('--data', default='/path/to/imagenet', type=str, metavar='DIR',
                    help='path to dataset')
parser.add_argument('-o', '--output-dir', default='/path/to/output', type=str, metavar='DIR',
                    help='path to output')
parser.add_argument('-a', '--arch', default='resnet50', choices=model_names, metavar='ARCH',
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N',
                    help='mini-batch size (default: 256), this is the total batch size of all GPUs on the current node '
                         'when using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float, metavar='LR',
                    help='initial learning rate', dest='learning_rate')
parser.add_argument('--milestones', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W',
                    help='weight decay (default: 1e-4)', dest="weight_decay")
parser.add_argument('-p', '--print-freq', default=100, type=int, metavar='N',
                    help='print frequency (default: 100)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training')
parser.add_argument('--distributed', default=True, type=bool,
                    help='if distributed training')
parser.add_argument('--mode', default=0, type=int,
                    help='running in GRAPH_MODE(0) or PYNATIVE_MODE(1).')
parser.add_argument('--amp-level', default='O0', type=str,
                    help='level for auto mixed precision training')
parser.add_argument('--loss-scale', default=128, type=int,
                    help='magnification factor of gradients')
parser.add_argument('--dataset-sink-mode', default=True, type=bool,
                    help='whether to sink data')

# moco specific configs:
parser.add_argument('--moco-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--moco-k', default=65536, type=int,
                    help='queue size; number of negative keys (default: 65536)')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--moco-t', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')

# options for moco v2
parser.add_argument('--mlp', action='store_true',
                    help='use mlp head')
parser.add_argument('--aug-plus', action='store_true',
                    help='use moco v2 data augmentation')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        ms.set_seed(args.seed)
    context.set_context(mode=args.mode)
    if args.distributed:
        init()
        rank_id = get_rank()
        device_num = get_group_size()
        context.set_auto_parallel_context(device_num=device_num, parallel_mode='data_parallel',
                                          gradients_mean=True, parameter_broadcast=True)
    else:
        rank_id = 0
        device_num = 1
    setup_logger(output_dir=args.output_dir, rank=rank_id)
    _logger = logging.getLogger('moco')
    _logger.info(f"Experiment Configuration:\n{pprint.pformat(args.__dict__)}")

    _logger.info(f"Building dataset from {args.data}...")
    assert args.batch_size % device_num == 0, "Global batch size must be divisible by the number of devices!"
    local_bs = args.batch_size // device_num
    global_bs = args.batch_size
    train_dataset = create_dataset_moco(dataset_path=os.path.join(args.data, 'train'), aug_plus=args.aug_plus,
                                        batch_size=local_bs, workers=args.workers,
                                        rank_id=rank_id, device_num=device_num)
    n_batches = train_dataset.get_dataset_size()
    _logger.info(f"Local batch size: {local_bs}, global batch size: {global_bs}, number of devices: {device_num}, "
                 f"got {n_batches} batches per epoch.")

    _logger.info(f"Building model {args.arch}...")
    network = MoCo(resnet50, args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp, rank_id, device_num)

    # define loss function (criterion) and optimizer
    criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    net_with_criterion = WithLossCell(network, criterion)
    if args.cos:  # cosine lr schedule
        lrs = cosine_lr(args.learning_rate, n_batches, args.epochs)
    else:  # stepwise lr schedule
        lrs = multi_step_lr(args.learning_rate, args.milestones, n_batches, args.epochs)

    if args.amp_level != "O0":
        _logger.warning("amp_level has to be O0, for MoCoTrainStep doesn't support loss scale!")
    optimizer = nn.Momentum(network.trainable_params(), learning_rate=lrs,
                            momentum=args.momentum, weight_decay=args.weight_decay)
    train_one_step = MoCoTrainStep(net_with_criterion, optimizer)
    model = Model(train_one_step)

    # callbacks
    callbacks = [TrainMonitor(per_print_steps=args.print_freq)]
    if rank_id == 0:
        callbacks.append(ModelCheckpoint(prefix=args.arch, directory=args.output_dir,
                                         config=CheckpointConfig(save_checkpoint_steps=n_batches)))

    _logger.info("Start training...")
    # Init Profiler
    # Note that the Profiler should be initialized before model.train
    # profiler = ms.Profiler(output_path='./profiler_data', profile_communication=True, profile_memory=True)
    model.train(args.epochs, train_dataset, callbacks=callbacks, dataset_sink_mode=args.dataset_sink_mode)
    # model.train(5, train_dataset, callbacks=callbacks, dataset_sink_mode=True, sink_size=100)
    # Profiler end
    # profiler.analyse()


if __name__ == '__main__':
    main()

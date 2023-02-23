import argparse
import logging
import pprint
import random
import os

import mindspore as ms
from mindspore import context, ops, nn, Tensor
from mindspore.common.initializer import initializer, Normal, Zero
from mindspore.train import FixedLossScaleManager, Model
from mindspore.communication import init, get_group_size, get_rank
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig

from moco.loader import create_dataset_lincls
from moco.resnet import resnet50
from moco.lr_scheduler import multi_step_lr, cosine_lr
from moco.logger import setup_logger, TrainMonitor, EvalMonitor

model_names = ["resnet50"]

parser = argparse.ArgumentParser(description='MindSpore Linear Classification')

parser.add_argument('--data', default='/path/to/imagenet', type=str, metavar='DIR',
                    help='path to dataset')
parser.add_argument('-o', '--output-dir', default='/path/to/output', type=str, metavar='DIR',
                    help='path to output')
parser.add_argument('-a', '--arch', default='resnet50', choices=model_names, metavar='ARCH',
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=256, type=int, metavar='N',
                    help='mini-batch size (default: 256), this is the total batch size of all GPUs on the current node '
                         'when using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=30, type=float, metavar='LR',
                    help='initial learning rate', dest='learning_rate')
parser.add_argument('--milestones', default=[60, 80], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10X)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=0.0, type=float, metavar='W',
                    help='weight decay (default: 0.)', dest="weight_decay")
parser.add_argument('-p', '--print-freq', default=100, type=int, metavar='N',
                    help='print frequency (default: 100)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training')
parser.add_argument('--distributed', default=True, type=bool,
                    help='if distributed training')
parser.add_argument('--amp-level', default='O0', type=str,
                    help='level for auto mixed precision training')
parser.add_argument('--loss-scale', default=128, type=int,
                    help='magnification factor of gradients')
parser.add_argument('--dataset-sink-mode', default=True, type=bool,
                    help='whether to sink data')

parser.add_argument('--pretrained', default='', type=str,
                    help='path to moco pretrained checkpoint')


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
    train_dataset = create_dataset_lincls(dataset_path=os.path.join(args.data, 'train'), do_train=True,
                                          batch_size=local_bs, workers=args.workers,
                                          rank_id=rank_id, device_num=device_num)
    val_dataset = create_dataset_lincls(dataset_path=os.path.join(args.data, 'val'), do_train=False,
                                        batch_size=50, workers=args.workers,
                                        rank_id=rank_id, device_num=device_num)
    n_batches_train = train_dataset.get_dataset_size()
    n_batches_val = val_dataset.get_dataset_size()
    _logger.info(f"Local batch size: {local_bs}, global batch size: {global_bs}, number of devices: {device_num}, "
                 f"got {n_batches_train} batches per epoch when training, {n_batches_val} when validating.")

    _logger.info(f"Building model {args.arch}...")
    network = resnet50()
    # freeze all layers but the last fc
    for param in network.trainable_params():
        if param.name not in ["classifier.weight", "classifier.bias"]:
            param.requires_grad = False
    # init the fc layer
    network.classifier.weight.set_data(initializer(Normal(0.01, 0), network.classifier.weight.shape))
    network.classifier.bias.set_data(initializer(Zero(), network.classifier.bias.shape))

    # load from pre-trained, before DistributedDataParallel constructor
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = ms.load_checkpoint(args.pretrained)

            # rename moco pre-trained keys
            for k in list(checkpoint.keys()):
                # retain only encoder_q up to before the embedding layer
                if k.startswith("encoder_q") and not k.startswith("encoder_q.classifier"):
                    # remove prefix
                    checkpoint[k[len("encoder_q."):]] = checkpoint[k]
                # delete renamed or unused k
                del checkpoint[k]

            msg = ms.load_param_into_net(network, checkpoint)
            assert set(msg.missing_keys) == {"classifier.weight", "classifier.bias"}
            print("=> loaded pre-trained model '{}'".format(args.pretrained))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))
    _logger.info(network.trainable_params())

    # define loss function (criterion) and optimizer
    criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    lrs = multi_step_lr(args.learning_rate, args.milestones, n_batches_train, args.epochs)
    optimizer = nn.Momentum(network.trainable_params(), learning_rate=lrs,
                            momentum=args.momentum, weight_decay=args.weight_decay, loss_scale=args.loss_scale)

    eval_metrics = {'Acc@1': nn.Top1CategoricalAccuracy(),
                    'Acc@5': nn.Top5CategoricalAccuracy()}
    loss_scale_manager = FixedLossScaleManager(loss_scale=args.loss_scale, drop_overflow_update=False)
    model = Model(network, loss_fn=criterion, optimizer=optimizer, metrics=eval_metrics,
                  amp_level=args.amp_level, loss_scale_manager=loss_scale_manager)

    # callbacks
    callbacks = [TrainMonitor(per_print_steps=args.print_freq),
                 EvalMonitor(model, val_dataset, rank_id, device_num)]
    if rank_id == 0:
        callbacks.append(ModelCheckpoint(prefix=args.arch, directory=args.output_dir,
                                         config=CheckpointConfig(save_checkpoint_steps=args.print_freq)))

    _logger.info("Start training...")
    model.train(args.epochs, train_dataset, callbacks=callbacks, dataset_sink_mode=args.dataset_sink_mode)


if __name__ == '__main__':
    main()

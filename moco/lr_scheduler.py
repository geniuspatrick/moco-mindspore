import math
from bisect import bisect_right


def multi_step_lr(lr_init, milestones, steps_per_epoch, epochs):
    """MultiStep decay learning rate."""
    steps = epochs * steps_per_epoch
    milestones = sorted(milestones)
    lrs = []
    for i in range(steps):
        epoch_idx = math.floor(i / steps_per_epoch)
        lr = lr_init * 0.1 ** bisect_right(milestones, epoch_idx)
        lrs.append(lr)
    return lrs


def cosine_lr(lr_init, steps_per_epoch, epochs):
    """Cosine decay learning rate."""
    steps = epochs * steps_per_epoch
    lrs = []
    for i in range(steps):
        t_cur = i // steps_per_epoch
        lr = lr_init * (1. + math.cos(math.pi * t_cur / epochs)) / 2
        lrs.append(lr)
    return lrs

"""create train or eval dataloader."""
import random
import numpy as np
from PIL import ImageFilter

import mindspore as ms
import mindspore.dataset as ds
from mindspore.dataset import vision, transforms

image_size = 224
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=(0.1, 2.0)):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class TwoCropTransform:
    """Take two random crops of one iamge as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


def create_dataset_moco(dataset_path, aug_plus, batch_size=32, workers=8, rank_id=0, device_num=1):
    if aug_plus:
        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        trans = [
            vision.Decode(to_pil=True),
            vision.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            transforms.RandomApply([
                vision.RandomColorAdjust(0.4, 0.4, 0.4, 0.4)
            ], prob=0.8),
            vision.RandomGrayscale(prob=0.2),
            transforms.RandomApply([
                GaussianBlur([0.1, 2.0])
            ], prob=0.5),
            vision.RandomHorizontalFlip(),
            vision.ToTensor(),
            vision.Normalize(mean=mean, std=std, is_hwc=False),
        ]
    else:
        # MoCo v1's aug: the same as InstDisc https://arxiv.org/abs/1805.01978
        trans = [
            vision.Decode(to_pil=True),
            vision.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            vision.RandomGrayscale(prob=0.2),
            vision.RandomColorAdjust(0.4, 0.4, 0.4, 0.4),
            vision.RandomHorizontalFlip(),
            vision.ToTensor(),
            vision.Normalize(mean=mean, std=std, is_hwc=False),
        ]
    # move label construction here!
    label_trans = transforms.Fill(0)

    dataset = ds.ImageFolderDataset(dataset_path, num_parallel_workers=workers, shuffle=True,
                                    num_shards=device_num, shard_id=rank_id)

    dataset = dataset.map(operation=transforms.Duplicate(), input_columns="image", output_columns=["im_q", "im_k"],
                          column_order=["im_q", "im_k", "label"],
                          num_parallel_workers=workers)
    dataset = dataset.map(operation=trans, input_columns="im_q",
                          num_parallel_workers=workers)
    dataset = dataset.map(operation=trans, input_columns="im_k",
                          num_parallel_workers=workers)
    dataset = dataset.map(operation=label_trans, input_columns="label",
                          num_parallel_workers=workers)
    dataset = dataset.map(operation=transforms.Duplicate(), input_columns="label",
                          output_columns=["label", "idx"],
                          column_order=["im_q", "im_k", "label", "idx"],
                          num_parallel_workers=workers)
    dataset = dataset.map(operation=transforms.Duplicate(), input_columns="idx",
                          output_columns=["idx_shuffle", "idx_unshuffle"],
                          column_order=["im_q", "im_k", "label", "idx_shuffle", "idx_unshuffle"],
                          num_parallel_workers=workers)

    def per_batch_map(im_q, im_k, label, idx_shuffle, idx_unshuffle, BatchInfo):
        idx_shuffle = np.random.permutation(batch_size * device_num)
        idx_unshuffle = np.argsort(idx_shuffle)
        idx_shuffle = np.reshape(idx_shuffle, (batch_size, device_num)).astype(np.int32)
        idx_unshuffle = np.reshape(idx_unshuffle, (batch_size, device_num)).astype(np.int32)
        return im_q, im_k, label, idx_shuffle, idx_unshuffle

    # apply batch operations
    dataset = dataset.batch(batch_size, drop_remainder=True, per_batch_map=per_batch_map)
    return dataset


def create_dataset_lincls(dataset_path, do_train, batch_size=32, workers=32, rank_id=0, device_num=1):
    if do_train:
        trans = [
            vision.Decode(to_pil=True),
            vision.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            vision.RandomHorizontalFlip(),
            vision.ToTensor(),
            vision.Normalize(mean=mean, std=std, is_hwc=False),
        ]
        dataset = ds.ImageFolderDataset(dataset_path, num_parallel_workers=workers, shuffle=True,
                                        num_shards=device_num, shard_id=rank_id)
    else:
        trans = [
            vision.Decode(to_pil=True),
            vision.Resize(256),
            vision.CenterCrop(image_size),
            vision.ToTensor(),
            vision.Normalize(mean=mean, std=std, is_hwc=False),
        ]
        dataset = ds.ImageFolderDataset(dataset_path, num_parallel_workers=workers, shuffle=False)

    label_trans = transforms.TypeCast(ms.int32)

    dataset = dataset.map(operation=trans, input_columns="image",
                          num_parallel_workers=workers)
    dataset = dataset.map(operation=label_trans, input_columns="label",
                          num_parallel_workers=workers)

    # apply batch operations
    dataset = dataset.batch(batch_size)
    return dataset

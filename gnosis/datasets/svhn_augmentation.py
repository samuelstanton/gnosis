import numpy as np
import torchvision
import functools


def make_svhn_augmented_dataset(base_ds_class, root, download, train, transform, num_svhn_data, seed=0):
    """Appends data from the SVHN dataset to the given dataset base_ds_class.

    Note: test set is not modified.

    Args:
      base_ds_class: class to create the base dataset; works for CIFAR10 and
            CIFAR100 from torchvision. For other datasets would need to
            re-implement the interface.
      root, download, train, transforms: same as torchvision
      num_svhn_data: number of datapoints from SVHN to add.
      seed: random seed for choosing the subset of SVHN.
    """

    base_ds = base_ds_class(root, train, transform, download=download)

    if train:
        original_data = base_ds.data
        original_targets = base_ds.targets

        svhn = torchvision.datasets.SVHN(root, "train", transform, download=download)
        new_data = np.transpose(svhn.data, (0, 2, 3, 1))
        np.random.seed(seed)
        np.random.shuffle(new_data)
        new_data = new_data[:num_svhn_data]
        new_targets = np.ones_like(svhn.labels[:num_svhn_data])

        base_ds.data = np.concatenate([original_data, new_data])
        base_ds.targets = np.concatenate(
            [original_targets, new_targets]).astype(np.int64)
    return base_ds


make_cifar100_svhn = functools.partial(
        make_svhn_augmented_dataset, base_ds_class=torchvision.datasets.CIFAR100)

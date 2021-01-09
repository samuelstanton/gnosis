import hydra
from torch.utils.data import random_split
import torchvision


def get_loaders(config):
    train_transform, test_transform = get_augmentation(config)
    train_dataset = hydra.utils.instantiate(config.dataset.init, train=True, transform=train_transform)
    test_dataset = hydra.utils.instantiate(config.dataset.init, train=False, transform=test_transform)

    subsample_ratio = config.dataset.subsample_ratio
    if subsample_ratio > 0.:
        train_dataset, _ = split_dataset(train_dataset, subsample_ratio)
        test_dataset, _ = split_dataset(test_dataset, subsample_ratio)
    if config.trainer.eval_dataset == 'val':
        train_dataset, test_dataset = split_dataset(train_dataset, 0.8)

    train_loader = hydra.utils.instantiate(config.dataloader, dataset=train_dataset)
    test_loader = hydra.utils.instantiate(config.dataloader, dataset=test_dataset)

    return train_loader, test_loader


def split_dataset(dataset, ratio):
    num_total = len(dataset)
    num_split = int(num_total * ratio)
    return random_split(dataset, [num_split, num_total - num_split])


def get_augmentation(config):
    assert 'augmentation' in config.keys()
    if config.augmentation.transforms_list is not None:
        transforms_list = [hydra.utils.instantiate(config.augmentation[name])
                           for name in config.augmentation["transforms_list"].split(",")]
        if 'random_apply' in config.augmentation.keys() and config.augmentation.random_apply.p < 1:
            transforms_list = [
                hydra.utils.instantiate(config.augmentation.random_apply, transforms=transforms_list)]
    else:
        transforms_list = []

    normalize_transforms = [
        torchvision.transforms.ToTensor(),
    ]
    if config.augmentation.normalization == 'zscore':
        # mean subtract and scale to unit variance
        normalize_transforms.append(
            torchvision.transforms.Normalize(config.dataset.statistics.mean_statistics,
                                             config.dataset.statistics.std_statistics)
        )
    elif config.augmentation.normalization == 'unitcube':
        # rescale values to [-1, 1]
        min_vals = config.dataset.statistics.min
        max_vals = config.dataset.statistics.max
        offset = [0.5 * (min_val + max_val) for min_val, max_val in zip(min_vals, max_vals)]
        scale = [(max_val - min_val) / 2 for max_val, min_val in zip(max_vals, min_vals)]
        normalize_transforms.append(
            torchvision.transforms.Normalize(offset, scale)
        )

    train_transform = torchvision.transforms.Compose(transforms_list + normalize_transforms)
    test_transform = torchvision.transforms.Compose(normalize_transforms)
    return train_transform, test_transform

import os
import torch
import hydra
from torch.utils.data import DataLoader, random_split
import torchvision
from upcycle.cuda import try_cuda


def get_generator(config):
    generator = hydra.utils.instantiate(config.generator)
    weight_file = os.path.join(hydra.utils.get_original_cwd(), config.generator.weight_file)
    weight_dict = torch.load(weight_file)
    generator.load_state_dict(weight_dict)
    return try_cuda(generator)


def get_loaders(config):
    train_transform, test_transform = get_augmentation(config)
    train_dataset = hydra.utils.instantiate(config.dataset.init, train=True, transform=train_transform)
    test_dataset = hydra.utils.instantiate(config.dataset.init, train=False, transform=test_transform)

    subsample_ratio = config.dataset.subsample_ratio
    if subsample_ratio > 0.:
        train_dataset, _ = split_dataset(train_dataset, subsample_ratio)
        test_dataset, _ = split_dataset(test_dataset, subsample_ratio)

    train_loader = hydra.utils.instantiate(config.dataloader, dataset=train_dataset)
    test_loader = hydra.utils.instantiate(config.dataloader, dataset=test_dataset)

    return train_loader, test_loader


def split_dataset(dataset, ratio):
    num_total = len(dataset)
    num_split = int(num_total * ratio)
    return random_split(dataset, [num_split, num_total - num_split])


def get_augmentation(config):
    transforms_list = [hydra.utils.instantiate(config.augmentation[name])
                       for name in config.augmentation["transforms_list"].split(",")]
    if config.augmentation.random_apply.p < 1:
        transforms_list = [
            hydra.utils.instantiate(config.augmentation.random_apply, transforms=transforms_list)]
    normalize_transforms = [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(config.dataset.statistics.mean_statistics,
                                         config.dataset.statistics.std_statistics)
    ]

    train_transform = torchvision.transforms.Compose(transforms_list + normalize_transforms)
    test_transform = torchvision.transforms.Compose(normalize_transforms)
    return train_transform, test_transform

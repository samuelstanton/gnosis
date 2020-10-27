import hydra
from torch.utils.data import DataLoader, random_split


def get_loaders(config):
    train_dataset = hydra.utils.call(config.dataset, train=True)
    test_dataset = hydra.utils.call(config.dataset, train=False)

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

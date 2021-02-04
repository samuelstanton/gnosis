from torchvision.datasets import EMNIST, MNIST
import numpy as np
import torch


def make_emnist(root, download, split, train, transform):
    if train:
        train = EMNIST(root, train=True, split=split, transform=transform, download=download)
        # re-setting labels for compatibility with having network with 10 classes outputs
        train.targets = torch.tensor(np.random.randint(0, 10, size=train.targets.size(0))).long()
        return train

    else:
        test = MNIST(root, train=False, download=download, transform=transform)
        return test

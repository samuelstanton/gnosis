from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms


def get_dataset(root, train, download, **kwargs):
    if train:
        input_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    dataset = CIFAR10(root, train, input_transform, download=download)
    return dataset

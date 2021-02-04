from torchvision.datasets import MNIST
import sklearn
from sklearn.model_selection import StratifiedShuffleSplit


def make_subsampled_mnist(root, download, train, transform, num_data, seed=0):
    if train:
        train = MNIST(root, train=True, transform=transform, download=download)
        sss = StratifiedShuffleSplit(n_splits=1, train_size=num_data, random_state=seed)
        for train_index, test_index in sss.split(train.data, train.targets):
            break
        train.data = train.data[train_index]
        train.targets = train.targets[train_index]
        return train

    else:
        test = MNIST(root, train=False, transform=transform, download=download)
        return test

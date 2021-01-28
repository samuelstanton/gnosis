from torch.utils.data import DataLoader
import torch
from upcycle import cuda
from gnosis.distillation.classification import reduce_ensemble_logits
import numpy as np
from omegaconf import ListConfig


def mixup_data(x, alpha):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = cuda.try_cuda(torch.randperm(batch_size))
    mixed_x = lam * x + (1 - lam) * x[index, :]
    return mixed_x


class DistillLoader(object):
    def __init__(self, teacher, datasets, temp, mixup_alpha, batch_size, shuffle, drop_last,
                 **kwargs):
        if isinstance(temp, ListConfig):
            assert len(temp) == len(datasets)
        if isinstance(temp, float):
            temp = [temp] * len(datasets)
        self.teacher = teacher
        self.temp = temp
        self.mixup_alpha = mixup_alpha
        self.batch_size = batch_size
        self.loaders = self._make_loaders(datasets, batch_size, shuffle, drop_last)

    def __len__(self):
        return min([len(ldr) for ldr in self.loaders])

    def __iter__(self):
        return self.generator

    def _make_loaders(self, datasets, total_batch_size, shuffle, drop_last):
        num_total = sum([len(dset) for dset in datasets])
        b_sizes = [
            int(len(dset) / num_total * total_batch_size) for dset in datasets[:-1]
        ]
        b_sizes.append(total_batch_size - sum(b_sizes))
        loaders = [
            DataLoader(dset, bsize, shuffle, drop_last=drop_last) for dset, bsize in zip(datasets, b_sizes)
        ]
        return loaders

    @property
    def generator(self):
        for batches in zip(*self.loaders):
            inputs = cuda.try_cuda(torch.cat([b[0] for b in batches]))
            if self.mixup_alpha > 0.:
                inputs = mixup_data(inputs, self.mixup_alpha)
            targets = cuda.try_cuda(torch.cat([b[1] for b in batches]))
            with torch.no_grad():
                logits = reduce_ensemble_logits(self.teacher(inputs))
            temp = torch.cat([torch.empty_like(b[1]).fill_(t) for b, t in zip(batches, self.temp)])
            temp = cuda.try_cuda(temp)
            yield inputs, targets, logits, temp

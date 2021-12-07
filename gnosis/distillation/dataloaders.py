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
    def __init__(self, teacher, datasets, temp, mixup_alpha, mixup_portion, batch_size, shuffle, drop_last,
                 synth_ratio, synth_sampler=None, synth_temp=1., **kwargs):
        if isinstance(temp, ListConfig):
            assert len(temp) == len(datasets)
        if isinstance(temp, float):
            temp = [temp] * len(datasets)
        if synth_ratio > 0:
            assert synth_sampler is not None
            temp.append(synth_temp)

        self.teacher = teacher
        self.temp = temp
        self.mixup_alpha = mixup_alpha
        self.mixup_portion = mixup_portion
        self.batch_size = batch_size

        self.synth_ratio = synth_ratio
        self.synth_sampler = synth_sampler
        self.synth_temp = synth_temp
        self.loaders = self._make_loaders(datasets, batch_size, shuffle, drop_last)

    def __len__(self):
        return min([len(ldr) for ldr in self.loaders])

    def __iter__(self):
        return self.generator

    def _make_loaders(self, datasets, total_batch_size, shuffle, drop_last):
        assert self.synth_ratio < 1
        num_real = sum([len(dset) for dset in datasets])
        num_total = int(num_real / (1 - self.synth_ratio))
        b_sizes = [
            int(len(dset) / num_total * total_batch_size) for dset in datasets[:-1]
        ]
        synth_bs = int(self.synth_ratio * total_batch_size)
        b_sizes.append(total_batch_size - sum(b_sizes) - synth_bs)
        loaders = [
            DataLoader(dset, bsize, shuffle, drop_last=drop_last) for dset, bsize in zip(datasets, b_sizes)
        ]
        return loaders

    @property
    def generator(self):
        for batches in zip(*self.loaders):
            bs_list = [b[0].size(0) for b in batches]
            inputs = cuda.try_cuda(torch.cat([b[0] for b in batches]))
            targets = cuda.try_cuda(torch.cat([b[1] for b in batches]))

            # mixup augmentation
            if self.mixup_alpha > 0:
                batch_size = inputs.size(0)
                num_mixup = int(np.ceil(self.mixup_portion * batch_size))
                input_mixup = mixup_data(inputs[:num_mixup], self.mixup_alpha)
                inputs = torch.cat((input_mixup, inputs[num_mixup:]))

            # synthetic augmentation
            if self.synth_ratio > 0:
                synth_bs = int(self.synth_ratio * self.batch_size)
                bs_list.append(synth_bs)
                with torch.no_grad():
                    synth_inputs = self.synth_sampler.sample(synth_bs)
                inputs = torch.cat([inputs, synth_inputs], dim=0)

            with torch.no_grad():
                logits = reduce_ensemble_logits(self.teacher(inputs))

            assert len(bs_list) == len(self.temp)
            temp = torch.cat([
                torch.ones(bs, 1) * t for bs, t in zip(bs_list, self.temp)
            ])
            temp = cuda.try_cuda(temp)
            yield inputs, targets, logits, temp


class DistillLoaderFromLoader(object):
    def __init__(self, loader, teacher, temp, **kwargs):
        self.teacher = teacher
        self.temp = temp
        self.loader = loader

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        return self.generator

    @property
    def generator(self):
        for batch in self.loader:
            inputs = cuda.try_cuda(batch[0])
            targets = cuda.try_cuda(batch[1])
            with torch.no_grad():
                logits = reduce_ensemble_logits(self.teacher(inputs))
            temp = cuda.try_cuda(torch.ones_like(logits) * self.temp)
            yield inputs, targets, logits, temp

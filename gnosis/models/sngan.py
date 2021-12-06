import torch
from torch import nn
import torch.nn.functional as F
from omegaconf import DictConfig
import hydra
from torch import nn
import torch.nn.functional as F
import numpy as np
# from .spectral_normalization import SpectralNorm
from torch.nn.utils import spectral_norm


from gnosis.utils.metrics import Expression, Named


class SNGAN(nn.Module):
    def __init__(self, gen_cfg: DictConfig, disc_cfg: DictConfig, *args, **kwargs):
        super().__init__()
        self.generator = hydra.utils.instantiate(gen_cfg)
        self.discriminator = hydra.utils.instantiate(disc_cfg)

    def gen_backward(self, batch_size):
        # Generator hinge loss
        fake_samples = self.generator.sample(batch_size)
        fake_logits = self.discriminator(fake_samples)
        loss = -torch.mean(fake_logits)
        loss.backward()
        return loss, fake_samples

    def disc_backward(self, real_samples):
        # Discriminator hinge loss
        batch_size = real_samples.size(0)
        with torch.no_grad():
            fake_samples = self.generator.sample(batch_size)
        real_logits = self.discriminator(real_samples)
        fake_logits = self.discriminator(fake_samples)
        loss = F.relu(1 - real_logits).mean() + F.relu(1 + fake_logits).mean()
        loss.backward()
        return loss


class GanBase(nn.Module, metaclass=Named):
    def __init__(self, z_dim, img_channels, num_classes=None):
        self.z_dim = z_dim
        self.img_channels = img_channels
        super().__init__()

    @property
    def device(self):
        try:
            return self._device
        except AttributeError:
            self._device = next(self.parameters()).device
            return self._device

    def sample_z(self, n=1):
        return torch.randn(n, self.z_dim).to(self.device)

    def sample(self, n=1):
        return self(self.sample_z(n))


# https://github.com/mfinzi/olive-oil-ml/blob/master/oil/architectures/img_gen/resnetgan.py
# Resnet GAN and Discriminator with Spectral normalization
# Implementation of architectures used in SNGAN (https://arxiv.org/abs/1802.05957)
class Generator(GanBase):
    def __init__(self, z_dim=128, img_channels=3, k=256, **kwargs):
        super().__init__(z_dim, img_channels, **kwargs)
        self.k = k
        self.model = nn.Sequential(
            nn.Linear(z_dim, 4 * 4 * k),
            Expression(lambda z: z.view(-1, k, 4, 4)),
            ResBlockGenerator(k, k, stride=2),
            ResBlockGenerator(k, k, stride=2),
            ResBlockGenerator(k, k, stride=2),
            nn.BatchNorm2d(k),
            nn.ReLU(),
            nn.Conv2d(k, img_channels, 3, stride=1, padding=1),
            nn.Tanh())

        self.apply(xavier_uniform_init)

    def forward(self, z):
        return self.model(z)


class Discriminator(nn.Module, metaclass=Named):
    def __init__(self, img_channels=3, k=128, out_size=1):
        super().__init__()
        self.img_channels = img_channels
        self.k = k
        self.model = nn.Sequential(
            FirstResBlockDiscriminator(img_channels, k, stride=2),
            ResBlockDiscriminator(k, k, stride=2),
            ResBlockDiscriminator(k, k),
            ResBlockDiscriminator(k, k),
            nn.ReLU(),
            nn.AvgPool2d(8),
            Expression(lambda u: u.view(-1, k)),
            nn.Linear(k, out_size)
        )
        self.apply(xavier_uniform_init)
        self.apply(add_spectral_norm)
        # Spectral norm on discriminator but not generator

    def forward(self, x):
        return self.model(x)


class ResBlockGenerator(nn.Module):

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=stride, mode='bilinear') if stride != 1 else nn.Sequential()
        self.model = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.ReLU(),
            self.upsample,
            nn.Conv2d(in_ch, out_ch, 3, 1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, 1, padding=1)
        )
        self.bypass = nn.Conv2d(in_ch, out_ch, 1, 1, padding=0) if in_ch != out_ch else nn.Sequential()

    def forward(self, x):
        return self.model(x) + self.bypass(self.upsample(x))


class ResBlockDiscriminator(nn.Module):

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.model = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_ch, out_ch, 3, 1, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, 1, padding=1)
        )
        self.downsample = nn.AvgPool2d(2, stride=stride, padding=0) if stride != 1 else nn.Sequential()
        self.bypass = nn.Conv2d(in_ch, out_ch, 1, 1, padding=0) if in_ch != out_ch else nn.Sequential()

    def forward(self, x):
        return self.downsample(self.model(x)) + self.downsample(self.bypass(x))


# special ResBlock just for the first layer of the discriminator
class FirstResBlockDiscriminator(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        # we don't want to apply ReLU activation to raw image before convolution transformation.
        self.model = nn.Sequential(
            # nn.ReLU(),
            nn.Conv2d(in_ch, out_ch, 3, 1, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, 1, padding=1)
        )
        self.downsample = nn.AvgPool2d(2, stride=stride, padding=0) if stride != 1 else nn.Sequential()
        self.bypass = nn.Conv2d(in_ch, out_ch, 1, 1, padding=0) if in_ch != out_ch else nn.Sequential()

    def forward(self, x):
        return self.downsample(self.model(x)) + self.downsample(self.bypass(x))


def add_spectral_norm(module):
    if isinstance(module, (nn.ConvTranspose1d,
                           nn.ConvTranspose2d,
                           nn.ConvTranspose3d,
                           )):
        spectral_norm(module, dim=1)
        # print("SN on conv layer: ",module)
    elif isinstance(module, (nn.Linear,
                             nn.Conv1d,
                             nn.Conv2d,
                             nn.Conv3d)):
        spectral_norm(module, dim=0)
        # print("SN on linear layer: ",module)


def xavier_uniform_init(module):
    if isinstance(module, (nn.ConvTranspose1d,
                           nn.ConvTranspose2d,
                           nn.ConvTranspose3d,
                           nn.Conv1d,
                           nn.Conv2d,
                           nn.Conv3d)):
        if module.kernel_size == (1, 1):
            nn.init.xavier_uniform_(module.weight.data, np.sqrt(2))
        else:
            nn.init.xavier_uniform_(module.weight.data, 1)
        # print("Xavier init on conv layer: ",module)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight.data, 1)
        # print("Xavier init on linear layer: ",module)

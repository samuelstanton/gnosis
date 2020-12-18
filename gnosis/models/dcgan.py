# https://raw.githubusercontent.com/pytorch/examples/master/dcgan/main.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from upcycle.cuda import try_cuda
import hydra


class DCGAN(nn.Module):
    def __init__(self, gen_cfg, disc_cfg, **kwargs):
        super().__init__()
        self.generator = hydra.utils.instantiate(gen_cfg)
        self.discriminator = hydra.utils.instantiate(disc_cfg)

    def gen_backward(self, batch_size):
        fake_samples = self.generator.sample(batch_size)
        fake_probs = self.discriminator(fake_samples)
        labels = try_cuda(torch.full((batch_size,), 1.))
        gen_loss = F.binary_cross_entropy(fake_probs, labels)
        gen_loss.backward()
        return gen_loss, fake_samples

    def disc_backward(self, real_samples, fake_samples):
        batch_size = real_samples.size(0)
        real_probs = self.discriminator(real_samples)
        labels = try_cuda(torch.full((batch_size,), 1.))
        real_loss = F.binary_cross_entropy(real_probs, labels)
        real_loss.backward()

        fake_probs = self.discriminator(fake_samples)
        labels = try_cuda(torch.full((batch_size,), 0.))
        fake_loss = F.binary_cross_entropy(fake_probs, labels)
        fake_loss.backward()
        return real_loss + fake_loss


class DCGenerator(nn.Sequential):
    def __init__(self, nc=3, nz=100, ngf=64, output_dim=64, **kwargs):
        super(DCGenerator, self).__init__()
        self.ngpu = torch.cuda.device_count()
        self.z_dim = nz
        self.output_dim = output_dim
        modules = [
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            # state size. (ngf*8) x 4 x 4
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            # state size. (ngf*4) x 8 x 8
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            # state size. (ngf*2) x 16 x 16
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            # state size. ngf x 32 x 32
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
        ]

        if self.output_dim == 32:
            modules.extend([
                nn.ConvTranspose2d(ngf, nc, kernel_size=1, stride=1, padding=0, bias=False),
                # state size. 3 x 32 x 32
                nn.Tanh(),
            ])
        elif self.output_dim == 64:
            modules.extend([
                nn.ConvTranspose2d(ngf, ngf, kernel_size=4, stride=2, padding=1, bias=False),
                # state size. ngf x 64 x 64
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf, nc, kernel_size=1, stride=1, padding=0, bias=False),
                # state size. 3 x 64 x 64
                nn.Tanh(),
            ])
        else:
            raise RuntimeError('[DC-GAN] only 32 x 32 and 64 x 64 outputs supported')
        super().__init__(*modules)
        self.apply(weights_init)

    def forward(self, input):
        return super().forward(input)  # real examples must lie in [-1, 1]

    def sample(self, num_samples):
        z_vecs = self.sample_z(num_samples)
        z_vecs = try_cuda(z_vecs)
        return self(z_vecs)

    def sample_z(self, num_samples):
        return torch.randn(num_samples, self.z_dim, 1, 1)

    @property
    def device(self):
        try: return self._device
        except AttributeError:
            self._device = next(self.parameters()).device
            return self._device


class DCDiscriminator(nn.Sequential):
    def __init__(self, input_size, nc=3, ndf=64, **kwargs):
        if input_size == 32:
            modules = [
                nn.Conv2d(nc, ndf, kernel_size=1, stride=1, padding=0, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        elif input_size == 64:
            modules = [
                nn.Conv2d(nc, ndf, kernel_size=4, stride=2, padding=1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        else:
            raise RuntimeError('[DC-GAN] only 32 x 32 or 64 x 64 images supported')

        modules.extend([
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        ])
        super().__init__(*modules)
        self.ngpu = torch.cuda.device_count()
        self.apply(weights_init)

    def forward(self, inputs):
        return super().forward(inputs).view(-1)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# https://raw.githubusercontent.com/pytorch/examples/master/dcgan/main.py

import torch
import torch.nn as nn
from upcycle.cuda import try_cuda


class DCGenerator(nn.Module):
    def __init__(self, nc=3, nz=100, ngf=64, **kwargs):
        super(DCGenerator, self).__init__()
        self.ngpu = torch.cuda.device_count()
        self.input_dim = nz
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, kernel_size=1, stride=1, padding=0, bias=False),
            # nn.AvgPool2d(2, 2),
            nn.Tanh()
        )
        self.apply(weights_init)

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

    def sample(self, num_samples):
        z_vecs = torch.randn(num_samples, self.input_dim, 1, 1)
        z_vecs = try_cuda(z_vecs)
        return self(z_vecs)


class DCDiscriminator(nn.Sequential):
    def __init__(self, input_size, nc=3, ndf=64, **kwargs):
        if input_size == 32:
            # upsample to 64 x 64
            modules = [nn.ConvTranspose2d(nc, nc, kernel_size=4, stride=2, padding=1, bias=False)]
        elif input_size == 64:
            modules = []
        else:
            raise RuntimeError('[DC-GAN] only 32 x 32 or 64 x 64 images supported')

        modules.extend([
            # state_size is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
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
        return super().forward(inputs)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

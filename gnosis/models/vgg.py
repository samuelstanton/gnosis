"""
Ported from https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
and https://github.com/timgaripov/swa/blob/master/models/vgg.py
"""

import math
import torch.nn as nn


CFG = {
    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M',
         512, 512, 512, 'M'],
    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512,
         512, 'M', 512, 512, 512, 512, 'M'],
}


def make_layers(layer_cfg, batch_norm=False):
    """Make a model given description of the layers.

    Used to make the convolutional part of VGG models.
    """
    layers = []
    in_channels = 3
    for v in layer_cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class VGG(nn.Module):
    def __init__(self, num_classes=10, depth=16, batch_norm=False, **kwargs):
        super(VGG, self).__init__()
        del kwargs
        self.features = make_layers(CFG[depth], batch_norm)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, num_classes),
        )

        # Initialization of convolutional layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


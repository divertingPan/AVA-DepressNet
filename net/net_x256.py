# -*- coding: utf-8 -*-
from torch import nn


class Generator(nn.Module):
    def __init__(self,):
        super(Generator, self).__init__()
        self.main = nn.Sequential(  # [batch, 136, 1, 1]  out = (in-1)*stride + out_padding - 2*padding + kernel_size
            nn.ConvTranspose2d(in_channels=136, out_channels=128, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),  # [batch, 128, 4, 4]
            nn.ConvTranspose2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),  # [batch, 256, 8, 8]
            nn.ConvTranspose2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),  # [batch, 512, 16, 16]
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),  # [batch, 256, 32, 32]
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),  # [batch, 128, 64, 64]
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),  # [batch, 64, 128, 128]
            nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Sigmoid()  # [batch, 3, 256, 256]
        )

    def forward(self, x):
        x = self.main(x)
        return x


class Perceptual_D(nn.Module):
    def __init__(self):
        super(Perceptual_D, self).__init__()
        self.main = nn.Sequential(  # [batch, 3, 256, 256]
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),  # [batch, 64, 128, 128]
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),  # [batch, 128, 64, 64]
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),  # [batch, 256, 32, 32]
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),  # [batch, 512, 16, 16]
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),  # [batch, 256, 8, 8]
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),  # [batch, 128, 4, 4]
            nn.Conv2d(in_channels=128, out_channels=136, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()  # [batch, 136, 1, 1]
        )

    def forward(self, x):
        x = self.main(x)
        return x


class Adversarial_D(nn.Module):
    def __init__(self):
        super(Adversarial_D, self).__init__()
        self.main = nn.Sequential(  # [batch, 3, 256, 256]
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=4, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),  # [batch, 64, 64, 64]
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),  # [batch, 128, 32, 32]
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),  # [batch, 256, 16, 16]
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),  # [batch, 128, 8, 8]
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),  # [batch, 64, 4, 4]
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False),
            # W-GAN: remove sigmoid
            # nn.Sigmoid()  # [batch, 1, 1, 1]
        )

    def forward(self, x):
        x = self.main(x)
        x = x.view(-1)
        return x

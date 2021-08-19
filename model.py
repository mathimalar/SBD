import numpy as np
from copy import deepcopy
import torch.nn as nn
from torch.nn import Conv2d, ReLU, MaxPool2d, Linear, BatchNorm2d, LeakyReLU
import torch.nn.functional as F
import torch


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) *
    """

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=(2, 2), stride=(2, 2))
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x = self.up(x)
        return self.conv(x)


class ActivationNet(nn.Module):
    def __init__(self, input_channels=1):
        super().__init__()
        #  These values make sure the image dimensions stay the same
        self.input_channels = input_channels
        self.down1 = Down(2 ** 2, 2 ** 4)
        self.down2 = Down(2 ** 4, 2 ** 6)
        self.up1 = Up(2 ** 6, 2 ** 4)
        self.up2 = Up(2 ** 4, 2 ** 2)
        pad = 2
        ker_same = (5, 5)
        self.features = nn.Sequential(

            # Input layer
            Conv2d(input_channels, 2 ** 2, kernel_size=ker_same, padding=pad, bias=False),
            BatchNorm2d(2 ** 2),
            LeakyReLU(),

            # Hidden layers:

            self.down1,

            self.down2,

            self.up1,

            self.up2,

            # Output layer

            Conv2d(2 ** 2, 1, kernel_size=ker_same, padding=pad, bias=False),
        )

    def forward(self, x):
        out = self.features(x)
        return out


class KerNet(nn.Module):
    """
    Cnn that shrinks by factor 8 and keeps the channel number the same.
    """
    def __init__(self):
        super(KerNet, self).__init__()
        # Doesn't change the size of the image
        self.filter_expand = nn.Sequential(
            DoubleConv(2 ** 0, 2 ** 2),
            DoubleConv(2 ** 2, 2 ** 4),
        )
        # shrinks transverse size by a factor of 2.
        self.shrink = nn.Sequential(
            Down(2 ** 4, 2 ** 6),
        )
        self.filter_flatten = nn.Sequential(
            DoubleConv(2 ** 6, 2 ** 3),
            nn.Conv2d(2 ** 3, 2 ** 0, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(2 ** 0),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(2 ** 0, 2 ** 0, kernel_size=(3, 3), padding=1),
        )

    def forward(self, x):
        expanded = self.filter_expand(x)
        shrunk = self.shrink(expanded)
        return self.filter_flatten(shrunk)



import model_tools
import torch.nn as nn
from torch.nn import Conv2d, ReLU, MaxPool2d, Linear, BatchNorm2d, LeakyReLU, Softmax
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
            nn.Dropout(0.1),
            nn.Conv2d(mid_channels, out_channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
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


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))

    def forward(self, x):
        return self.conv(x)


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


class UpCombine(nn.Module):
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

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class ActivationNet(nn.Module):
    """
    Takes a QPI measurement and returns the activation map (defect locations and intensity)
    """
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
            ReLU()
        )

    def forward(self, x_in):
        # x = (x_in - torch.min(x_in)) / torch.max(x_in)
        x = model_tools.normalize_tensor_0to1(x_in)
        x = self.features(x)
        x = model_tools.normalize_tensor_sumto1(x)
        return x


class ActivationResNet(nn.Module):
    def __init__(self, input_channels=1):
        super().__init__()
        self.input_channels = input_channels
        self.down1 = Down(2 ** 2, 2 ** 3)
        self.down2 = Down(2 ** 3, 2 ** 4 // 2)
        self.up1 = UpCombine(2 ** 4, 2 ** 3 // 2)
        self.up2 = UpCombine(2 ** 3, 2 ** 2)
        pad = 2  # These values make sure the image dimensions stay the same
        ker_same = (5, 5)
        self.input = nn.Sequential(
            Conv2d(input_channels, 2 ** 2, kernel_size=ker_same, padding=pad, bias=False),
            BatchNorm2d(2 ** 2),
            LeakyReLU(),
        )
        self.output = nn.Sequential(
            Conv2d(2 ** 2, 1, kernel_size=ker_same, padding=pad, bias=False),
            ReLU()
        )

    def forward(self, x_in):
        x = (x_in - torch.min(x_in)) / torch.max(x_in)
        x1 = self.input(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        x = self.output(x)
        x = model_tools.normalize_tensor_sumto1(x)
        return x


class ActivationSmiResNet(nn.Module):
    def __init__(self, input_channels=1):
        super().__init__()
        self.input_channels = input_channels
        self.down1 = Down(2 ** 2, 2 ** 3)
        self.down2 = Down(2 ** 3, 2 ** 4 // 2)
        self.up1 = UpCombine(2 ** 4, 2 ** 3)
        self.up2 = Up(2 ** 3, 2 ** 2)
        pad = 2  # These values make sure the image dimensions stay the same
        ker_same = (5, 5)
        self.input = nn.Sequential(
            Conv2d(input_channels, 2 ** 2, kernel_size=ker_same, padding=pad, bias=False),
            BatchNorm2d(2 ** 2),
            LeakyReLU(),
        )
        self.output = nn.Sequential(
            Conv2d(2 ** 2, 1, kernel_size=ker_same, padding=pad, bias=False),
            ReLU()
        )

    def forward(self, x_in):
        x = (x_in - torch.min(x_in)) / torch.max(x_in)
        x1 = self.input(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x)
        x = self.output(x)
        x = model_tools.normalize_tensor_sumto1(x)
        return x


class KerNet(nn.Module):
    """
    Cnn that shrinks by factor 8 and keeps the channel number the same.
    """

    def __init__(self):
        super(KerNet, self).__init__()
        # Doesn't change the size of the image
        self.filter_expand = nn.Sequential(
            DoubleConv(1, 2 ** 6),
        )
        # shrinks transverse size by a factor of 2.
        self.down1 = Down(2 ** 6, 2 ** 7)
        self.down2 = Down(2 ** 7, 2 ** 8)
        self.down3 = Down(2 ** 8, 2 ** 9)
        factor = 2
        self.up1 = Up(2 ** 9, 2 ** 8)
        self.up2 = Up(2 ** 8, 2 ** 7)
        self.out = OutConv(2 ** 7, 1)

    def forward(self, x):
        expanded = self.filter_expand(x)
        x = self.down1(expanded)
        x = self.down2(x)
        x = self.down3(x)
        x = self.up1(x)
        x = self.up2(x)
        return self.out(x)


class LISTA(nn.Module):
    def __init__(self, layer_num, iter_num=10):
        super(LISTA, self).__init__()
        self.iter_num = iter_num
        self.layer_num = layer_num
        pad = (2, 2)  # These values make sure the image dimensions stay the same
        ker_same = (5, 5)
        self.x_layers = nn.ModuleList()
        self.y_layers = nn.ModuleList()
        self.slu_layers = nn.ModuleList()
        for i in range(self.layer_num):
            self.x_layers.append(nn.Conv2d(1, 1, kernel_size=ker_same, padding=pad))
            self.y_layers.append(nn.Conv2d(1, 1, kernel_size=ker_same, padding=pad))
        self.relu = nn.ReLU()

    def forward(self, x):
        x = model_tools.normalize_tensor_0to1(x)
        y = torch.clone(x)
        for iteration in range(self.iter_num):
            for layer_idx in range(self.layer_num):
                x = self.relu(self.x_layers[layer_idx](x) + self.y_layers[layer_idx](y))
            x = model_tools.normalize_tensor_sumto1(x)
        return x

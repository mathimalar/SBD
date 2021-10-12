import torch
import torch.nn.functional as F
import torch.nn as nn


class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, relu_first=True):
        super().__init__()
        depthwise = nn.Conv2d(inplanes, inplanes, kernel_size,
                              stride=stride, padding=dilation,
                              dilation=dilation, groups=inplanes, bias=False)
        bn_depth = nn.BatchNorm2d(inplanes)
        pointwise = nn.Conv2d(inplanes, planes, 1, bias=False)
        bn_point = nn.BatchNorm2d(planes)

        if relu_first:
            self.block = nn.Sequential(OrderedDict([('relu', nn.ReLU()),
                                                    ('depthwise', depthwise),
                                                    ('bn_depth', bn_depth),
                                                    ('pointwise', pointwise),
                                                    ('bn_point', bn_point)
                                                    ]))
        else:
            self.block = nn.Sequential(OrderedDict([('depthwise', depthwise),
                                                    ('bn_depth', bn_depth),
                                                    ('relu1', nn.ReLU()),
                                                    ('pointwise', pointwise),
                                                    ('bn_point', bn_point),
                                                    ('relu2', nn.ReLU())
                                                    ]))

    def forward(self, x):
        return self.block(x)

        
class DecoderSPP(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(256, 48, 1, bias=False)
        self.bn = nn.BatchNorm2d(48)
        self.relu = nn.ReLU(inplace=True)
        self.sep1 = SeparableConv2d(304, 256, relu_first=False)
        self.sep2 = SeparableConv2d(256, 256, relu_first=False)

    def forward(self, x, low_level_feat):
        x = F.interpolate(x, size=low_level_feat.shape[2:], mode='bilinear', align_corners=True)
        low_level_feat = self.conv(low_level_feat)
        low_level_feat = self.bn(low_level_feat)
        low_level_feat = self.relu(low_level_feat)
        x = torch.cat((x, low_level_feat), dim=1)
        x = self.sep1(x)
        x = self.sep2(x)
        return x
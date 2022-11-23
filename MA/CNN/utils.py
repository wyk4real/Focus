import torch
import torch.nn as nn


class Down_Conv(nn.Module):
    def __init__(self, in_c, out_c):
        super(Down_Conv, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(inplace=True))

    def forward(self, x):
        return self.layer(x)


class Up_Conv(nn.Module):
    def __init__(self, in_c, out_c):
        super(Up_Conv, self).__init__()
        self.layer = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.layer(x)

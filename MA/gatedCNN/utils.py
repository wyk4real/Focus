import torch
import torch.nn as nn


class Gated_Conv(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, stride=1, rate=1, activation=nn.ELU()):
        super(Gated_Conv, self).__init__()
        padding = int(rate * (kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_c, 2 * out_c, kernel_size=kernel_size, stride=stride, padding=padding, dilation=rate)
        self.activation = activation

    def forward(self, x):
        raw = self.conv(x)
        x1 = raw.split(int(raw.shape[1] / 2), dim=1)
        gate = torch.sigmoid(x1[0])
        out = self.activation(x1[1]) * gate
        return out


class Down_Conv(nn.Module):
    def __init__(self, in_c, out_c):
        super(Down_Conv, self).__init__()
        self.layer = nn.Sequential(
            Gated_Conv(in_c, out_c, kernel_size=4, stride=2),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(inplace=True))

    def forward(self, x):
        return self.layer(x)


class Up_Conv(nn.Module):
    def __init__(self, in_c, out_c):
        super(Up_Conv, self).__init__()
        self.layer = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            Gated_Conv(in_c, out_c, kernel_size=3, stride=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.layer(x)


class Dis_Down_Conv(nn.Module):
    def __init__(self, in_c, out_c):
        super(Dis_Down_Conv, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(inplace=True))

    def forward(self, x):
        return self.layer(x)

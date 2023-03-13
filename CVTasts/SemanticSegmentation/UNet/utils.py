import torch
import torch.nn as nn


def _init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super(DoubleConv, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=(3, 3), padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, kernel_size=(3, 3), padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )
        self.apply(_init_weights)

    def forward(self, x):
        return self.layer(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.layer = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2)),
            DoubleConv(in_channels, out_channels)
        )
        self.apply(_init_weights)

    def forward(self, x):
        return self.layer(x)


class Up(nn.Module):
    def __init__(self, in_c, out_c):
        super(Up, self).__init__()
        self.layer = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_c, out_c, kernel_size=(1, 1), bias=False)
        )
        self.apply(_init_weights)

    def forward(self, x):
        return self.layer(x)

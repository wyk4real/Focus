import torch
from utils import *


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.inc = DoubleConv(1, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512)
        self.conv1 = DoubleConv(1024, 512)
        self.up2 = Up(512, 256)
        self.conv2 = DoubleConv(512, 256)
        self.up3 = Up(256, 128)
        self.conv3 = DoubleConv(256, 128)
        self.up4 = Up(128, 64)
        self.conv4 = DoubleConv(128, 64)
        self.outc = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(1, 1)),
            nn.Sigmoid())

    def forward(self, x):
        # encoding
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # decoding
        x = self.up1(x5)
        x = self.conv1(torch.cat((x4, x), dim=1))
        x = self.up2(x)
        x = self.conv2(torch.cat((x3, x), dim=1))
        x = self.up3(x)
        x = self.conv3(torch.cat((x2, x), dim=1))
        x = self.up4(x)
        x = self.conv4(torch.cat((x1, x), dim=1))
        return self.outc(x)


if __name__ == '__main__':
    input = torch.randn(1, 1, 512, 512)
    UNet = UNet()
    out = UNet(input)
    print(out.shape)

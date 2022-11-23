import torch
from utils import *


class CNN_G(nn.Module):
    def __init__(self):
        super(CNN_G, self).__init__()
        self.Avg_Pool = nn.AvgPool2d(kernel_size=2)
        self.Down_Conv1 = Down_Conv(in_c=1, out_c=64)
        self.Down_Conv2 = Down_Conv(in_c=65, out_c=128)
        self.Down_Conv3 = Down_Conv(in_c=129, out_c=256)
        self.Down_Conv4 = Down_Conv(in_c=257, out_c=512)
        self.Down_Conv5 = Down_Conv(in_c=513, out_c=1024)
        self.Up_Conv1 = Up_Conv(in_c=1025, out_c=512)
        self.Up_Conv2 = Up_Conv(in_c=1025, out_c=256)
        self.Up_Conv3 = Up_Conv(in_c=513, out_c=128)
        self.Up_Conv4 = Up_Conv(in_c=257, out_c=64)
        self.Up_Conv5 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=129, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.ReLU())

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def forward(self, image, mask):
        # Encoder
        i1 = self.Down_Conv1(image)  # torch.Size([1, 64, 256, 256])
        m1 = self.Avg_Pool(mask)  # torch.Size([1, 1, 256, 256])
        i2 = self.Down_Conv2(torch.cat((m1, i1), dim=1))  # torch.Size([1, 128, 128, 128])
        m2 = self.Avg_Pool(m1)  # torch.Size([1, 1, 128, 128])
        i3 = self.Down_Conv3(torch.cat((m2, i2), dim=1))  # torch.Size([1, 256, 64, 64])
        m3 = self.Avg_Pool(m2)  # torch.Size([1,  1, 64, 64])
        i4 = self.Down_Conv4(torch.cat((m3, i3), dim=1))  # torch.Size([1, 512, 32, 32])
        m4 = self.Avg_Pool(m3)  # torch.Size([1, 1, 32, 32])
        i5 = self.Down_Conv5(torch.cat((m4, i4), dim=1))  # torch.Size([1, 1024, 16, 16])
        m5 = self.Avg_Pool(m4)  # torch.Size([1, 1, 16, 16])
        # Decoder
        x = self.Up_Conv1(torch.cat((m5, i5), dim=1))  # torch.Size([1, 512, 32, 32])
        x = self.Up_Conv2(torch.cat((m4, i4, x), dim=1))  # torch.Size([1, 256, 64, 64])
        x = self.Up_Conv3(torch.cat((m3, i3, x), dim=1))  # torch.Size([1, 128, 128, 128])
        x = self.Up_Conv4(torch.cat((m2, i2, x), dim=1))  # torch.Size([1, 64, 256, 256])
        x = self.Up_Conv5(torch.cat((m1, i1, x), dim=1))  # torch.Size([1, 1, 512, 512])

        return mask * x + (1 - mask) * image


class CNN_D(nn.Module):
    def __init__(self):
        super(CNN_D, self).__init__()
        self.Avg_Pool = nn.AvgPool2d(kernel_size=2)
        self.Down_Conv1 = Dis_Down_Conv(in_c=1, out_c=64)
        self.Down_Conv2 = Dis_Down_Conv(in_c=64, out_c=128)
        self.Down_Conv3 = Dis_Down_Conv(in_c=128, out_c=256)
        self.Out_Conv = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1),
            nn.LeakyReLU())
        self.Activ = nn.Sigmoid()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def forward(self, image, mask):
        m1 = self.Avg_Pool(mask)  # torch.Size([1, 1, 256, 256])
        m2 = self.Avg_Pool(m1)  # torch.Size([1, 1, 128, 128])
        m3 = self.Avg_Pool(m2)  # torch.Size([1, 1, 64, 64])
        i1 = self.Down_Conv1(image)  # torch.Size([1, 64, 256, 256])
        i2 = self.Down_Conv2(i1)  # torch.Size([1, 128, 128, 128])
        i3 = self.Down_Conv3(i2)  # torch.Size([1, 256, 64, 64])
        i4 = self.Out_Conv(i3)  # torch.Size([1, 1, 64, 64])
        return self.Activ(m3 * i4)


if __name__ == '__main__':
    CNN_G = CNN_G()
    CNN_D = CNN_D()

    total_num = sum(p.numel() for p in CNN_G.parameters())
    print(total_num)
    trainable_num = sum(p.numel() for p in CNN_G.parameters() if p.requires_grad)
    print(trainable_num)

    image = torch.randn(1, 1, 512, 512)
    mask = torch.randn(1, 1, 512, 512)
    out1 = CNN_G(image, mask)
    out2 = CNN_D(out1, mask)

    print(out1.shape)
    print(out2.shape)

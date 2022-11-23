import torch
from utils import *


class ViT_G(nn.Module):
    def __init__(self, img_size=512, patch_size=16, in_c=1, embed_dim=512, nhead=8, num_layers=12):
        super(ViT_G, self).__init__()
        self.ViT = ViT(img_size=img_size,
                       patch_size=patch_size,
                       in_c=in_c,
                       embed_dim=embed_dim,
                       nhead=nhead,
                       num_layers=num_layers)
        self.In_Conv = nn.Conv2d(in_channels=embed_dim, out_channels=1024, kernel_size=3, stride=1, padding=1)
        self.Avg_Pool = nn.AvgPool2d(kernel_size=2)
        self.Up_Conv1 = Up_Conv(in_c=1025, out_c=512)
        self.Up_Conv2 = Up_Conv(in_c=513, out_c=256)
        self.Up_Conv3 = Up_Conv(in_c=257, out_c=128)
        self.Up_Conv4 = Up_Conv(in_c=129, out_c=64)
        self.Out_Conv = nn.Sequential(
            nn.Conv2d(in_channels=65, out_channels=1, kernel_size=3, stride=1, padding=1),
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
        B, C, H, W = image.shape
        x = self.ViT(image).view(B, self.ViT.grid_size, self.ViT.grid_size, -1).permute(0, 3, 1, 2)
        x = self.In_Conv(x)  # torch.Size([1, 1024, 32, 32])
        m1 = self.Avg_Pool(mask)  # torch.Size([1, 1, 256, 256])
        m2 = self.Avg_Pool(m1)  # torch.Size([1, 1, 128, 128])
        m3 = self.Avg_Pool(m2)  # torch.Size([1, 1, 64, 64])
        m4 = self.Avg_Pool(m3)  # torch.Size([1, 1, 32, 32])
        # Decoder
        x = self.Up_Conv1(torch.cat((m4, x), dim=1))  # torch.Size([1, 512, 64, 64])
        x = self.Up_Conv2(torch.cat((m3, x), dim=1))  # torch.Size([1, 256, 128, 128])
        x = self.Up_Conv3(torch.cat((m2, x), dim=1))  # torch.Size([1, 128, 256, 256])
        x = self.Up_Conv4(torch.cat((m1, x), dim=1))  # torch.Size([1, 64, 512, 512])
        x = self.Out_Conv(torch.cat((mask, x), dim=1))  # torch.Size([1, 1, 512, 512])

        return mask * x + (1 - mask) * image


class CNN_D(nn.Module):
    def __init__(self):
        super(CNN_D, self).__init__()
        self.Avg_Pool = nn.AvgPool2d(kernel_size=2)
        self.Down_Conv1 = Down_Conv(in_c=1, out_c=64)
        self.Down_Conv2 = Down_Conv(in_c=64, out_c=128)
        self.Down_Conv3 = Down_Conv(in_c=128, out_c=256)
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

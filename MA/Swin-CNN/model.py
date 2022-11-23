import torch
from utils import *


class Swin_G(nn.Module):
    def __init__(self, img_size=512, patch_size=4, in_c=1, embed_dim=128,
                 depths=[2, 4, 4], num_heads=[4, 8, 16], window_size=8,
                 mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 qkv_bias=True, norm_layer=nn.LayerNorm):
        super(Swin_G, self).__init__()
        self.Avg_Pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.PatchEmbed = PatchEmbed(img_size=img_size,
                                     patch_size=patch_size,
                                     in_c=in_c,
                                     embed_dim=embed_dim,
                                     norm_layer=norm_layer)
        self.patch_grid = self.PatchEmbed.grid_size
        # build encoder layers
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList([
            BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                       input_resolution=self.patch_grid // (2 ** i_layer),
                       depth=depths[i_layer],
                       num_heads=num_heads[i_layer],
                       window_size=window_size,
                       mlp_ratio=mlp_ratio,
                       qkv_bias=qkv_bias,
                       drop=drop_rate,
                       attn_drop=attn_drop_rate,
                       drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                       norm_layer=norm_layer)
            for i_layer in range(len(depths))])
        # build patch merging layer
        self.merging_layers = nn.ModuleList([
            PatchMerging(input_resolution=self.patch_grid // (2 ** i_layer),
                         dim=int(embed_dim * 2 ** i_layer),
                         norm_layer=norm_layer)
            for i_layer in range(len(depths) - 1)])
        # build connection layers
        self.connect_layers = nn.ModuleList([
            Reshape_Conv(input_resolution=self.patch_grid // (2 ** i_layer),
                         dim=int(embed_dim * 2 ** i_layer))
            for i_layer in range(len(depths))])
        # build decoder layers
        self.Conv_ReLU_BN1 = Conv_ReLU_BN(in_c=513, out_c=512)
        self.Conv_ReLU_BN2 = Conv_ReLU_BN(in_c=513, out_c=256)
        self.Conv_ReLU_BN3 = Conv_ReLU_BN(in_c=257, out_c=128)
        self.Conv_ReLU_BN4 = Conv_ReLU_BN(in_c=65, out_c=64)
        self.Conv_ReLU_BN5 = Conv_ReLU_BN(in_c=34, out_c=32)
        self.Up_Conv1 = Up_Conv(in_c=512, out_c=256)
        self.Up_Conv2 = Up_Conv(in_c=256, out_c=128)
        self.Up_Conv3 = Up_Conv(in_c=128, out_c=64)
        self.Up_Conv4 = Up_Conv(in_c=64, out_c=32)
        self.Out_Conv = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
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

    def forward_features(self, x):
        x = self.PatchEmbed(x)

        cnt_layer = []
        for inx, layer in enumerate(self.layers):
            x = layer(x)
            cnt = self.connect_layers[inx](x)
            cnt_layer.append(cnt)
            if inx > 1:
                break
            else:
                x = self.merging_layers[inx](x)

        return cnt_layer

    def forward(self, image, mask):
        m_1 = self.Avg_Pool(mask)  # torch.Size([1, 1, 256, 256])
        m_2 = self.Avg_Pool(m_1)  # torch.Size([1, 1, 128, 128])
        m_3 = self.Avg_Pool(m_2)  # torch.Size([1, 1, 64, 64])
        m_4 = self.Avg_Pool(m_3)  # torch.Size([1, 1, 32, 32])

        cnt_x = self.forward_features(x=image)

        x = self.Conv_ReLU_BN1(torch.cat([cnt_x[2], m_4], dim=1))  # torch.Size([1, 512, 32, 32])
        x = self.Up_Conv1(x)  # torch.Size([1, 256, 64, 64])
        x = self.Conv_ReLU_BN2(torch.cat([x, cnt_x[1], m_3], dim=1))  # torch.Size([1, 256, 64, 64])
        x = self.Up_Conv2(x)  # torch.Size([1, 128, 128, 128])
        x = self.Conv_ReLU_BN3(torch.cat([x, cnt_x[0], m_2], dim=1))  # torch.Size([1, 128, 128, 128])
        x = self.Up_Conv3(x)  # torch.Size([1, 64, 256, 256])
        x = self.Conv_ReLU_BN4(torch.cat([x, m_1], dim=1))  # torch.Size([1, 64, 256, 256])
        x = self.Up_Conv4(x)  # torch.Size([1, 32, 512, 512])
        x = self.Conv_ReLU_BN5(torch.cat([x, image, mask], dim=1))  # torch.Size([1, 32, 512, 512])
        x = self.Out_Conv(x)  # torch.Size([1, 1, 512, 512])

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

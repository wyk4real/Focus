import torch
import torch.nn as nn
from torchvision import models
from torch.nn import functional as F


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


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=16, in_c=1, embed_dim=512, norm_layer=nn.LayerNorm):
        super(PatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class ViT(nn.Module):
    def __init__(self, img_size=512, patch_size=16, in_c=1, embed_dim=512, nhead=8, num_layers=12):
        super(ViT, self).__init__()
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.dim_feedforward = 4 * embed_dim
        self.PatchEmbed = PatchEmbed(patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim,
                                                        nhead=nhead,
                                                        dim_feedforward=self.dim_feedforward,
                                                        activation=F.gelu,
                                                        batch_first=True,
                                                        norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=num_layers)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        x = self.PatchEmbed(x)
        x = x + self.pos_embed
        x = self.encoder(x)
        return x

import numpy as np
import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple


class Reshape_Conv(nn.Module):
    def __init__(self, input_resolution, dim):
        super(Reshape_Conv, self).__init__()
        self.input_resolution = to_2tuple(input_resolution)
        self.layer = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True))

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        x = x.permute(0, 2, 1).contiguous().view(B, C, H, W)
        return self.layer(x)


class Conv_ReLU_BN(nn.Module):
    def __init__(self, in_c, out_c):
        super(Conv_ReLU_BN, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True))

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


class Down_Conv(nn.Module):
    def __init__(self, in_c, out_c):
        super(Down_Conv, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(inplace=True))

    def forward(self, x):
        return self.layer(x)


def window_partition(x, window_size: int):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # permute: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H//Mh, W//Mh, Mw, Mw, C]
    # view: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B*num_windows, Mh, Mw, C]
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size: int, H: int, W: int):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    # view: [B*num_windows, Mh, Mw, C] -> [B, H//Mh, W//Mw, Mh, Mw, C]
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    # permute: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B, H//Mh, Mh, W//Mw, Mw, C]
    # view: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H, W, C]
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_c, embed_dim, norm_layer=None):
        super(PatchEmbed, self).__init__()
        self.img_size = to_2tuple(img_size)
        self.grid_size = img_size // patch_size
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]})."
        assert W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]})."
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super(PatchMerging, self).__init__()
        self.input_resolution = to_2tuple(input_resolution)
        self.norm = norm_layer(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        # cat: [B, H/2, W/2, C] -> [B, H/2, W/2, 4*C]
        # view: [B, H/2, W/2, 4*C] -> [B, H/2*W/2, 4*C]
        # reduction: [B, H/2*W/2, 4*C] -> [B, H/2*W/2, 2*C]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super(MLP, self).__init__()
        hidden_features = hidden_features if hidden_features else in_features
        out_features = out_features if out_features else in_features
        self.layer = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_features, out_features),
            nn.Dropout(drop))

    def forward(self, x):
        return self.layer(x)


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super(WindowAttention, self).__init__()
        self.window_size = to_2tuple(window_size)
        win_h, win_w = self.window_size
        self.window_area = win_h * win_w
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        # define a parameter table of relative position bias, shape: 2*Wh-1 * 2*Ww-1, nH
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * win_h - 1) * (2 * win_w - 1), num_heads))
        # get pair-wise relative position index for each token inside the window
        coords = torch.stack(torch.meshgrid([torch.arange(win_h), torch.arange(win_w)], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += win_h - 1
        relative_coords[:, :, 1] += win_w - 1
        relative_coords[:, :, 0] *= 2 * win_w - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

    def _get_rel_pos_bias(self) -> torch.Tensor:
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        return relative_position_bias.unsqueeze(0)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn + self._get_rel_pos_bias()
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size, shift_size=0, mlp_ratio=4., qkv_bias=True,
                 drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm):
        super(SwinTransformerBlock, self).__init__()
        self.input_resolution = to_2tuple(input_resolution)
        self.window_size = window_size
        self.shift_size = shift_size
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim=dim,
                                    window_size=window_size,
                                    num_heads=num_heads,
                                    qkv_bias=qkv_bias,
                                    attn_drop=attn_drop,
                                    proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(in_features=dim, hidden_features=int(mlp_ratio * dim), drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)
        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size, mlp_ratio=4.,
                 qkv_bias=True, drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm):
        super(BasicLayer, self).__init__()
        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x

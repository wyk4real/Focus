import torch
import torch.nn as nn
from math import exp
import torch.nn.functional as F
from torch.autograd import Variable


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, smooth=1e-8):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1 - dice


class Precision_and_Recall(nn.Module):
    def __init__(self):
        super(Precision_and_Recall, self).__init__()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, smooth=1e-8):
        inputs_inv, targets_inv = torch.logical_not(inputs), torch.logical_not(targets)
        true_pos = torch.logical_and(inputs, targets).sum()
        true_neg = torch.logical_and(inputs_inv, targets_inv).sum()
        false_pos = torch.logical_and(inputs, targets_inv).sum()
        false_neg = torch.logical_and(inputs_inv, targets).sum()
        Precision = true_pos / (true_pos + false_pos + smooth)
        Recall = true_pos / (true_pos + false_neg + smooth)
        return Precision, Recall


class L1_ROI(nn.Module):
    def __init__(self):
        super(L1_ROI, self).__init__()

    def forward(self, inputs: torch.Tensor, masks: torch.Tensor, targets: torch.Tensor):
        mae = F.l1_loss(inputs, targets, reduction='sum')
        roi = torch.sum(masks)
        return mae / roi, roi


class PSNR(nn.Module):
    def __init__(self):
        super(PSNR, self).__init__()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        max = torch.max(inputs) if torch.max(inputs) > torch.max(targets) else torch.max(targets)
        mse = torch.mean(torch.square(inputs - targets))
        psnr = 10 * torch.log10(max ** 2 / mse)
        return psnr


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(nn.Module):
    def __init__(self):
        super(SSIM, self).__init__()

    def forward(self, img1, img2, window_size=11, size_average=True):
        (_, channel, _, _) = img1.size()
        window = create_window(window_size, channel)
        if img1.is_cuda:
            window = window.cuda(img1.get_device())
        window = window.type_as(img1)
        return _ssim(img1, img2, window, window_size, channel, size_average)

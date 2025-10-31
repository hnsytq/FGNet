import torch.nn.functional as F
import torch
from math import exp
import torch.nn as nn
from einops import rearrange
from torch.autograd import Variable

from torch.nn.modules.loss import _Loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ir_loss (fused_result,input_ir ):
    a=fused_result-input_ir
    b=torch.square(fused_result-input_ir)
    c=torch.mean(torch.square(fused_result-input_ir))
    ir_loss=c
    return ir_loss

def vi_loss (fused_result , input_vi):
    vi_loss=torch.mean(torch.square(fused_result-input_vi))
    return vi_loss


def ssim_loss (fused_result,input_ir,input_vi ):
    ssim_loss=ssim(fused_result,torch.maximum(input_ir,input_vi))

    return ssim_loss


def gra_loss( fused_result,input_ir, input_vi):
    gra_loss =torch.norm( Gradient(fused_result)- torch.maximum(Gradient(input_ir), Gradient(input_vi)))
    return gra_loss

class gradient(nn.Module):
    def __init__(self, group_size=15):
        super(gradient, self).__init__()
        x_kernel = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        y_kernel = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
        x_kernel = torch.FloatTensor(x_kernel).unsqueeze(0).unsqueeze(0)
        y_kernel = torch.FloatTensor(y_kernel).unsqueeze(0).unsqueeze(0)
        x_kernel = x_kernel.repeat(1, group_size, 1, 1)
        y_kernel = y_kernel.repeat(1, group_size, 1, 1)
        self.x_weight = nn.Parameter(data=x_kernel, requires_grad=False)
        self.y_weight = nn.Parameter(data=y_kernel, requires_grad=False)

    def forward(self, input):
        _, c, _, _ = input.shape
        x_grad = torch.nn.functional.conv2d(input, self.x_weight, padding=1)
        y_grad = torch.nn.functional.conv2d(input, self.y_weight, padding=1)
        gradRes = torch.mean((x_grad + y_grad).float())
        return gradRes

def Gradient(x, group_num):
    gradient_model =gradient(group_num).to(device)
    g = gradient_model(x)
    return g

class FuseLoss(_Loss):
    def __init__(self, group_num, weights):
        super().__init__()

        self.gradient = gradient(group_num)
        self.weights = weights
        self.group_num = group_num

    def forward(self, fused, hsi):
        hsi_group = rearrange(hsi, 'b (gn gs) h w -> b gn gs h w', gn=self.group_num)
        gs = hsi_group.shape[2]
        hsi_sum = torch.nansum(hsi_group, dim=2, keepdim=False) / gs
        hsi_max = torch.max(hsi_group, dim=2, keepdim=False)[0]
        mse_ = vi_loss(fused, hsi_sum)
        ssim_ = 1 - ssim(fused, hsi_max)
        gra_ = torch.norm(Gradient(fused, hsi_group.shape[1]) - Gradient(hsi_max, hsi_group.shape[1]))
        total_ = self.weights[0] * mse_ + self.weights[1] * ssim_ + self.weights[2] * gra_
        return total_


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


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

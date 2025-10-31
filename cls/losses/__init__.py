

import torch.nn.functional as F
import torch
from math import exp
import torch.nn as nn
from einops import rearrange
from torch.nn.modules.loss import _Loss

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, window=None, val_range=None):
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)
    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2
    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)
    ret = ssim_map.mean()
    return 1-ret

class gradient(nn.Module):
    def __init__(self, group_size=15, device='cuda'):
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

def Gradient(x, group_num, device):
    gradient_model =gradient(group_num).to(device)
    g = gradient_model(x)
    return g

class FuseLoss(_Loss):
    def __init__(self, group_num, weights, device):
        super().__init__()

        self.gradient = gradient(group_num).to(device)
        self.weights = weights
        self.group_num = group_num

    def forward(self, fused, hsi_sum, hsi_max):
        # t_num = hsi.shape[1] % self.group_num
        # group_size = hsi.shape[1] // self.group_num
        #
        # former = hsi[:, :-(t_num+group_size), :, :]
        # latter = hsi[:, -(t_num+group_size):, :, :]
        #
        # former_group = rearrange(former, 'b (gn gs) h w -> b gn gs h w', gn=self.group_num-1)
        # gs = former_group.shape[2]
        # hsi_sum = torch.nansum(former_group, dim=2, keepdim=False) / gs
        # hsi_max = torch.max(former_group, dim=2, keepdim=False)[0]
        #
        # latter_mean = torch.nansum(latter, dim=1, keepdim=True) / latter.shape[1]
        # latter_max = torch.max(latter, dim=1, keepdim=True)[0]
        #
        # hsi_sum = torch.cat([hsi_sum, latter_mean], dim=1)
        # hsi_max = torch.cat([hsi_max, latter_max], dim=1)

        # hsi_group = rearrange(hsi, 'b (gn gs) h w -> b gn gs h w', gn=self.group_num)
        # gs = hsi_group.shape[2]
        # hsi_sum = torch.nansum(hsi_group, dim=2, keepdim=False) / gs
        # hsi_max = torch.max(hsi_group, dim=2, keepdim=False)[0]
        mse_ = vi_loss(fused, hsi_sum)
        ssim_ = ssim(fused, hsi_max)
        gra_ = torch.norm(self.gradient(fused) - self.gradient(hsi_max))
        total_ = self.weights[0] * mse_ + self.weights[1] * ssim_ + self.weights[2] * gra_
        # total_ = self.weights[0] * mse_ + self.weights[2] * gra_
        return total_

def KL(alpha, c):
    """
    Args:
        alpha: the Dirichlet of cls
        c: num of cls classes

    Returns: KL loss of cls

    """

    beta = torch.ones((1, c)).cuda()
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl


def cls_evidence_loss(p, alpha, c, current_epoch, annealing_epoch):
    """
    Args:
        p: label of cls result (bs, 1)
        alpha: the Dirichlet of cls (bs, c)
        c: classes of cls
        current_epoch: train_epoch (changing while train)
        annealing_epoch: set in config

    Returns:the overall loss of classification (ace_loss + lamda * kl_loss)

    """

    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    label = F.one_hot(p, num_classes=c)

    # ---- Loss 1. L_ace ---- #
    L_ace = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)

    # ---- Loss 2. L_KL ---- #
    annealing_coef = min(1, current_epoch / annealing_epoch)  # gradually increase from 0 to 1
    alp = E * (1 - label) + 1
    L_kL = annealing_coef * KL(alp, c)

    return torch.mean(L_ace + L_kL)


def fuse_loss(fused, hsi, group_size, weights):
    # fuse_loss = mse + ssim + grad
    hsi_group = rearrange(hsi, 'b (c gs) h w -> b c gs h w', gs=group_size)
    hsi_sum = torch.nansum(hsi_group, dim=2, keepdim=False) / group_size
    hsi_max = torch.max(hsi_group, dim=2, keepdim=False)[0]
    mse_ = vi_loss(fused, hsi_sum)
    ssim_ = ssim(fused, hsi_max)
    gra_ = torch.norm(Gradient(fused, hsi_group.shape[1]) - Gradient(hsi_max, hsi_group.shape[1]))
    total_ = weights[0] * mse_ + weights[1] * ssim_ + weights[2] * gra_
    return total_

if __name__ == '__main__':
    a = torch.randn((2, 60, 192, 192)).to(device)
    b = torch.randn((2, 15, 192, 192)).to(device)
    t_num, group_num = 0, 12
    former = a[:, :-(t_num + group_num), :, :]
    latter = a[:, -(t_num + group_num):, :, :]
    print(former.shape, latter.shape)
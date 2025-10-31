import numpy as np
import torch
from einops import rearrange
import random


def MSDS(imgs_path, mask_path, times=6, cls=False):
    len_imgs = len(imgs_path)

    for time in range(1, times):

        for i in range(len_imgs):
            imgs_path.append(imgs_path[i] + f'&{time}')
            if cls:
                mask_path.append(mask_path[i])
            else:
                mask_path.append(mask_path[i] + f'&{time}')

    return imgs_path, mask_path


def dataset_enlarge(group_num, max_dim, imgs_path, mask_path, cls=False):
    len_imgs = len(imgs_path)

    for dim in range(group_num * 2, max_dim, group_num):

        for i in range(len_imgs):
            imgs_path.append(imgs_path[i] + f'&{dim}')
            if not cls:
                mask_path.append(mask_path[i] + f'&{dim}')
            else:
                mask_path.append(mask_path[i])

    return imgs_path, mask_path


def main_selected_bands(x, group_num):
    s, h, w = x.shape

    if s % group_num == 0:
        x_sel, x_res = selected_bands(x, group_num)
    else:
        size_f = (s // group_num) * (group_num - 1)
        x_former = x[:size_f, :, :]
        x_latter = x[size_f:, :, :]
        xf_sel, xf_res = selected_bands(x_former, group_num - 1)
        xl_sel, xl_res = selected_bands(x_latter, 1)
        x_sel = np.concatenate((xf_sel, xl_sel), axis=0)
        x_res = np.concatenate((xf_res, xl_res), axis=0)

    decompose = np.concatenate((x_sel, x_res), axis=0)
    return decompose


def selected_bands(x, num_selected):
    s, h, w = x.shape

    # x_mean = np.repeat(x_mean, s, 0)
    x_group = rearrange(x, '(gn gs) h w -> gn gs h w', gn=num_selected)
    x_mean = np.mean(x_group, axis=1, keepdims=False)
    attn_x = rearrange(x_group, 'gn gs h w -> gn gs (h w)')
    attn_y = rearrange(x_group, 'gn gs h w -> gn (h w) gs')
    attn_logit = (attn_x @ attn_y)
    for i in range(s // num_selected):
        attn_logit[:, i, i] = 0
    attn_logit = np.sum(attn_logit, axis=1, keepdims=False)
    selected_index = np.argmax(attn_logit, axis=1, keepdims=False)
    selected = np.zeros((num_selected, h, w), x_group.dtype)
    for i in range(num_selected):
        selected[:] = x_group[:, selected_index[i]]
    # x_mean_reshape = np.resize(x_mean, (1, h * w))
    # x_reshape = np.resize(x, (s, h * w))
    # sim = (x_reshape @ x_mean_reshape.transpose(-1, -2))
    x_mean = x_mean - selected / (s // num_selected)
    return selected, x_mean
    # x_out = np.concatenate((selected, x_mean), axis=0)
    # # x_out = rearrange(x_out, 's c h w -> c s h w')
    # return x_out


def selected_bands_old(x, num_selected):
    s, h, w = x.shape

    # x_mean = np.repeat(x_mean, s, 0)
    x_group = rearrange(x, '(gn gs) h w -> gn gs h w', gn=num_selected)
    x_mean = np.mean(x_group, axis=1, keepdims=False)
    attn_x = rearrange(x_group, 'gn gs h w -> gn gs (h w)')
    attn_y = rearrange(x_group, 'gn gs h w -> gn (h w) gs')
    attn_logit = (attn_x @ attn_y)
    for i in range(s // num_selected):
        attn_logit[:, i, i] = 0
    attn_logit = np.sum(attn_logit, axis=1, keepdims=False)
    selected_index = np.argmax(attn_logit, axis=1, keepdims=False)
    selected = np.zeros((num_selected, h, w), x_group.dtype)
    for i in range(num_selected):
        selected[:] = x_group[:, selected_index[i]]
    # x_mean_reshape = np.resize(x_mean, (1, h * w))
    # x_reshape = np.resize(x, (s, h * w))
    # sim = (x_reshape @ x_mean_reshape.transpose(-1, -2))
    x_mean = x_mean - selected / (s // num_selected)
    x_out = np.concatenate((selected, x_mean), axis=0)
    # x_out = rearrange(x_out, 's c h w -> c s h w')
    return x_out


def spectral_down(hsi, dim, group_num, ensemble=False):

    if ensemble:
        c, h, w = hsi.shape
        hsi = rearrange(hsi, 'c h w -> h w c')
    else:
        h, w, c = hsi.shape
    hsi_reduce = np.zeros((h, w, dim), hsi.dtype)
    group_size = c // group_num
    assign = dim // group_num
    # counts_list = [assign] * group_num
    residual = dim - (assign * group_num)
    # print(counts_list)
    # add_idx = np.random.choice(list(range(group_num)), residual)
    # counts_list[add_idx] += 1
    # print(c, dim)
    band_begin, count_reduce = 0, 0
    for i in range(group_num):
        band_size = assign
        now_group = group_size
        if residual != 0:
            band_size += 1
            if band_size > now_group:
                now_group += 1
            residual -= 1
        if i == group_num - 1:
            now_group = c - band_begin
        # print(group_size, band_size)
        selected = random.sample(list(range(now_group)), band_size)
        for j in range(len(selected)):
            hsi_reduce[:, :, count_reduce] = hsi[:, :, selected[j] + band_begin]
            count_reduce += 1
        band_begin += group_size
    if ensemble:
        hsi_reduce = rearrange(hsi_reduce, 'h w c -> c h w')
    return hsi_reduce


def spectral_down_old(hsi, dim, group_num):
    h, w, c = hsi.shape
    total = c // group_num
    now = dim // group_num
    hsi_reduce = np.zeros((hsi.shape[0], hsi.shape[1], dim), hsi.dtype)
    count_reduce = 0
    for i in range(0, c, total):
        selected = random.sample([i for i in range(total)], now)
        for j in range(len(selected)):
            hsi_reduce[:, :, count_reduce] = hsi[:, :, selected[j] + i]
            count_reduce += 1
    return hsi_reduce


def hsi_aug(hsi):
    s, h, w = hsi.shape
    noise = np.random.normal(0, 0.1, size=hsi.size).reshape(s, h, w)
    hsi_aug = hsi + noise
    return hsi_aug


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    from scipy.io import loadmat
    hsi_path = 'D:/HSI_RS/WHU-Hi/Matlab_data_format/WHU-Hi-HanChuan/WHU_Hi_HanChuan.mat'
    gt_path = 'D:/HSI_RS/WHU-Hi/Matlab_data_format/WHU-Hi-HanChuan/WHU_Hi_HanChuan_gt.mat'
    train_path = 'D:/HSI_RS/WHU-Hi/Matlab_data_format/WHU-Hi-HanChuan/Training samples and test samples/Train300.mat'
    train = loadmat(train_path)
    print(train['Train300'].shape)
    print(np.unique(train['Train300']))
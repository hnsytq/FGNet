from torch.utils.data.dataset import Dataset

import numpy as np

from scipy.io import loadmat

import warnings
from utils import main_selected_bands
warnings.filterwarnings('ignore')
from einops import rearrange
import torch


from utils import spectral_down

def fuse_target_gen(hsi, group_num):
    t_num = hsi.shape[0] % group_num
    group_size = hsi.shape[0] // group_num

    former = hsi[:-(t_num + group_size), :, :]
    latter = hsi[-(t_num + group_size):, :, :]

    former_group = rearrange(former, '(gn gs) h w -> gn gs h w', gn=group_num - 1)
    gs = former_group.shape[1]
    hsi_sum = np.nansum(former_group, axis=1, keepdims=False) / gs
    hsi_max = np.max(former_group, axis=1, keepdims=False)

    latter_mean = np.nansum(latter, axis=0, keepdims=True) / latter.shape[0]
    latter_max = np.max(latter, axis=0, keepdims=True)

    hsi_sum = np.concatenate([hsi_sum, latter_mean], axis=0)
    hsi_max = np.concatenate([hsi_max, latter_max], axis=0)

    return hsi_sum, hsi_max

class RSDataset(Dataset):
    def __init__(self, hsi, label, group_num, patch_size=13, repeat_time=1, transform=None):
        super().__init__()
        self.hsi = hsi
        self.label = label - 1
        self.group_num = group_num
        self.patch_size = patch_size
        self.data_all_offset = np.zeros(
            (hsi.shape[0] + self.patch_size - 1, hsi.shape[1] + self.patch_size - 1, hsi.shape[2]))
        self.start = int((self.patch_size - 1) / 2)
        self.data_all_offset[self.start:hsi.shape[0] + self.start, self.start:hsi.shape[1] + self.start, :] \
            = hsi[:, :, :]
        x_pos, y_pos = np.nonzero(label)
        self.transform = transform

        self.indices = np.array([(x, y) for x, y in zip(x_pos, y_pos)])
        self.labels = [self.label[x, y] for x, y in self.indices]
        self.thre = len(self.indices)
        if repeat_time > 1:
            self.indices = np.tile(self.indices, (repeat_time, 1))
            self.labels = np.tile(self.labels, (repeat_time, 1)).reshape(self.indices.shape[0])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.indices[i]
        x1, y1 = x - self.patch_size // 2 + self.start, y - self.patch_size // 2 + self.start
        x2, y2 = x + self.patch_size // 2 + self.start + 1, y + self.patch_size // 2 + self.start + 1

        # data = np.zeros((self.patch_size, self.patch_size, self.hsi.shape[2]))
        # print(data.shape, self.data_all_offset[x1:x2, y1:y2, :].shape, x, y, self.patch_size)
        data = self.data_all_offset[x1:x2, y1:y2, :]

        label = self.label[x, y]
        if i > self.thre:
            re_dim = np.random.randint(2 * self.group_num, data.shape[2])
            data = spectral_down(data, re_dim, group_num=self.group_num)
        if self.transform is not None:
            data = self.transform(data)
        data = np.asarray(data.transpose((2, 0, 1)), dtype='float32')
        label = np.asarray(label, dtype='int64')
        img_mean, img_max = fuse_target_gen(data, self.group_num)
        img_mean = img_mean.astype(np.float32)
        img_max = img_max.astype(np.float32)
        data_decompose = main_selected_bands(data, self.group_num)
        data_tensor = torch.from_numpy(data_decompose)
        label = torch.from_numpy(label)
        return data_tensor, label, x, y, img_mean, img_max


from sklearn import model_selection


def main_Dataset_RS(args, transform):
    data_mat = loadmat(args.hsi_path)
    labels_mat = loadmat(args.label_path)
    hsi = data_mat[args.data_name]
    train_gt = labels_mat['train']
    val_gt = labels_mat['val']
    test_gt = labels_mat['test']

    # hsi_max = np.max(hsi)
    # hsi_min = np.min(hsi)
    # hsi = (hsi - hsi_min) / (hsi_max - hsi_min)
    args.group_num = hsi.shape[2] // 5
    args.MD_R = hsi.shape[2] // 5
    args.spectral_number = hsi.shape
    args.classes = len(np.unique(test_gt)) - 1
    train_ds = RSDataset(hsi, train_gt, args.group_num, args.patch_size, repeat_time=4, transform=transform)
    val_ds = RSDataset(hsi, val_gt, args.group_num, args.patch_size, repeat_time=1)
    test_ds = RSDataset(hsi, test_gt, args.group_num, args.patch_size)
    return train_ds, val_ds, test_ds


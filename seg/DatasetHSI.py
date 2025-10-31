from torch.utils.data.dataset import Dataset
import skimage.io
# from skimage.metrics import normalized_mutual_information
from sklearn.metrics import normalized_mutual_info_score
import numpy as np
import cv2
from scipy.io import loadmat
import os
# from argument import Transform
from spectral import *
from spectral import open_image
import random
import math
from scipy.ndimage import zoom
import warnings
# from utils import selected_bands, spectral_down, hsi_aug
from utils import main_selected_bands, spectral_down

warnings.filterwarnings('ignore')


class Dataset_HSI(Dataset):  #
    def __init__(self, img_paths, seg_paths=None,
                 cutting=None, transform=None,
                 group_num=12, dataset_name='MDC'):
        self.img_paths = img_paths
        self.seg_paths = seg_paths
        self.transform = transform
        self.cutting = cutting
        self.group_num = group_num
        self.dataset_name = dataset_name

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        mask_path = self.seg_paths[index]

        #         print(img_path)
        path_units = img_path.split('&')
        dim = 0
        # img_name_units = path_units[-1].split('&')
        if len(path_units) > 1:
            dim = int(path_units[1])

        img_path = path_units[0]
        mask_path = mask_path.split('&')[0]
        if self.dataset_name != 'ODSI':
            mask = cv2.imread(mask_path + '.png', 0) / 255
        else:
            mask = loadmat(mask_path)['mask']

        if img_path[-4:] == '.mat':
            img = loadmat(img_path)['hsi']
        else:
            img = envi.open(img_path)[:, :, :]

        if self.dataset_name == 'Organ':
            img_max = np.max(img)
            img_min = np.min(img)
            img = (img - img_min) / (img_max - img_min)

        if dim != 0:
            max_dim = img.shape[2]
            re_dim = np.random.randint(2 * self.group_num, max_dim)
            img = spectral_down(img, re_dim, group_num=self.group_num)
        # img = img[:, :, self.channels] if self.channels is not None else img

        if img.shape != mask.shape:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
            # mask = np.resize(mask, (img.shape[0], img.shape[1]))

        if self.transform is not None:
            img, mask = self.transform((img, mask))

        mask = mask.astype(np.uint8)
        if self.cutting is not None:
            while (1):
                xx = random.randint(0, img.shape[0] - self.cutting)
                yy = random.randint(0, img.shape[1] - self.cutting)
                patch_img = img[xx:xx + self.cutting, yy:yy + self.cutting]
                patch_mask = mask[xx:xx + self.cutting, yy:yy + self.cutting]
                if patch_mask.sum() != 0: break
            img = patch_img
            mask = patch_mask

        img = img[:, :, None] if len(img.shape) == 2 else img
        img = np.transpose(img, (2, 0, 1))

        mask = mask[None,].astype(np.float32)

        img = main_selected_bands(img, self.group_num)

        img = img.astype(np.float32)

        return img, mask

    def __len__(self):
        return len(self.img_paths)

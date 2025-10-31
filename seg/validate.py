import os
import torch

import argparse
import json
import numpy as np

from torch.utils.data import DataLoader

from local_utils.seed_everything import seed_reproducer

from tqdm import tqdm
from DatasetHSI import Dataset_HSI

import cv2

from local_utils.metrics import iou, dice
from FSINet import FGNet

import warnings

warnings.filterwarnings('ignore')

from medpy.metric import binary


def compute_HD95(predict, label, idx):
    if np.sum(predict) > 0 and np.sum(label) > 0:
        hd95 = binary.hd95(predict, label)
        return hd95
    else:
        print(f'{idx}: {np.sum(predict)}, {np.sum(label)}')
        return -1


def main(args):
    seed_reproducer(42)

    root_path = args.root_path
    dataset_hyper = args.dataset_hyper
    dataset_mask = args.dataset_mask
    dataset_divide = args.dataset_divide

    cutting = args.cutting

    worker = args.worker

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    images_root_path = os.path.join(root_path, dataset_hyper)
    mask_root_path = os.path.join(root_path, dataset_mask)

    with open(dataset_divide, 'r') as load_f:
        dataset_dict = json.load(load_f)

    device = torch.device(device)

    Miou = iou
    MDice = dice

    # For slide window operation in the validation stage
    def patch_index(shape, patchsize, stride):
        s, h, w = shape
        sx = (w - patchsize[1]) // stride[1] + 1
        sy = (h - patchsize[0]) // stride[0] + 1
        sz = (s - patchsize[2]) // stride[2] + 1

        for x in range(sx):
            xs = stride[1] * x
            for y in range(sy):
                ys = stride[0] * y
                for z in range(sz):
                    zs = stride[2] * z
                    yield slice(zs, zs + patchsize[2]), slice(ys, ys + patchsize[0]), slice(xs, xs + patchsize[1])

    if args.dataset_name != 'Organ':
        val_data_files = dataset_dict['val']
        val_mask_files = dataset_dict['val']
    else:
        val_data_files = dataset_dict['val_data']
        val_mask_files = dataset_dict['val_mask']

    val_images_path = [os.path.join(images_root_path, i) for i in val_data_files]
    val_masks_path = [os.path.join(mask_root_path, f'{i[:-4]}') for i in val_mask_files]

    val_db = Dataset_HSI(val_images_path, val_masks_path, cutting=None, transform=None,
                         group_num=args.group_num, dataset_name=args.dataset_name)
    val_loader = DataLoader(val_db, batch_size=1, shuffle=False, num_workers=worker)

    ham_parser = argparse.ArgumentParser()
    ham_parser.add_argument('--HAM_TYPE', '-ht', type=str, default=args.factor_method)
    ham_parser.add_argument('--MD_R', '-mdr', type=int, default=args.group_num)
    ham_args = ham_parser.parse_args()
    model = FGNet(group_num=args.group_num, base_feature=64, ham_args=ham_args)
    model.load_state_dict(torch.load(args.pretrained_path)).to(device)

    val_iou, val_dice, val_hd95, count_none = 0, 0, 0, 0

    save_img_dir = args.save_img_dir
    show_img = args.show_img
    if not os.path.exists(save_img_dir):
        os.makedirs(save_img_dir + '/png/')
        os.makedirs(save_img_dir + '/seg/')

    model.eval()

    for idx, sample in enumerate(tqdm(val_loader)):
        image, label = sample
        image = image.squeeze()
        spectrum_shape, shape_h, shape_w = image.shape
        stride_h, stride_w = shape_h - cutting, shape_w - cutting
        patch_idx = list(patch_index((spectrum_shape, shape_h, shape_w), (cutting, cutting, spectrum_shape),
                                     (stride_h, stride_w, 1)))
        num_collect = torch.zeros((shape_h, shape_w), dtype=torch.uint8).to(device)
        pred_collect = torch.zeros((shape_h, shape_w)).to(device)
        for i in range(0, len(patch_idx), 1):
            with torch.no_grad():
                _, output = model(torch.stack([image[x] for x in patch_idx[i:i + 1]]).to(device))
                output = output.squeeze(1)
            for j in range(output.size(0)):
                num_collect[patch_idx[i + j][1:]] += 1
                pred_collect[patch_idx[i + j][1:]] += output[j]

        out = pred_collect / num_collect.float()
        out[torch.isnan(out)] = 0

        out, label = out.cpu().detach().numpy()[None][None], label.cpu().detach().numpy()

        out = np.where(out > 0.5, 1, 0)
        label = np.where(label > 0.5, 1, 0)
        t_dice = MDice(out, label)

        dist = compute_HD95(out, label, idx)
        if dist < 0:
            dist = 0
            count_none += 1
        val_hd95 += dist

        val_dice = val_dice + t_dice
        val_iou = val_iou + Miou(out, label)
        image = image.cpu().detach().numpy()
        if show_img:
            out = np.where(out > 0.5, 255, 0)
            cv2.imwrite(save_img_dir + f'{idx}_{t_dice}.png', out[0, 0])
            png = cv2.normalize(image[0], None, 0, 255, cv2.NORM_MINMAX)
            cv2.imwrite(save_img_dir + f'{idx}_{t_dice}.png', png)

    val_iou = val_iou / (idx + 1)
    val_dice = val_dice / (idx + 1)
    val_hd95 = val_hd95 / (idx - count_none + 1)
    metrics_str = f'IoU: {val_iou}, Dice: {val_dice}, HD95: {val_hd95}'
    print(metrics_str)
    with open(args.save_img_dir + '/metrics.txt', 'w+') as f:
        f.write(metrics_str)
        f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    '''Dataset'''
    parser.add_argument('--dataset_name', '-dn', type=str, default='MDC')
    parser.add_argument('--root_path', '-r', type=str, default='/data1/chentq/MHSI Choledoch Dataset (Preprocessed '
                                                               'Dataset)/')
    parser.add_argument('--dataset_hyper', '-dh', type=str, default='MHSI')
    parser.add_argument('--dataset_mask', '-dm', type=str, default='Mask')
    parser.add_argument('--dataset_divide', '-dd', type=str, default='../dataset/mdc_train_val.json')
    parser.add_argument('--cutting', '-cut', default=192, type=int)

    '''Fusion'''
    parser.add_argument('--group_num', '-gn', type=int, default=12)
    parser.add_argument('--factor_method', '-fm', type=str, default='NMF')

    '''Misc'''
    # parser.add_argument('--device', '-dev', type=str, default='cuda:7')
    parser.add_argument('--worker', '-nw', type=int, default=4)
    parser.add_argument('--show_img', '-si', type=bool, default=False)
    parser.add_argument('--save_img_dir', '-sid', type=str, default='../checkpoint/mdc/')
    parser.add_argument('--pretrained_path', '-pp', type=str, default='../checkpoint/mdc/best_model.pth')

    args = parser.parse_args()
    main(args)

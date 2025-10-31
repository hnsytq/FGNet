import os
import torch
import torch.nn as nn
import argparse
import json
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp

from torch import optim
from torch.utils.data import DataLoader

from local_utils.tools import save_dict
from local_utils.seed_everything import seed_reproducer

from tqdm import tqdm
from DatasetHSI import Dataset_HSI
from argument import Transform
from local_utils.misc import AverageMeter
from local_utils.tools import EarlyStopping
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from local_utils.metrics import iou, dice
from FGNet import FGNet
from losses import FuseLoss

import warnings

warnings.filterwarnings('ignore')


def main(args):
    seed_reproducer(42)

    root_path = args.root_path
    dataset_hyper = args.dataset_hyper
    dataset_mask = args.dataset_mask
    dataset_divide = args.dataset_divide
    batch = args.batch
    lr = args.lr
    wd = args.wd

    epochs = args.epochs
    cutting = args.cutting

    worker = args.worker

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    images_root_path = os.path.join(root_path, dataset_hyper)
    mask_root_path = os.path.join(root_path, dataset_mask)

    with open(dataset_divide, 'r') as load_f:
        dataset_dict = json.load(load_f)

    # Data Augmentation
    transform = Transform(Rotate_ratio=0.2, Flip_ratio=0.2)
    # transform = None
    device = torch.device(device)

    if not os.path.exists(f'{args.output_folder}'):
        os.makedirs(f'{args.output_folder}')
    save_dict(os.path.join(f'{args.output_folder}', 'args.csv'), args.__dict__)

    dice_criterion = smp.losses.DiceLoss(eps=1., mode='binary', from_logits=False)
    bce_criterion = nn.BCELoss()
    fuse_criterion = FuseLoss(group_num=args.group_num, weights=args.fuse_weight)

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
        train_data_files = dataset_dict['train']
        val_data_files = dataset_dict['val']
        train_mask_files = dataset_dict['train']
        val_mask_files = dataset_dict['val']
    else:
        train_data_files = dataset_dict['train_data']
        val_data_files = dataset_dict['val_data']
        train_mask_files = dataset_dict['train_mask']
        val_mask_files = dataset_dict['val_mask']

    train_images_path = [os.path.join(images_root_path, i) for i in train_data_files]
    train_masks_path = [os.path.join(mask_root_path, f'{i[:-4]}') for i in train_mask_files]
    val_images_path = [os.path.join(images_root_path, i) for i in val_data_files]
    val_masks_path = [os.path.join(mask_root_path, f'{i[:-4]}') for i in val_mask_files]

    from utils import MSDS
    train_images_path, train_masks_path = MSDS(imgs_path=train_images_path,
                                               mask_path=train_masks_path,
                                               times=args.scale_time)

    train_db = Dataset_HSI(train_images_path, train_masks_path, cutting=cutting,
                           transform=transform, group_num=args.group_num, dataset_name=args.dataset_name)
    train_loader = DataLoader(train_db, batch_size=batch, shuffle=True, num_workers=worker)

    val_db = Dataset_HSI(val_images_path, val_masks_path, cutting=None, transform=None,
                         group_num=args.group_num, dataset_name=args.dataset_name)
    val_loader = DataLoader(val_db, batch_size=1, shuffle=False, num_workers=worker)

    model = FGNet(group_num=args.group_num, base_feature=64, ham_args=args).to(device)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=wd)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-8)

    # only record, we are not use early stop.
    early_stopping_val = EarlyStopping(patience=1000, verbose=True,
                                       path=os.path.join(f'{args.output_folder}',
                                                         f'best_fold.pth'))

    history = dict(epoch=[], LR=[], train_loss=[], train_iou=[], val_dice=[], val_iou=[], val_count=[])

    for epoch in range(epochs):
        train_losses = AverageMeter()
        val_losses = AverageMeter()
        train_iou, val_iou, val_dice, test_iou, test_dice = 0, 0, 0, 0, 0
        print('now start train ..')
        print('epoch {}/{}, LR:{}'.format(epoch + 1, epochs, optimizer.param_groups[0]['lr']))
        train_losses.reset()
        model.train()
        try:
            pass
            for idx, sample in enumerate(tqdm(train_loader)):
                image, label = sample
                image, label = image.to(device), label.to(device)
                fuse, out = model(image)
                loss = dice_criterion(out, label) + bce_criterion(out, label) + fuse_criterion(fuse, image)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_losses.update(loss.item())
                out = out.cpu().detach().numpy()
                label = label.cpu().detach().numpy()
                out = np.where(out > 0.5, 1, 0)
                label = np.where(label > 0.5, 1, 0)

                train_iou = train_iou + np.mean(
                    [Miou(out[b], label[b]) for b in range(len(out))])

            train_iou = train_iou / (idx + 1)

        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory, please reduce batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                return
            else:
                raise e

        print('now start evaluate ...')
        model.eval()
        val_losses.reset()
        for idx, sample in enumerate(tqdm(val_loader)):
            image, label = sample
            image = image.squeeze()
            spectrum_shape, shape_h, shape_w = image.shape
            stride_h, stride_w = shape_h - cutting, shape_w - cutting
            patch_idx = list(patch_index((spectrum_shape, shape_h, shape_w), (cutting, cutting, spectrum_shape),
                                         (stride_h, stride_w, 1)))
            num_collect = torch.zeros((shape_h, shape_w), dtype=torch.uint8).to(device)
            pred_collect = torch.zeros((shape_h, shape_w)).to(device)
            for i in range(0, len(patch_idx), batch):
                with torch.no_grad():
                    _, output = model(torch.stack([image[x] for x in patch_idx[i:i + batch]]).to(device))
                    output = output.squeeze(1)
                for j in range(output.size(0)):
                    num_collect[patch_idx[i + j][1:]] += 1
                    pred_collect[patch_idx[i + j][1:]] += output[j]

            out = pred_collect / num_collect.float()
            out[torch.isnan(out)] = 0

            out, label = out.cpu().detach().numpy()[None][None], label.cpu().detach().numpy()

            out = np.where(out > 0.5, 1, 0)
            label = np.where(label > 0.5, 1, 0)
            val_dice = val_dice + MDice(out, label)
            val_iou = val_iou + Miou(out, label)

        val_iou = val_iou / (idx + 1)
        val_dice = val_dice / (idx + 1)

        print(
            'epoch {}/{}\t LR:{}\t train loss:{}\t train_iou:{}\t val_dice:{}\t val_iou:{}\t' \
                .format(epoch + 1, epochs, optimizer.param_groups[0]['lr'], train_losses.avg, train_iou, val_dice,
                        val_iou, ))
        history['train_loss'].append(train_losses.avg)
        history['val_dice'].append(val_dice)
        history['val_iou'].append(val_iou)
        history['train_iou'].append(train_iou)

        history['epoch'].append(epoch + 1)
        history['LR'].append(optimizer.param_groups[0]['lr'])

        scheduler.step()
        early_stopping_val(-val_dice, model)
        history['val_count'].append(early_stopping_val.counter)

        if epoch + 1 == epochs:
            torch.save(model.state_dict(),
                       os.path.join(f'{args.output_folder}', f'final_{epoch}.pth'))

        if early_stopping_val.early_stop:
            print("Early stopping")
            break

        history_pd = pd.DataFrame(history)
        history_pd.to_csv(os.path.join(f'{args.output_folder}', f'log.csv'), index=False)
    history_pd = pd.DataFrame(history)
    history_pd.to_csv(os.path.join(f'{args.output_folder}', f'log.csv'), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    '''Dataset'''
    parser.add_argument('--dataset_name', '-dn', type=str, default='MDC')
    parser.add_argument('--root_path', '-r', type=str, default='path/MHSI Choledoch Dataset (Preprocessed '
                                                               'Dataset)/')
    parser.add_argument('--dataset_hyper', '-dh', type=str, default='MHSI')
    parser.add_argument('--dataset_mask', '-dm', type=str, default='Mask')
    parser.add_argument('--dataset_divide', '-dd', type=str, default='./mdc_train_val.json')
    parser.add_argument('--cutting', '-cut', default=192, type=int)

    '''Learning'''
    parser.add_argument('--batch', '-b', type=int, default=4)
    parser.add_argument('--lr', '-l', default=0.0003, type=float)
    parser.add_argument('--wd', '-w', default=5e-4, type=float)
    parser.add_argument('--epochs', '-e', type=int, default=100)

    '''Fusion'''
    parser.add_argument('--group_num', '-gn', type=int, default=12)
    parser.add_argument('--fuse_weight', '-fw', type=int, default=[1, 1, 1])
    parser.add_argument('--factor_method', '-fm', type=str, default='NMF')

    '''Misc'''
    parser.add_argument('--worker', '-nw', type=int, default=4)
    parser.add_argument('--output_folder', '-of', type=str, default='./checkpoint/mdc')

    parser.add_argument('--HAM_TYPE', '-ht', type=str, default='NMF')
    parser.add_argument('--MD_R', '-mdr', type=int, default=12)
    parser.add_argument('--scale_time', type=int, default=4)
    args = parser.parse_args()
    main(args)

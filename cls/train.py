import os
import torch
import torch.nn as nn
import argparse
import pandas as pd

from torch import optim
from torch.utils.data import DataLoader

from local_utils.tools import save_dict
from local_utils.seed_everything import seed_reproducer

from tqdm import tqdm

from local_utils.misc import AverageMeter
from local_utils.tools import EarlyStopping
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from FGNet import FGNet
from losses import FuseLoss
from DatasetHSI_CLS import main_Dataset_RS
from argument import Transform

import warnings

warnings.filterwarnings('ignore')


def main(args):
    seed_reproducer(42)

    batch = args.batch
    lr = args.lr
    wd = args.wd
    experiment_name = args.experiment_name
    output_path = args.output
    epochs = args.epochs
    worker = args.worker
    device = args.device
    transform = Transform(Rotate_ratio=0.2, Flip_ratio=0.2, cls=True)
    train_ds, val_ds, test_ds = main_Dataset_RS(args, transform)
    group_num = args.group_num
    classes = args.classes

    # Data Augmentation
    # transform = Transform(Rotate_ratio=0.2, Flip_ratio=0.2, cls=True)
    device = torch.device(device)

    if not os.path.exists(f'{output_path}/{experiment_name}'):
        os.makedirs(f'{output_path}/{experiment_name}')
    save_dict(os.path.join(f'{output_path}/{experiment_name}', 'args.csv'), args.__dict__)

    ce_criterion = nn.CrossEntropyLoss()
    fuse_criterion = FuseLoss(group_num=group_num, weights=[1, 1, 1], device=device)

    from utils import MSDS
    # train_images_path, train_label = MSDS(imgs_path=train_images_path, mask_path=train_label, cls=True)

    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=worker)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=worker)
    test_loader = DataLoader(test_ds, batch_size=batch, shuffle=False, num_workers=worker)

    model = FGNet(group_num=group_num, base_feature=256, ham_args=args, classes=classes, cls=True).to(device)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=wd)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-8)

    # only record, we are not use early stop.
    early_stopping_val = EarlyStopping(patience=1000, verbose=True,
                                       path=os.path.join(f'{output_path}/{experiment_name}',
                                                         f'best_fold_{experiment_name}.pth'))

    history = {'epoch': [], 'LR': [], 'train_loss': [], 'train_acc': [], 'val_acc': [], 'test_acc': [], 'val_count': []}

    for epoch in range(epochs):
        train_losses = AverageMeter()
        val_losses = AverageMeter()

        print('now start train ..')
        print('epoch {}/{}, LR:{}'.format(epoch + 1, epochs, optimizer.param_groups[0]['lr']))
        train_losses.reset()
        model.train()
        try:

            correct_train, total_train = 0, 0
            for idx, sample in enumerate(tqdm(train_loader)):
                image, label, _, _, image_mean, image_max = sample
                image_mean, image_max = image_mean.to(device), image_max.to(device)
                image, label = image.to(device), label.to(device)
                fuse, out = model(image)
                loss = fuse_criterion(fuse, image_mean, image_max) * 0.5 + ce_criterion(out, label)
    
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_losses.update(loss.item())

                _, predicted = torch.max(out.data, dim=1)
                total_train += label.size(0)
                correct_train += (predicted == label).sum()

            train_acc = correct_train / total_train
            train_acc = train_acc.cpu().detach().numpy()

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
        correct, total = 0, 0
        for idx, sample in enumerate(tqdm(val_loader)):
            image, label, _, _, _, _ = sample
            image, label = image.to(device), label.to(device)
            out = model(image, mode='val')
            _, predicted = torch.max(out.data, dim=1)
            total += label.size(0)
            correct += (predicted == label).sum()
        val_acc = correct / total
        val_acc = val_acc.cpu().detach().numpy()

        test_acc = 0.
        print(
            'epoch {}/{}\t LR:{}\t train loss:{}\t val_acc:{}\t test_acc:{}'.format(
                epoch + 1, epochs, optimizer.param_groups[0]['lr'], train_losses.avg, val_acc, test_acc))
        history['train_loss'].append(train_losses.avg)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['test_acc'].append(test_acc)

        history['epoch'].append(epoch + 1)
        history['LR'].append(optimizer.param_groups[0]['lr'])

        scheduler.step()
        early_stopping_val(-val_acc, model)
        history['val_count'].append(early_stopping_val.counter)

        if args.save_every_epoch:
            if (epoch + 1) % 5 == 0:
                torch.save(model.state_dict(),
                           os.path.join(f'{output_path}/{experiment_name}', f'middle_{epoch}.pth'))

        if epoch + 1 == epochs:
            torch.save(model.state_dict(),
                       os.path.join(f'{output_path}/{experiment_name}', f'final_{epoch}.pth'))

        if early_stopping_val.early_stop:
            print("Early stopping")
            break

        history_pd = pd.DataFrame(history)
        history_pd.to_csv(os.path.join(f'{output_path}/{experiment_name}', f'log.csv'), index=False)
    history_pd = pd.DataFrame(history)
    history_pd.to_csv(os.path.join(f'{output_path}/{experiment_name}', f'log.csv'), index=False)

    "test phase"
    model.load_state_dict(torch.load(f"{args.output}{args.experiment_name}/best_fold_{args.experiment_name}.pth"))
    from validate_utils import valid, output_metric
    p_t, t_t = valid(device, model, test_loader)
    OA, AA_mean, Kappa, AA = output_metric(t_t, p_t)
    print('Test \n', 'OA: ', OA, 'AA_mean: ', AA_mean, ' Kappa: ', Kappa, 'AA: ', AA)
    with open(f"{args.output}{args.experiment_name}/metrics.txt", 'w+') as f:
        f.write(f'Test \n, OA: , {OA}, AA_mean: , {AA_mean},  Kappa: , {Kappa}, AA: , {AA}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default='WHU_Hi_HanChuan')
    parser.add_argument('--hsi_path', type=str,
                        default='path/WHU_Hi_HanChuan/WHU_Hi_HanChuan.mat')
    parser.add_argument('--label_path', type=str,
                        default='path/WHU_Hi_HanChuan/HanChuan_split_gt.mat')
    parser.add_argument('--device', '-dev', type=str, default='cuda')

    parser.add_argument('--worker', '-nw', type=int,
                        default=4)

    parser.add_argument('--batch', '-b', type=int, default=128)

    parser.add_argument('--lr', '-l', default=0.0003, type=float)
    parser.add_argument('--wd', '-w', default=5e-4, type=float)

    parser.add_argument('--spectral_number', '-sn', default=274, type=int)
    parser.add_argument('--group_num', '-gn', default=50, type=int)
    parser.add_argument('--patch_size', default=17, type=int)

    parser.add_argument('--output', '-o', type=str, default='./checkpoints/')
    parser.add_argument('--experiment_name', '-name', type=str, default='WHU_Hi_HanChuan')
    parser.add_argument('--epochs', '-e', type=int, default=300)
    parser.add_argument('--init_values', '-initv', type=float, default=0.01)
    parser.add_argument('--save_every_epoch', '-see', default=False, action='store_true')
    parser.add_argument('--classes', '-c', default=16, type=int)
    parser.add_argument('--HAM_TYPE', '-ht', type=str, default='NMF')
    parser.add_argument('--MD_R', '-mdr', type=int, default=50)
    args = parser.parse_args()
    main(args)
    torch.cuda.empty_cache()

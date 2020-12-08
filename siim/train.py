from dataset import SIIMDataset
from unet import UnetOverResnet34, UnetOverResnet34Class, UnetOverResnet34SCSE
import torch.nn as nn
import torch.utils.data
import datetime
from datetime import timedelta
import sys
import time
import pandas as pd
import numpy as np
import pickle
from losses import FocalLoss, SoftDiceLoss, FocalLoss_2, BCELoss2d
from metrics import dice_overall, get_iou_vector
import albumentations as A
from lovash_loss import lovasz_hinge


def symmetric_lovasz(outputs, targets):
    return (lovasz_hinge(outputs, targets) + lovasz_hinge(-outputs, 1 - targets)) / 2


def set_lr(optimizer, new_lr):
    for i, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = new_lr


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def warm_up_lr(batch, num_batch_warm_up, init_lr, optimizer):
    for params in optimizer.param_groups:
        params['lr'] = batch * init_lr / num_batch_warm_up


AUGMENTATIONS_TRAIN = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomScale(scale_limit=0.2),
    A.Rotate(limit=(-15, 15)),
    A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=0.5),
    A.Resize(1024, 1024)
])


def print_one_line(s):
    time_string = datetime.datetime.now().strftime('%H:%M:%S')
    sys.stdout.write('\r' + time_string + ' ' + s)
    sys.stdout.flush()


csv_root = "/datasets/cell/SIIM/cats_w_label.csv"
data_root = "/datasets/cell/SIIM/img_1024/"
mask_root = "/datasets/cell/SIIM/msk_1024/"

batch_size = 32

df = pd.read_csv(csv_root)

# в категории 4 только 3 сэмпла, присвоим в 3ю
index = df[df['category'] == 4].index
df['category'].iloc[index] = 3


for j in [1, 2, 3, 4, 5]:
    print('FOLD:', j)
    with open(f'/code/siim_new/fold_{j}.pickle', 'rb') as f:
        idx = pickle.load(f)
    train_index = idx[0]
    val_index = idx[1]
    train_df, val_df = df.iloc[train_index], df.iloc[val_index]

    dataset_train = SIIMDataset(train_df, data_root, mask_root, transform=AUGMENTATIONS_TRAIN)
    dataset_val = SIIMDataset(val_df, data_root, mask_root)

    data_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True,
                                              num_workers=8)

    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=31, shuffle=False, num_workers=8)

    model = UnetOverResnet34()
    weights = torch.load(f'/datasets/cell/SIIM_unet34_512_{j}.pth')
    model.load_state_dict(weights)

    model = nn.DataParallel(model)
    model.cuda()

    lr = 1e-5

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    bce = BCELoss2d().cuda()
    prob = nn.Sigmoid()

    optimizer.zero_grad()
    it = 0
    batch_mult = 1
    val_dice = 0
    best_thr = 0
    best_epoch = 0

    NUM_EPOCH_WARM_UP = 10
    NUM_BATCH_WARM_UP = len(data_loader) * NUM_EPOCH_WARM_UP
    batch = 0
    for epoch in range(1, 1000):

        model.train()
        start_time = time.time()
        num_img = 0
        batch_idx = 0
        train_loss = 0

        for images, masks in data_loader:
            batch += 1

            optimizer.zero_grad()

            images = images.cuda()
            masks = masks.cuda()

            logit = model(images)

            loss = bce(logit, masks)
            # loss = symmetric_lovasz(logit, masks)

            train_loss += loss

            loss /= batch_mult
            loss.backward()

            optimizer.step()

            num_img += images.size(0)
            batch_idx += 1

            print_one_line('Epoch {} Loss {:.6f} | ({}/{})'.format(epoch, train_loss / batch_idx,
                                                                   num_img, len(data_loader) * batch_size))

            it += 1

        elapsed_time_secs = time.time() - start_time
        print('')
        print("Epoch took: %s secs (Wall clock time)" % timedelta(seconds=round(elapsed_time_secs)))
        print('')

        start_time = time.time()

        model.eval()
        preds = []
        mask = []
        dices = []
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.cuda()
                output = model(images)

                output = prob(output)
                output = output.detach().cpu().numpy()
                masks = masks.detach().cpu().numpy()

                preds.append(output)
                mask.append(masks)
        preds = np.concatenate(preds, axis=0)
        mask = np.concatenate(mask, axis=0)

        dices = []
        thrs = np.arange(0.3, 0.6, 0.1)
        for i in thrs:
            preds_m = (preds > i)
            dices.append(dice_overall(preds_m, mask).mean())
        dices = np.array(dices)

        print('EPOCH:', epoch, 'DICE:', dices.max(), 'THR:', thrs[dices.argmax()])
        print('')
        print(val_dice, best_epoch, best_thr)
        print('')
        elapsed_time_secs = time.time() - start_time
        print("Validation took: %s secs (Wall clock time)" % timedelta(seconds=round(elapsed_time_secs)))
        if dices.max() > val_dice:
            val_dice = dices.max()
            best_thr = thrs[dices.argmax()]
            best_epoch = epoch
            torch.save(model.module.state_dict(), f'/datasets/cell/SIIM_unet34_1024_{j}.pth')

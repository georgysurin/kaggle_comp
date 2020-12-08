import pandas as pd
from unet import UnetOverResnet34
import torch
from tqdm import tqdm
from dataset import SIIMDataset
import cv2
import numpy as np
import torch.nn as nn
from metrics import dice_overall
import pickle
import torch.utils.data


def mask2rle(img, width, height):
    rle = []
    lastColor = 0
    currentPixel = 0
    runStart = -1
    runLength = 0

    for x in range(width):
        for y in range(height):
            currentColor = img[x][y]
            if currentColor != lastColor:
                if currentColor == 255:
                    runStart = currentPixel
                    runLength = 1
                else:
                    rle.append(str(runStart))
                    rle.append(str(runLength))
                    runStart = -1
                    runLength = 0
                    currentPixel = 0
            elif runStart > -1:
                runLength += 1
            lastColor = currentColor
            currentPixel += 1

    return " ".join(rle)


csv_root = "/datasets/cell/SIIM/cats_w_label.csv"
data_root = "/datasets/cell/SIIM/img_1024/"
mask_root = "/datasets/cell/SIIM/msk_1024/"

sample_df = pd.read_csv("/datasets/cell/SIIM/sample_submit.csv")
df = pd.read_csv(csv_root)

sublist = []
test_path = "/datasets/cell/SIIM/images_1024/test_png/"

net = UnetOverResnet34()
net.cuda()
net.eval()

for index, row in tqdm(sample_df.iterrows(), total=len(sample_df)):
    image_id = row['ImageId']
    output = np.zeros((1, 1024, 1024))
    img = cv2.imread(test_path + image_id + '.png')

    img = img.transpose(2, 0, 1)
    img = (img - 127.5) / 128.0
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)
    img = torch.from_numpy(img).float()
    img = img.cuda()
    for j in [1, 2, 3, 4, 5]:
        with open(f'/code/siim_new/fold_{j}.pickle', 'rb') as f:
            idx = pickle.load(f)
        train_index = idx[0]
        val_index = idx[1]
        train_df, val_df = df.iloc[train_index], df.iloc[val_index]
        dataset_val = SIIMDataset(val_df, data_root, mask_root)
        val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=31, shuffle=False, num_workers=8)

        weights = torch.load(f'/datasets/cell/SIIM_unet34_1024_{j}.pth')

        net.load_state_dict(weights)

        net.eval()
        preds = []
        mask = []
        dices = []
        with torch.no_grad():
            for images, masks in tqdm(val_loader):
                images = images.cuda()
                output = net(images)

                output = nn.Sigmoid()(output)
                output = output.detach().cpu().numpy()
                masks = masks.detach().cpu().numpy()
                preds.append(output)
                mask.append(masks)
        preds = np.concatenate(preds, axis=0)
        mask = np.concatenate(mask, axis=0)

        dices = []
        thrs = np.arange(0.1, 1, 0.1)
        for i in thrs:
            preds_m = (preds > i)
            dices.append(dice_overall(preds_m, mask).mean())
        dices = np.array(dices)
        thr = thrs[dices.argmax()]
        pred = net(img)
        pred = nn.Sigmoid()(pred)
        pred = pred[0].detach().cpu().numpy()

        pred = np.asarray(pred > thr, dtype=np.uint8)
        if pred.sum() < 1000:
            pred[:] = 0
        output += pred

    output = np.asarray((output >= 3), dtype=np.uint8).squeeze()

    if output.max() > 0:
        rles = mask2rle(output.T * 255, 1024, 1024)

    else:
        rles = '-1'

    sublist.append([image_id, rles])

submission_df = pd.DataFrame(sublist, columns=sample_df.columns.values)
submission_df.loc[submission_df.EncodedPixels == ' ', 'EncodedPixels'] = '-1'
submission_df.to_csv("submission_unet34_1024_5fold_vouting.csv", index=False)

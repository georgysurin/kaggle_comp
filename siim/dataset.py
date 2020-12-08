import numpy as np
import torch
import torch.utils.data
import cv2


def preprocces_img_and_mask(img, mask):
    img = img.transpose(2, 0, 1)
    img = (img - 127.5) / 128.0
    img = img.astype(np.float32)

    mask = mask / 255.
    mask = mask.round().astype(np.float32)
    mask = np.expand_dims(mask, axis=0)

    return img, mask


class SIIMDataset(torch.utils.data.Dataset):
    def __init__(self, df, img_dir, mask_dir, transform=None):
        df = df[df['id'] != '1.2.276.0.7230010.3.1.4.8323329.21631.1517874435.266171.png']
        df = df[df['id'] != '1.2.276.0.7230010.3.1.4.8323329.4832.1517875185.36480.png']
        self.df = df
        self.image_dir = img_dir
        self.mask_dir = mask_dir
        self.augment = transform
        print(len(self.df))

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['id']
        img = cv2.imread(self.image_dir + img_name)
        mask = cv2.imread(self.mask_dir + img_name, cv2.IMREAD_GRAYSCALE)

        if self.augment is None:

            img, mask = preprocces_img_and_mask(img, mask)

        else:
            augmented = self.augment(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

            img, mask = preprocces_img_and_mask(img, mask)

        return torch.from_numpy(img).float(), torch.from_numpy(mask).float()

    def __len__(self):
        return len(self.df)

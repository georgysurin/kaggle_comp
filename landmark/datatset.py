import numpy as np
import cv2
import os
import torch
from torch.utils.data import Dataset
from sklearn import preprocessing

from utils import preprocess_image


class LandmarkDataset(Dataset):
    def __init__(self, df, MIN_SAMPLES_PER_CLASS=50):
        counts = df.landmark_id.value_counts()
        selected_classes = counts[counts >= MIN_SAMPLES_PER_CLASS].index
        num_classes = selected_classes.shape[0]
        print('classes with at least N samples:', num_classes)

        df = df.loc[df.landmark_id.isin(selected_classes)]
        self.id = np.array(df['id'])
        le = preprocessing.LabelEncoder()
        self.label = np.array(df['landmark_id'])
        self.label = le.fit_transform(self.label)

    def __getitem__(self, idx):
        id = self.id[idx]
        label = self.label[idx]
        image = cv2.imread('/home/iris/formemorte/landmark/train/' + os.path.join(id[0], id[1], id[2], id) + '.jpg')
        image = cv2.resize(image, (300, 300))
        image = preprocess_image(image)
        return torch.from_numpy(image).float(), torch.tensor(label).long()

    def __len__(self):
        return len(self.id)

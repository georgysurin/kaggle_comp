import os
import pandas as pd
import cv2
import numpy as np
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

df = pd.read_csv('/home/iris/formemorte/landmark/train.csv')
print(df.head())
label = np.array(df['landmark_id'])
# num_classes = len(np.unique(label))
print(np.unique(label))
label = le.fit_transform(label)
print(np.unique(label))
# mean_h, mean_w = 0, 0
# for i in range(len(df)):
#     id = df.iloc[i]['id']
#     img = cv2.imread('/home/iris/formemorte/landmark/train/' + os.path.join(id[0], id[1], id[2], id) + '.jpg')
#     height, width, _ = img.shape
#     mean_h += height
#     mean_w += width
#
# print(mean_h / len(df))
# print(mean_w / len(df))
# Mean height = 611.2653153808677
# Mean width =  735.6030237840642

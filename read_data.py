

# Data set -
# 
# "Leafsnap: A Computer Vision System for Automatic Plant Species Identification,"
# Neeraj Kumar, Peter N. Belhumeur, Arijit Biswas, David W. Jacobs, W. John Kress, Ida C. Lopez, João V. B. Soares,
# Proceedings of the 12th European Conference on Computer Vision (ECCV),
# October 2012

# ## creating training and test data

import numpy as np
import argparse
import glob
import pandas as pd
import os

path = '/dataset/images/field/'

filename = [x[0] for x in os.walk(path)]
target = []
for item in filename:
    item = item.split('/')
    target_name = item[-1]
    target.append(target_name)

target = target[1:]

dataframe = pd.DataFrame(columns=['ImgURL', 'label'])
Img_url_1 = []
target_label_1 = []
for item in target:
    ImgPath = path + item
    images = glob.glob(ImgPath + "/*.jpg")
    for ix in images:
        Img_url_1.append(ix)
    for i in range(len(images)):
        target_label_1.append(item)

dataframe['ImgURL'] = Img_url_1
dataframe['label'] = target_label_1

print dataframe.shape

# Handling categorical class labels
class_mapping = {label:idx for idx,label in 
                enumerate(np.unique(dataframe['label']))}


dataframe['label'] = dataframe['label'].map(class_mapping)

dataframe.to_csv('/home/rai5/Downloads/data.csv')






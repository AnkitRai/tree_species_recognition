
#creating training and test data
import glob
import os

import numpy as np
import pandas as pd

from convnet import config


path = config.INPUT_DATA # '/dataset/images/field/'

def load_data(path):

    print('[INFO]: Grabbing data..')

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

    # Handling categorical class labels
    class_mapping = {label:idx for idx,label in enumerate(np.unique(dataframe['label']))}

    dataframe['label'] = dataframe['label'].map(class_mapping)

    print('[INFO]: Writing data to csv..')

    dataframe.to_csv(config.OUTPUT_PATH)

    return None

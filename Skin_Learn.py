'''
ref: https://www.kaggle.com/vbookshelf/skin-lesion-analyzer-tensorflow-js-web-app
The flow of data preprocessing
The flow of training the model -- this is im
'''

from numpy.random import seed
seed(101)
from tensorflow import set_random_seed
set_random_seed(101)

import pandas as pd
import numpy as np
#import keras
#from keras import backend as K
import tensorflow
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import os
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import itertools
import shutil
import matplotlib.pyplot as plt

class SkinNet():

    def __init__(self):
        self.dataset_dir = '/home/mgs/PycharmProjects/Skin_DL/skin-cancer-mnist-ham10000/'
        self.base_dir = '/home/mgs/PycharmProjects/Skin_DL/skin_base/'
        self.train_dir = '/home/mgs/PycharmProjects/Skin_DL/skin_base/train_dir/'
        self.val_dir = '/home/mgs/PycharmProjects/Skin_DL/skin_base/val_dir/'

    def CreateFolders_TestTrain(self):

        nv = os.path.join(self.train_dir, 'nv')
        if os.path.isdir(nv):
            print("folder training nv is finished")
        else:
            os.mkdir(nv)

        mel = os.path.join(self.train_dir, 'mel')
        if os.path.isdir(mel):
            print("folder training mel is finished")
        else:
            os.mkdir(mel)

        bkl = os.path.join(self.train_dir, 'bkl')
        if os.path.isdir(bkl):
            print("folder training bkl is finished")
        else:
            os.mkdir(bkl)

        bcc = os.path.join(self.train_dir, 'bcc')
        if os.path.isdir(bcc):
            print("folder training bcc is finished")
        else:
            os.mkdir(bcc)

        akiec = os.path.join(self.train_dir, 'akiec')
        if os.path.isdir(akiec):
            print("folder training akiec is finished")
        else:
            os.mkdir(akiec)

        vasc = os.path.join(self.train_dir, 'vasc')
        if os.path.isdir(vasc):
            print("folder training vasc is finished")
        else:
            os.mkdir(vasc)

        df = os.path.join(self.train_dir, 'df')
        if os.path.isdir(df):
            print("folder training df is finished")
        else:
            os.mkdir(df)

        # Testing
        nv = os.path.join(self.val_dir, 'nv')
        if os.path.isdir(nv):
            print("folder testing nv is finished")
        else:
            os.mkdir(nv)

        mel = os.path.join(self.val_dir, 'mel')
        if os.path.isdir(mel):
            print("folder testing mel is finished")
        else:
            os.mkdir(mel)

        bkl = os.path.join(self.val_dir, 'bkl')
        if os.path.isdir(bkl):
            print("folder testing bkl is finished")
        else:
            os.mkdir(bkl)

        bcc = os.path.join(self.val_dir, 'bcc')
        if os.path.isdir(bcc):
            print("folder testing bcc is finished")
        else:
            os.mkdir(bcc)

        akiec = os.path.join(self.val_dir, 'akiec')
        if os.path.isdir(akiec):
            print("folder testing akiec is finished")
        else:
            os.mkdir(akiec)

        vasc = os.path.join(self.val_dir, 'vasc')
        if os.path.isdir(vasc):
            print("folder testing vasc is finished")
        else:
            os.mkdir(vasc)

        df = os.path.join(self.val_dir, 'df')
        if os.path.isdir(df):
            print("folder testing df is finished")
        else:
            os.mkdir(df)

    def CheckData(self):

        df_data = pd.read_csv(self.dataset_dir + 'HAM10000_metadata.csv')
        print(df_data.head())
        print(df_data.shape)
        df = df_data.groupby('lesion_id').count()
        df = df[df['image_id'] == 1]
        df.reset_index(inplace=True)
        print(df.head())
        print(df.shape)

    def plots(self, ims, figsize=(12, 6), rows=5, interp=False, titles=None):  # 12,6
        if type(ims[0]) is np.ndarray:
            ims = np.array(ims).astype(np.uint8)
            if (ims.shape[-1] != 3):
                ims = ims.transpose((0, 2, 3, 1))
        f = plt.figure(figsize=figsize)
        cols = len(ims) // rows if len(ims) % 2 == 0 else len(ims) // rows + 1
        for i in range(len(ims)):
            sp = f.add_subplot(rows, cols, i + 1)
            sp.axis('Off')
            if titles is not None:
                sp.set_title(titles[i], fontsize=16)
            plt.imshow(ims[i], interpolation=None if interp else 'none')

def CheckDuplicates(df_data, x):

    df_data['duplicates'] = df_data['lesion_id']
    unique_list = list(df_data['lesion_id'])
    df_data['duplicates'] = df_data['duplicates'].apply(identify_duplicates(unique_list))

    def identify_duplicates(x, unique_list):
        if x in unique_list:
            return 'no_duplicates'
        else:
            return 'has_duplicates'

if __name__ == "__main__":

    test = SkinNet()
    # Check the training and testing folders -- total 7 classes
    # test.CreateFolders_TestTrain()
    # Check the csv data file
    # test.CheckData()






















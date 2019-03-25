'''
ref: https://github.com/hoang-ho/Skin_Lesions_Classification_DCNNs/blob/master/Skin_Cancer_EDA.ipynb
ref: https://github.com/hoang-ho/Skin_Lesions_Classification_DCNNs
Algorithm 1: Train with data augmentation
    DataPreprocess: We need to save the data into the npy file
Algorithm 2: Train without data augmentation
    DataPreprocess: We need to save the data into the npy file
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from glob import glob
import seaborn as sns
from skimage.io import imread
from PIL import Image
from sklearn.model_selection import train_test_split
import os
from glob import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras import Model
from keras.applications.densenet import DenseNet201
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
import keras.backend as K
import matplotlib.pyplot as plt

class Skin_DataProcess_1():

    '''
    This is the second method for data processing
    load the data into an array but it takes time for reading -- .npy is a large file and thus it is not recommened
    Can be finished in the remote server maybe

    Algorithm:
    1. Load the data with tables
    2. read all the data into tables
    3. read the image array into tables
    '''

    def __init__(self):
        self.a = 1
        self.base_folder = "/home/mgs/PycharmProjects/Skin_DL/Skin_base_ModelTune/"

    def DataRead_1(self):

        # The first method for data reading
        # Read the skin_df with the csv file
        skin_df = pd.read_csv(os.path.join(self.base_folder, 'HAM10000_metadata.csv'))

        # Read the dictionary of the folder with all the image paths
        imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x
                             for x in glob(os.path.join(self.base_folder, '*', '*.jpg'))}

        print("All the image paths are ", imageid_path_dict)

        # Different types of lesion
        lesion_type_dict =  {
                            'nv': 'Melanocytic_nevi',
                            'mel': 'melanoma',
                            'bkl': 'Benign_keratosis-like_lesions',
                            'bcc': 'Basal_cell_carcinoma',
                            'akiec': 'Actinic_keratoses',
                            'vasc': 'Vascular_lesions',
                            'df': 'Dermatofibroma'
                            }

        # Different type of danger for the cancer -- this is a multi-task problem
        lesion_danger = {
                        'nv': 0,        # 0 for benign
                        'mel': 1,       # 1 for malignant
                        'bkl': 0,       # 0 for benign
                        'bcc': 1,       # 1 for malignant
                        'akiec': 1,     # 1 for malignant
                        'vasc': 0,
                        'df': 0
                        }

        # Map the image path and the cell type to the panda folder
        # Create a table to store all the information
        print("test 1")
        skin_df["path"] = skin_df["image_id"].map(imageid_path_dict.get)
        skin_df["cell_type"] = skin_df["dx"].map(lesion_type_dict.get)
        skin_df["Malignant"] = skin_df["dx"].map(lesion_danger.get)
        skin_df["cell_type_idx"] = pd.Categorical(skin_df["cell_type"]).codes  # give each cell a categorical ID
        print(skin_df.head())

        # Read the image data
        print("test 2")
        skin_df["image"] = skin_df["path"].map(imread)
        # print(skin_df["image"].map(lambda x: x.shape).value_counts())

        # Create the label and save as npy
        labels = skin_df["cell_type_idx"].values
        np.save('Data_labels.npy', labels)

        # Resize the image date
        # print("test 3")
        # reshaped_image = skin_df["path"].map(lambda x: np.asarray(Image.open(x).resize((256, 192), resample=Image.LANCZOS).convert("RGB")))
        # out_vec = np.stack(reshaped_image, 0)
        # print(out_vec.shape)

        # Svae the data into different format
        # print("test 4")
        # out_vec = out_vec.astype("float32")
        # out_vec /= 255

        # print("test 5")
        # np.save("/home/mgs/PycharmProjects/Skin_DL/Skin_base_ModelTune/Data_outvec.npy", out_vec)

    def DataRead_2(self):

        # Load the original data
        print("test 1")
        Data_org = np.load('/home/mgs/PycharmProjects/Skin_DL/Skin_base_ModelTune/Data_outvec_process.npy')

        print("test 2")
        Data_labels = np.load('/home/mgs/PycharmProjects/Skin_DL/Skin_base_ModelTune/Data_labels.npy')

        print("test 3")
        X_train_orig, X_test, y_train_orig, y_test = train_test_split(Data_org, Data_labels, test_size = 0.1, random_state = 0)

        print("test 4")
        np.save("256_192_X_train_orig.npy", X_train_orig)
        np.save("256_192_y_train_orig.npy", y_train_orig)

        # np.save("256_192_test.npy", X_test)
        # np.save("test_labels.npy", y_test)

    def DataRead_3(self):

        print("test 1")
        X_train_orig = np.load("/home/mgs/PycharmProjects/Skin_DL/Skin_base_ModelTune/256_192_X_train_orig.npy")
        y_train_orig = np.load("/home/mgs/PycharmProjects/Skin_DL/Skin_base_ModelTune/256_192_y_train_orig.npy")

        print("test 2")
        X_train, X_val, y_train, y_val = train_test_split(X_train_orig, y_train_orig, test_size = 0.1, random_state = 1)

        print("test 3")
        np.save("256_192_val.npy", X_val)
        np.save("val_labels.npy", y_val)

        print("test 4")
        np.save("256_192_train.npy", X_train)
        np.save("train_labels.npy", y_train)

    def Model_DenseNet(self):

        # Fine tune the DenseNet model
        X_train = np.load("/home/mgs/PycharmProjects/Skin_DL/Skin_base_ModelTune/256_192_train.npy")
        y_train = np.load("/home/mgs/PycharmProjects/Skin_DL/Skin_base_ModelTune/train_labels.npy")
        X_val = np.load("/home/mgs/PycharmProjects/Skin_DL/Skin_base_ModelTune/256_192_val.npy")
        y_val = np.load("/home/mgs/PycharmProjects/Skin_DL/Skin_base_ModelTune/val_labels.npy")

        print(X_train.shape)
        print(X_val.shape)
        print(y_train.shape)
        print(y_val.shape)

        # Load the pretrained model
        pre_trained_model = DenseNet201(input_shape=(192, 256, 3), include_top=False, weights="imagenet")
        for layer in pre_trained_model.layers:
            print(layer.name)
            if hasattr(layer, 'moving_mean') and hasattr(layer, 'moving_variance'):
                layer.trainable = True
                K.eval(K.update(layer.moving_mean, K.zeros_like(layer.moving_mean)))
                K.eval(K.update(layer.moving_variance, K.zeros_like(layer.moving_variance)))
            else:
                layer.trainable = False
        print(len(pre_trained_model.layers))

        # Last few layers
        last_layer = pre_trained_model.get_layer('relu')
        print('last layer output shape:', last_layer.output_shape)
        last_output = last_layer.output

        # Define the model
        # Flatten the output layer to 1 dimension
        x = layers.GlobalMaxPooling2D()(last_output)
        # Add a fully connected layer with 512 hidden units and ReLU activation
        x = layers.Dense(512, activation='relu')(x)
        # Add a dropout rate of 0.7
        x = layers.Dropout(0.5)(x)
        # Add a final sigmoid layer for classification
        x = layers.Dense(7, activation='softmax')(x)

        # Configure and compile the model
        model = Model(pre_trained_model.input, x)
        optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])

        # model.summary()

        # Training
        train_datagen = ImageDataGenerator(rotation_range=60, width_shift_range=0.2, height_shift_range=0.2,
                                           shear_range=0.2, zoom_range=0.2, fill_mode='nearest')
        train_datagen.fit(X_train)
        val_datagen = ImageDataGenerator()
        val_datagen.fit(X_val)

        batch_size = 32
        epochs = 3

        '''
        Data from numpy array (.npy data)
        '''
        history = model.fit_generator(train_datagen.flow(X_train, y_train, batch_size=batch_size),
                                      epochs=epochs, validation_data=val_datagen.flow(X_val, y_val),
                                      verbose=1, steps_per_epoch=(X_train.shape[0] // batch_size),
                                      validation_steps=(X_val.shape[0] // batch_size))

        '''
        We need more code here.
        '''

    def Test_1(self):

        out_vec = np.load('/home/mgs/PycharmProjects/Skin_DL/Skin_base_ModelTune/Data_outvec.npy')
        out_vec = out_vec.astype("float32")
        out_vec /= 255
        print(out_vec.shape)
        np.save('Data_outvec_process.npy', out_vec)

if __name__ == "__main__":

    test = Skin_DataProcess_1()
    # test.Test_1()
    # test.DataRead_1()
    # test.DataRead_2()
    # test.DataRead_3()
    test.Model_DenseNet()










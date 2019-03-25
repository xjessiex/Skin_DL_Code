import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from glob import glob
import sys
import shutil
from sklearn.model_selection import train_test_split
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout
from keras.models import Model
from keras.metrics import categorical_accuracy, top_k_categorical_accuracy
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

class Skin_Server():

    def __init__(self):
        print("Please create a new folder on your repository before you start with this program \n ")
        print("Remember to put '/' after the folder \n ")
        print("Please input three arguments for this case \n ")
        print("The first argv[1] is the folder path where the csv file locates \n ")
        print("The first argv[2] is the folder path where the base folder locates \n ")
        print("The first argv[3] is the folder path where the images source folder locates \n ")
        self.CsvDir = sys.argv[1]       # The folder where the csv file locates
        self.BaseDir = sys.argv[2]      # The base folder created by user
        self.src_folder = sys.argv[3]   # The folder where the images locate
        self.skin_df = pd.read_csv(os.path.join(self.CsvDir, 'HAM10000_metadata.csv'))

    def CreateFolder(self):

        # Create the image folder with different skin lesion type
        flag_createFolder = input("Do you want to create the initialized folder (1/0)?")
        base_dir = self.BaseDir
        self.train_dir = os.path.join(base_dir, 'train_dir')
        self.val_dir = os.path.join(base_dir, 'val_dir')
        self.test_dir = os.path.join(base_dir, 'test_dir')

        if flag_createFolder == '1':

            os.mkdir(self.train_dir)
            os.mkdir(self.val_dir)
            os.mkdir(self.test_dir)

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

            # Validation
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

            # Testing
            nv = os.path.join(self.test_dir, 'nv')
            if os.path.isdir(nv):
                print("folder testing nv is finished")
            else:
                os.mkdir(nv)

            mel = os.path.join(self.test_dir, 'mel')
            if os.path.isdir(mel):
                print("folder testing mel is finished")
            else:
                os.mkdir(mel)

            bkl = os.path.join(self.test_dir, 'bkl')
            if os.path.isdir(bkl):
                print("folder testing bkl is finished")
            else:
                os.mkdir(bkl)

            bcc = os.path.join(self.test_dir, 'bcc')
            if os.path.isdir(bcc):
                print("folder testing bcc is finished")
            else:
                os.mkdir(bcc)

            akiec = os.path.join(self.test_dir, 'akiec')
            if os.path.isdir(akiec):
                print("folder testing akiec is finished")
            else:
                os.mkdir(akiec)

            vasc = os.path.join(self.test_dir, 'vasc')
            if os.path.isdir(vasc):
                print("folder testing vasc is finished")
            else:
                os.mkdir(vasc)

            df = os.path.join(self.test_dir, 'df')
            if os.path.isdir(df):
                print("folder testing df is finished")
            else:
                os.mkdir(df)

    def DataProcess(self):

        # Load the dataset
        imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x
                             for x in glob(os.path.join(self.src_folder, '*.jpg'))}
        print(imageid_path_dict)

        # Different lesion type
        lesion_type_dict = {
            'nv': 'Melanocytic_nevi',
            'mel': 'melanoma',
            'bkl': 'Benign_keratosis-like_lesions',
            'bcc': 'Basal_cell_carcinoma',
            'akiec': 'Actinic_keratoses',
            'vasc': 'Vascular_lesions',
            'df': 'Dermatofibroma'
        }
        # Different danger of the lesion type
        lesion_danger = {
            'nv': 0, # 0 for benign
            'mel': 1, # 1 for malignant
            'bkl': 0, # 0 for benign
            'bcc': 1, # 1 for malignant
            'akiec': 1, # 1 for malignant
            'vasc': 0,
            'df': 0
        }
#
        # Read all the data into the skin_df folder
        self.skin_df = pd.read_csv(os.path.join(self.CsvDir, 'HAM10000_metadata.csv'))  # load in the data
        self.skin_df["path"] = self.skin_df["image_id"].map(imageid_path_dict.get)
        self.skin_df["path"] = self.skin_df["image_id"].map(imageid_path_dict.get)
        self.skin_df["cell_type"] = self.skin_df["dx"].map(lesion_type_dict.get)
        self.skin_df["Malignant"] = self.skin_df["dx"].map(lesion_danger.get)
        self.skin_df["cell_type_idx"] = pd.Categorical(self.skin_df["cell_type"]).codes
        self.skin_df.sample(3)

        # Define the label
        labels = self.skin_df["cell_type_idx"].values
        print(labels.shape)

        # Split the data into different sets
        X_train_orig, X_test, y_train_orig, y_test = train_test_split(self.skin_df, labels, test_size = 0.1,random_state = 0)
        X_train, X_val, y_train, y_val = train_test_split(X_train_orig, y_train_orig, test_size = 0.1, random_state = 1)

        print(X_train['dx'].shape)
        print(X_test['dx'].shape)
        print(X_val['dx'].shape)

        self.df_train = X_train
        self.df_test = X_test
        self.df_val = X_val

    def SaveImgToFolder(self):

        folder = os.listdir(self.src_folder)

        train_list = list(self.df_train['image_id'])
        val_list = list(self.df_val['image_id'])
        test_list = list(self.df_test['image_id'])

        print(train_list)

        self.skin_df.set_index('image_id', inplace=True)
        print(self.skin_df.head())
        print(self.skin_df.shape)

        # Transfer the train images
        flag_copyimg = input("Do you want to copy the images?")
        if flag_copyimg == '1':
            for image in train_list:
                # print(image)
                fname = image + '.jpg'
                label = self.skin_df.loc[image, 'dx']
                # print(label)
                # print(fname in folder)
                if fname in folder:
                    # source path to image
                    src = os.path.join(self.src_folder, fname)
                    # destination path to image
                    dst = os.path.join(self.train_dir, label, fname)
                    shutil.copyfile(src, dst)
                    print(dst)

            # Transfer the val images
            for image in val_list:
                fname = image + '.jpg'
                label = self.skin_df.loc[image, 'dx']
                if fname in folder:
                    # source path to image
                    src = os.path.join(self.src_folder, fname)
                    # destination path to image
                    dst = os.path.join(self.val_dir, label, fname)
                    # copy the image from the source to the destination
                    shutil.copyfile(src, dst)
                    print(dst)

            # Transfer the test images
            for image in test_list:
                fname = image + '.jpg'
                label = self.skin_df.loc[image, 'dx']
                if fname in folder:
                    # source path to image
                    src = os.path.join(self.src_folder, fname)
                    # destination path to image
                    dst = os.path.join(self.test_dir, label, fname)
                    # copy the image from the source to the destination
                    shutil.copyfile(src, dst)
                    print(dst)

    def ModelTrain(self):

        # Set up the generator
        train_path = self.BaseDir + 'train_dir'
        valid_path = self.BaseDir + 'val_dir'
        test_path = self.BaseDir + 'test_dir'

        # Set up the training parameters
        num_train_samples = len(self.df_train)
        num_val_samples = len(self.df_val)
        train_batch_size = 10
        val_batch_size = 10
        # image_size = 224
        img_h = 256
        img_w = 192
        train_steps = np.ceil(num_train_samples / train_batch_size)
        val_steps = np.ceil(num_val_samples / val_batch_size)

        # Set up the data generator
        datagen = ImageDataGenerator(preprocessing_function = keras.applications.mobilenet.preprocess_input,
                                    rotation_range=180,
                                    width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    zoom_range=0.1,
                                    horizontal_flip=True,
                                    vertical_flip=True,
                                    #brightness_range=(0.9,1.1),
                                    fill_mode='nearest')

        # Create the batches for training
        train_batches = datagen.flow_from_directory(train_path,
                                                    target_size = (img_h, img_w),
                                                    batch_size  = train_batch_size)
        valid_batches = datagen.flow_from_directory(valid_path,
                                                    target_size = (img_h, img_w),
                                                    batch_size  = val_batch_size)
        test_batches = datagen.flow_from_directory(test_path,
                                                    target_size = (img_h, img_w),
                                                    batch_size  = 1,
                                                    shuffle=False)

        # Load the pretrain model from keras.applications
        mobile = keras.applications.mobilenet.MobileNet()

        # Fine tune the model for the last few layers
        x = mobile.layers[-6].output
        x = Dropout(0.25)(x)
        predictions = Dense(7, activation='softmax')(x)
        model = Model(inputs = mobile.input, outputs = predictions)

        for layer in model.layers[:-23]:
            layer.trainable = False
        def top_3_accuracy(y_true, y_pred):
            return top_k_categorical_accuracy(y_true, y_pred, k=3)
        def top_2_accuracy(y_true, y_pred):
            return top_k_categorical_accuracy(y_true, y_pred, k=2)

        model.compile(Adam(lr=0.01), loss='categorical_crossentropy', metrics=[categorical_accuracy, top_2_accuracy, top_3_accuracy])
        print(valid_batches.class_indices)

        class_weights = {
                0: 1.0, # akiec
                1: 1.0, # bcc
                2: 1.0, # bkl
                3: 1.0, # df
                4: 3.0, # mel   # Try to make the model more sensitive to Melanoma.
                5: 1.0, # nv
                6: 1.0, # vasc
                        }

        filepath = self.BaseDir + "MobileNet.h5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_top_3_accuracy', verbose=1,
                                     save_best_only=True, mode='max')

        reduce_lr = ReduceLROnPlateau(monitor='val_top_3_accuracy', factor=0.5, patience=2,
                                           verbose=1, mode='max', min_lr=0.00001)

        callbacks_list = [checkpoint, reduce_lr]

        # Train the model
        history = model.fit_generator(train_batches, steps_per_epoch=train_steps,
                                      class_weight=class_weights,
                                      validation_data=valid_batches,
                                      validation_steps=val_steps,
                                      epochs = 30, verbose=1,
                                      callbacks=callbacks_list)
        model.save(filepath)

        # Test the model
        val_loss, val_cat_acc, val_top_2_acc, val_top_3_acc = model.evaluate_generator(test_batches, steps=len(self.df_val))
        print('val_loss:', val_loss)
        print('val_cat_acc:', val_cat_acc)
        print('val_top_2_acc:', val_top_2_acc)
        print('val_top_3_acc:', val_top_3_acc)
        print("The new model")

        model.load_weights(self.BaseDir + "MobileNet.h5")
        val_loss, val_cat_acc, val_top_2_acc, val_top_3_acc = model.evaluate_generator(test_batches, steps=len(self.df_val))
        print('val_loss:', val_loss)
        print('val_cat_acc:', val_cat_acc)
        print('val_top_2_acc:', val_top_2_acc)
        print('val_top_3_acc:', val_top_3_acc)

    def DataAug(self):

        # Data Augmentation for further application
        a = 1
if __name__ == "__main__":

    test = Skin_Server()
    test.CreateFolder()
    test.DataProcess()
    test.SaveImgToFolder()
    test.ModelTrain()
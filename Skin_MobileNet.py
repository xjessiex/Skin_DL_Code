'''
Convert the images to numpy file
ref: https://keras.io/preprocessing/image/
ref: https://github.com/keras-team/keras/blob/master/examples/cifar10_cnn.py
'''

# Basic package
from numpy.random import seed
from tensorflow import set_random_seed
import pandas as pd
import numpy as np
import os
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
# import itertools
import shutil
# import matplotlib.pyplot as plt
import tensorflow
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy
from keras.applications.densenet import DenseNet201
import keras
import keras.backend as K
from keras import layers
import sys
seed(101)
set_random_seed(101)

class Skin_DataProcess_2():

    def __init__(self):

        self.dataset_dir = sys.argv[1] # '/home/mgs/PycharmProjects/Skin_DL/skin-cancer-mnist-ham10000/'
        self.base_dir = sys.argv[2] # '/home/mgs/PycharmProjects/Skin_DL/skin_base/'

        # HAM10000_images_part_1.zip
        # /datacommons/plusds/skin_cancer
        # self.dataset_dir = '/home/mgs/PycharmProjects/Skin_DL/skin-cancer-mnist-ham10000/'
        # self.base_dir = '/home/mgs/PycharmProjects/Skin_DL/Skin_server_test/'

        # /home/mgs/PycharmProjects/Skin_DL/Skin_server_test

    def DataFolder(self):

        flag_createFolder = input("Do you want to create intialized folder (1/0)?")

        if flag_createFolder == '1':
            dataset_dir = self.dataset_dir
            base_dir = self.base_dir
            os.listdir(dataset_dir)

            train_dir = os.path.join(base_dir, 'train_dir')
            os.mkdir(train_dir)

            val_dir = os.path.join(base_dir, 'val_dir')
            os.mkdir(val_dir)

            nv = os.path.join(train_dir, 'nv')
            if os.path.isdir(nv):
                print("folder training nv is finished")
            else:
                os.mkdir(nv)

            mel = os.path.join(train_dir, 'mel')
            if os.path.isdir(mel):
                print("folder training mel is finished")
            else:
                os.mkdir(mel)

            bkl = os.path.join(train_dir, 'bkl')
            if os.path.isdir(bkl):
                print("folder training bkl is finished")
            else:
                os.mkdir(bkl)

            bcc = os.path.join(train_dir, 'bcc')
            if os.path.isdir(bcc):
                print("folder training bcc is finished")
            else:
                os.mkdir(bcc)

            akiec = os.path.join(train_dir, 'akiec')
            if os.path.isdir(akiec):
                print("folder training akiec is finished")
            else:
                os.mkdir(akiec)

            vasc = os.path.join(train_dir, 'vasc')
            if os.path.isdir(vasc):
                print("folder training vasc is finished")
            else:
                os.mkdir(vasc)

            df = os.path.join(train_dir, 'df')
            if os.path.isdir(df):
                print("folder training df is finished")
            else:
                os.mkdir(df)

            # Testing
            nv = os.path.join(val_dir, 'nv')
            if os.path.isdir(nv):
                print("folder testing nv is finished")
            else:
                os.mkdir(nv)

            mel = os.path.join(val_dir, 'mel')
            if os.path.isdir(mel):
                print("folder testing mel is finished")
            else:
                os.mkdir(mel)

            bkl = os.path.join(val_dir, 'bkl')
            if os.path.isdir(bkl):
                print("folder testing bkl is finished")
            else:
                os.mkdir(bkl)

            bcc = os.path.join(val_dir, 'bcc')
            if os.path.isdir(bcc):
                print("folder testing bcc is finished")
            else:
                os.mkdir(bcc)

            akiec = os.path.join(val_dir, 'akiec')
            if os.path.isdir(akiec):
                print("folder testing akiec is finished")
            else:
                os.mkdir(akiec)

            vasc = os.path.join(val_dir, 'vasc')
            if os.path.isdir(vasc):
                print("folder testing vasc is finished")
            else:
                os.mkdir(vasc)

            df = os.path.join(val_dir, 'df')
            if os.path.isdir(df):
                print("folder testing df is finished")
            else:
                os.mkdir(df)

    def DataRead(self):

        # Read the csv file
        df_data = pd.read_csv(self.dataset_dir + '/HAM10000_metadata.csv')
        df = df_data.groupby('lesion_id').count()
        df = df[df['image_id'] == 1]
        df.reset_index(inplace=True)

        print(df_data['lesion_id'].count())
        print(df_data.head())
        print(df.shape)

        # Check for the duplicate feature
        def identify_duplicates(x):
            unique_list = list(df['lesion_id'])
            # print("The unique_list is ", unique_list)
            if x in unique_list:
                return 'no_duplicates'
            else:
                return 'has_duplicates'

        # Check for the duplicates features and filter out non-duplicate features
        df_data['duplicates'] = df_data['lesion_id']
        df_data['duplicates'] = df_data['duplicates'].apply(identify_duplicates)
        df_data['duplicates'].value_counts()
        df = df_data[df_data['duplicates'] == 'no_duplicates']
        print(df_data.head())
        print(df.shape)

        # "dx" is a lesion type -- for validation set
        y = df['dx']
        _, df_val = train_test_split(df, test_size = 0.17, random_state = 101, stratify = y)
        print(df_val['dx'].value_counts())

        def identify_val_rows(x):
            # create a list of all the lesion_id's in the val set
            val_list = list(df_val['image_id'])
            if str(x) in val_list:
                return 'val'
            else:
                return 'train'

        df_data['train_or_val'] = df_data['image_id']
        df_data['train_or_val'] = df_data['train_or_val'].apply(identify_val_rows)
        df_train = df_data[df_data['train_or_val'] == 'train']
        df_data.set_index('image_id', inplace = True)

        print(len(df_train))
        print(len(df_val))
        print(df_train['dx'].value_counts())
        print(df_val['dx'].value_counts())

        # Get a list of images in each of the two folders
        folder_1 = os.listdir(self.dataset_dir + 'HAM10000_images_part_1')
        folder_2 = os.listdir(self.dataset_dir + 'HAM10000_images_part_2')
        print("The folder_1 is ", folder_1)
        print("The folder_2 is ", folder_2)

        # Get a list of train and val images -- a list of image paths
        train_list = list(df_train['image_id'])
        val_list = list(df_val['image_id'])
        print(train_list)
        print(val_list)

        flag_Createimg = input("Do you want to create the images? (1/0)")

        if flag_Createimg == '1':
            print("Create images in the folder")
            # Transfer the train images to a new folder
            train_dir = os.path.join(self.base_dir, 'train_dir')
            for image in train_list:
                fname = image + '.jpg'
                label = df_data.loc[image, 'dx']
                if fname in folder_1:
                    # source path to image
                    src = os.path.join(self.dataset_dir + 'HAM10000_images_part_1', fname)
                    # destination path to image
                    dst = os.path.join(train_dir, label, fname)
                    # copy the image from the source to the destination
                    shutil.copyfile(src, dst)
                if fname in folder_2:
                    # source path to image
                    src = os.path.join(self.dataset_dir + 'HAM10000_images_part_2', fname)
                    # destination path to image
                    dst = os.path.join(train_dir, label, fname)
                    # copy the image from the source to the destination
                    shutil.copyfile(src, dst)

            # Transfer the val images to a new folder -- this is im
            val_dir = os.path.join(self.base_dir, 'val_dir')
            for image in val_list:
                fname = image + '.jpg'
                label = df_data.loc[image, 'dx']
                if fname in folder_1:
                    # source path to image
                    src = os.path.join(self.dataset_dir + 'HAM10000_images_part_1', fname)
                    # destination path to image
                    dst = os.path.join(val_dir, label, fname)
                    # copy the image from the source to the destination
                    shutil.copyfile(src, dst)
                if fname in folder_2:
                    # source path to image
                    src = os.path.join(self.dataset_dir + 'HAM10000_images_part_2', fname)
                    # destination path to image
                    dst = os.path.join(val_dir, label, fname)
                    # copy the image from the source to the destination
                    shutil.copyfile(src, dst)

            # note that we are not augmenting class 'nv'
            class_list = ['mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']
            # Augmentation of the image data
            for item in class_list:
                # We are creating temporary directories here because we delete these directories later
                aug_dir = self.base_dir + '/aug_dir'
                os.mkdir(aug_dir)
                # create a dir within the base dir to store images of the same class
                img_dir = aug_dir + '/img_dir'
                os.mkdir(img_dir)
                # Choose a class
                img_class = item
                # list all images in that directory
                img_list = os.listdir(self.base_dir + '/train_dir/' + img_class)
                # Copy images from the class train dir to the img_dir e.g. class 'mel'
                for fname in img_list:
                    # source path to image
                    src = os.path.join(self.base_dir + '/train_dir/' + img_class, fname)
                    # destination path to image
                    dst = os.path.join(img_dir, fname)
                    # copy the image from the source to the destination
                    shutil.copyfile(src, dst)
                # point to a dir containing the images and not to the images themselves
                path = aug_dir
                save_path = self.base_dir + '/train_dir/' + img_class
                # Create a data generator
                datagen = ImageDataGenerator(
                        rotation_range=180,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        zoom_range=0.1,
                        horizontal_flip=True,
                        vertical_flip=True,
                        # brightness_range=(0.9,1.1),
                        fill_mode='nearest')
                batch_size = 50
                aug_datagen = datagen.flow_from_directory(path,
                                                          save_to_dir=save_path,
                                                          save_format='jpg',
                                                          target_size=(224, 224),
                                                          batch_size=batch_size)
                # Generate the augmented images and add them to the training folders
                num_aug_images_wanted = 6000  # total number of images we want to have in each class
                num_files = len(os.listdir(img_dir))
                num_batches = int(np.ceil((num_aug_images_wanted - num_files) / batch_size))
                # run the generator and create about 6000 augmented images
                for i in range(0, num_batches):
                    imgs, labels = next(aug_datagen)
                # delete temporary directory with the raw image files
                shutil.rmtree(aug_dir)

        return df_train, df_val

    def DataTrain_MobileNet(self):

        # Set up the generator
        train_path = self.base_dir + '/train_dir'
        valid_path = self.base_dir + '/val_dir'

        # Load the df training data
        df_train, df_val = self.DataRead()

        # The ratio between training and testing is close to 9:1
        num_train_samples = len(df_train)
        num_val_samples = len(df_val)
        train_batch_size = 10
        val_batch_size = 10
        image_size = 224
        train_steps = np.ceil(num_train_samples / train_batch_size)
        val_steps = np.ceil(num_val_samples / val_batch_size)
        print("The df_train is ", df_train.shape)
        print("The df_val is ", df_val.shape)
        print("The training step is ", train_steps)
        print("The validation step is ", val_steps)

        # Data generation and data from directory
        datagen = ImageDataGenerator(preprocessing_function = tensorflow.keras.applications.mobilenet.preprocess_input)
        train_batches = datagen.flow_from_directory(train_path,
                                                    target_size = (image_size, image_size),
                                                    batch_size = train_batch_size)
        valid_batches = datagen.flow_from_directory(valid_path,
                                                    target_size = (image_size, image_size),
                                                    batch_size = val_batch_size)
        print("The train batches are ", train_batches)
        print("The valid batches are ", valid_batches)

        mobile = tensorflow.keras.applications.mobilenet.MobileNet()
        # mobile.summary()

        ''' Create a new architecture
        1. Exclude the last 5 layers of the above model
        2. Create a new dense layer for predictions
        '''
        x = mobile.layers[-6].output
        x = Dropout(0.25)(x)
        predictions = Dense(7, activation='softmax')(x)
        model = Model(inputs = mobile.input, outputs = predictions)
        # model.summary()

        # The last 23 layers of the model will be trained.
        for layer in model.layers[:-23]:
            layer.trainable = False

        def top_3_accuracy(y_true, y_pred):
            return top_k_categorical_accuracy(y_true, y_pred, k = 3)

        def top_2_accuracy(y_true, y_pred):
            return top_k_categorical_accuracy(y_true, y_pred, k = 2)

        model.compile(Adam(lr=0.01), loss = 'categorical_crossentropy', metrics=[categorical_accuracy, top_2_accuracy, top_3_accuracy])
        print(valid_batches.class_indices)

        class_weights = {
            0: 1.0,  # akiec
            1: 1.0,  # bcc
            2: 1.0,  # bkl
            3: 1.0,  # df
            4: 3.0,  # mel # Try to make the model more sensitive to Melanoma.
            5: 1.0,  # nv
            6: 1.0,  # vasc
                        }

        filepath = self.base_dir + "model_MobileNet.h5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_top_3_accuracy', verbose=1,
                                     save_best_only=True, mode='max')
        reduce_lr = ReduceLROnPlateau(monitor='val_top_3_accuracy', factor=0.5, patience=2,
                                      verbose=1, mode='max', min_lr=0.00001)
        callbacks_list = [checkpoint, reduce_lr]
        history = model.fit_generator(train_batches,
                                      steps_per_epoch=train_steps,
                                      class_weight=class_weights,
                                      validation_data=valid_batches,
                                      validation_steps=val_steps,
                                      epochs=30, verbose=1,
                                      callbacks=callbacks_list)

    def DataTest_MobileNet(self):

        # Test the model from the MobileNet
        a = 1

    def DataTrain_DenseNet(self):

        # Set up the generator
        train_path = self.base_dir + '/train_dir'
        valid_path = self.base_dir + '/val_dir'

        # Load the df training data
        df_train, df_val = self.DataRead()

        print("check")
        # The ratio between training and testing is close to 9:1
        num_train_samples = len(df_train)
        num_val_samples = len(df_val)
        train_batch_size = 10
        val_batch_size = 10
        image_size = 224
        train_steps = np.ceil(num_train_samples / train_batch_size)
        val_steps = np.ceil(num_val_samples / val_batch_size)

        # Data generation and data from directory
        datagen = ImageDataGenerator(preprocessing_function = tensorflow.keras.applications.mobilenet.preprocess_input)
        train_batches = datagen.flow_from_directory(train_path,
                                                    target_size=(image_size, image_size),
                                                    batch_size=train_batch_size)
        valid_batches = datagen.flow_from_directory(valid_path,
                                                    target_size=(image_size, image_size),
                                                    batch_size=val_batch_size)

        # Load the pretrained model
        pre_trained_model = DenseNet201(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
        for layer in pre_trained_model.layers:
            print(layer.name)
            if hasattr(layer, 'moving_mean') and hasattr(layer, 'moving_variance'):
                layer.trainable = True
                K.eval(K.update(layer.moving_mean, K.zeros_like(layer.moving_mean)))
                K.eval(K.update(layer.moving_variance, K.zeros_like(layer.moving_variance)))
            else:
                layer.trainable = False

        # Last few layers
        last_layer = pre_trained_model.get_layer('relu')
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
        # model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        # The last 23 layers of the model will be trained.
        for layer in model.layers[:-23]:
            layer.trainable = False

        def top_3_accuracy(y_true, y_pred):
            return top_k_categorical_accuracy(y_true, y_pred, k=3)

        def top_2_accuracy(y_true, y_pred):
            return top_k_categorical_accuracy(y_true, y_pred, k=2)

        optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
        model.compile(loss='categorical_crossentropy',  optimizer = optimizer, metrics=[categorical_accuracy, top_2_accuracy, top_3_accuracy])
        class_weights = {
            0: 1.0,  # akiec
            1: 1.0,  # bcc
            2: 1.0,  # bkl
            3: 1.0,  # df
            4: 3.0,  # mel # Try to make the model more sensitive to Melanoma.
            5: 1.0,  # nv
            6: 1.0,  # vasc
        }

        filepath = self.base_dir + "model_DenseNet.h5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_top_3_accuracy', verbose=1,
                                     save_best_only=True, mode='max')
        reduce_lr = ReduceLROnPlateau(monitor='val_top_3_accuracy', factor=0.5, patience=2,
                                      verbose=1, mode='max', min_lr=0.00001)
        callbacks_list = [checkpoint, reduce_lr]
        history = model.fit_generator(train_batches,
                                      steps_per_epoch=train_steps,
                                      class_weight=class_weights,
                                      validation_data=valid_batches,
                                      validation_steps=val_steps,
                                      epochs = 30, verbose = 1,
                                      callbacks=callbacks_list)

if __name__ == "__main__":

    test = Skin_DataProcess_2()
    test.DataFolder()
    # test.DataRead()
    test.DataTrain_MobileNet()
    # test.DataTrain_DenseNet()

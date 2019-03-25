import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from glob import glob
import seaborn as sns
import sys

if __name__ == "__main__":

    DatasetDir = sys.argv[1] # '/home/mgs/PycharmProjects/Skin_DL/skin-cancer-mnist-ham10000/'
    CsvDir = DatasetDir
    BaseDir = sys.argv[2] # '/home/mgs/PycharmProjects/Skin_DL/test/'
    skin_df = pd.read_csv(os.path.join(CsvDir, 'HAM10000_metadata.csv')) # load in the data

    flag_createFolder = input("Do you want to create the initialized folder (1/0)?")

    if flag_createFolder == '1':
        dataset_dir = DatasetDir
        base_dir = BaseDir
        os.listdir(dataset_dir)

        train_dir = os.path.join(base_dir, 'train_dir')
        os.mkdir(train_dir)

        val_dir = os.path.join(base_dir, 'val_dir')
        os.mkdir(val_dir)

        test_dir = os.path.join(base_dir, 'test_dir')
        os.mkdir(test_dir)

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

        # Validation
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

        # Testing
        nv = os.path.join(test_dir, 'nv')
        if os.path.isdir(nv):
            print("folder testing nv is finished")
        else:
            os.mkdir(nv)

        mel = os.path.join(test_dir, 'mel')
        if os.path.isdir(mel):
            print("folder testing mel is finished")
        else:
            os.mkdir(mel)

        bkl = os.path.join(test_dir, 'bkl')
        if os.path.isdir(bkl):
            print("folder testing bkl is finished")
        else:
            os.mkdir(bkl)

        bcc = os.path.join(test_dir, 'bcc')
        if os.path.isdir(bcc):
            print("folder testing bcc is finished")
        else:
            os.mkdir(bcc)

        akiec = os.path.join(test_dir, 'akiec')
        if os.path.isdir(akiec):
            print("folder testing akiec is finished")
        else:
            os.mkdir(akiec)

        vasc = os.path.join(test_dir, 'vasc')
        if os.path.isdir(vasc):
            print("folder testing vasc is finished")
        else:
            os.mkdir(vasc)

        df = os.path.join(test_dir, 'df')
        if os.path.isdir(df):
            print("folder testing df is finished")
        else:
            os.mkdir(df)

    # This is a dictionary to read all the data
    imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x
                         for x in glob(os.path.join(CsvDir, '*', '*.jpg'))}

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

    # Data Processing
    skin_df["path"] = skin_df["image_id"].map(imageid_path_dict.get)
    skin_df["path"] = skin_df["image_id"].map(imageid_path_dict.get)
    skin_df["cell_type"] = skin_df["dx"].map(lesion_type_dict.get)
    skin_df["Malignant"] = skin_df["dx"].map(lesion_danger.get)
    skin_df["cell_type_idx"] = pd.Categorical(skin_df["cell_type"]).codes
    skin_df.sample(3)

    labels = skin_df["cell_type_idx"].values
    print(labels.shape)

    from sklearn.model_selection import train_test_split
    X_train_orig, X_test, y_train_orig, y_test = train_test_split(skin_df, labels, test_size = 0.1,random_state = 0)
    X_train, X_val, y_train, y_val = train_test_split(X_train_orig, y_train_orig, test_size = 0.1, random_state = 1)

    print(X_train['dx'].shape)
    print(X_test['dx'].shape)
    print(X_val['dx'].shape)

    df_train = X_train
    df_test = X_test
    df_val = X_val

    # Save the images to the folder -- this is im
    folder_1 = os.listdir(DatasetDir + 'HAM10000_images_part_1')
    folder_2 = os.listdir(DatasetDir + 'HAM10000_images_part_2')

    train_list = list(df_train['image_id'])
    val_list = list(df_val['image_id'])
    test_list = list(df_test['image_id'])

    import shutil

    skin_df.set_index('image_id', inplace=True)

    src1 = DatasetDir + 'HAM10000_images_part_1/'
    src2 = DatasetDir + 'HAM10000_images_part_2/'

    print(src1)
    print(src2)

    # Transfer the train images
    for image in train_list:
        fname = image + '.jpg'
        label = skin_df.loc[image, 'dx']
        if fname in folder_1:
            # source path to image
            src = os.path.join(src1, fname)
            # destination path to image
            dst = os.path.join(train_dir, label, fname)
            print(dst)
            # copy the image from the source to the destination
            shutil.copyfile(src, dst)
        if fname in folder_2:
            # source path to image
            src = os.path.join(src2, fname)
            # destination path to image
            dst = os.path.join(train_dir, label, fname)
            print(dst)
            # copy the image from the source to the destination
            shutil.copyfile(src, dst)

    # Transfer the val images
    for image in val_list:
        fname = image + '.jpg'
        label = skin_df.loc[image, 'dx']
        if fname in folder_1:
            # source path to image
            src = os.path.join(src1, fname)
            # destination path to image
            dst = os.path.join(val_dir, label, fname)
            # copy the image from the source to the destination
            shutil.copyfile(src, dst)
        if fname in folder_2:
            # source path to image
            src = os.path.join(src2, fname)
            # destination path to image
            dst = os.path.join(val_dir, label, fname)
            # copy the image from the source to the destination
            shutil.copyfile(src, dst)

    # Transfer the test images
    for image in test_list:
        fname = image + '.jpg'
        label = skin_df.loc[image, 'dx']
        if fname in folder_1:
            # source path to image
            src = os.path.join(src1, fname)
            # destination path to image
            dst = os.path.join(test_dir, label, fname)
            # copy the image from the source to the destination
            shutil.copyfile(src, dst)
        if fname in folder_2:
            # source path to image
            src = os.path.join(src2, fname)
            # destination path to image
            dst = os.path.join(test_dir, label, fname)
            # copy the image from the source to the destination
            shutil.copyfile(src, dst)

    # Set up the generator
    train_path = BaseDir + 'train_dir'
    valid_path = BaseDir + 'val_dir'
    test_path = BaseDir + 'test_dir'

    num_train_samples = len(df_train)
    num_val_samples = len(df_val)
    train_batch_size = 10
    val_batch_size = 10
    image_size = 224
    train_steps = np.ceil(num_train_samples / train_batch_size)
    val_steps = np.ceil(num_val_samples / val_batch_size)

    import keras
    from keras.preprocessing.image import ImageDataGenerator

    datagen = ImageDataGenerator(preprocessing_function = keras.applications.mobilenet.preprocess_input,
                                rotation_range=180,
                                width_shift_range=0.1,
                                height_shift_range=0.1,
                                zoom_range=0.1,
                                horizontal_flip=True,
                                vertical_flip=True,
                                #brightness_range=(0.9,1.1),
                                fill_mode='nearest')

    train_batches = datagen.flow_from_directory(train_path,
                                                target_size = (image_size,image_size),
                                                batch_size  = train_batch_size)

    valid_batches = datagen.flow_from_directory(valid_path,
                                                target_size = (image_size,image_size),
                                                batch_size  = val_batch_size)

    test_batches = datagen.flow_from_directory(test_path,
                                                target_size = (image_size,image_size),
                                                batch_size  = 1,
                                                shuffle=False)

    mobile = keras.applications.mobilenet.MobileNet()

    from keras.layers import Dense, Dropout
    from keras.models import Model
    x = mobile.layers[-6].output
    x = Dropout(0.25)(x)
    predictions = Dense(7, activation='softmax')(x)
    model = Model(inputs=mobile.input, outputs=predictions)
    for layer in model.layers[:-23]:
        layer.trainable = False

    from keras.metrics import categorical_accuracy, top_k_categorical_accuracy
    def top_3_accuracy(y_true, y_pred):
        return top_k_categorical_accuracy(y_true, y_pred, k=3)
    def top_2_accuracy(y_true, y_pred):
        return top_k_categorical_accuracy(y_true, y_pred, k=2)

    from keras.optimizers import Adam
    model.compile(Adam(lr=0.01), loss='categorical_crossentropy',
                  metrics=[categorical_accuracy, top_2_accuracy, top_3_accuracy])
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

    from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    filepath = BaseDir + "Mobile.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_top_3_accuracy', verbose=1,
                                 save_best_only=True, mode='max')

    reduce_lr = ReduceLROnPlateau(monitor='val_top_3_accuracy', factor=0.5, patience=2,
                                       verbose=1, mode='max', min_lr=0.00001)

    callbacks_list = [checkpoint, reduce_lr]
    history = model.fit_generator(train_batches, steps_per_epoch=train_steps,
                                  class_weight=class_weights,
                                  validation_data=valid_batches,
                                  validation_steps=val_steps,
                                  epochs=1, verbose=1,
                                  callbacks=callbacks_list)
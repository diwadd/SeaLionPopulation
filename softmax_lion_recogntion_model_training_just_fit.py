import os
import sys
import glob
import random
import math

import cv2
import numpy as np

import keras
from keras import backend as K
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split

import keras_recogntion_model_definitions as krmd
import data_handling_and_preparation as dhap


if __name__ == "__main__":
    print("Preparing to train model...")

    #data_dir = "/home/tadek/Coding/Kaggle/SeaLionPopulation/temp_data_24_24_24_24_24/"
    #model_filename = "/home/tadek/Coding/Kaggle/SeaLionPopulation/softmax_model_just_fit_24_24_24_24_24_49.h5"

    #data_dir = "/home/tadek/Coding/Kaggle/SeaLionPopulation/temp_data_28_28_28_16_12/"
    #model_filename = "/home/tadek/Coding/Kaggle/SeaLionPopulation/softmax_model_just_fit_28_28_28_16_12.h5"
    
    data_dir = "/home/tadek/Coding/Kaggle/SeaLionPopulation/temp_data_32_32_32_32_32/"
    model_filename = "/home/tadek/Coding/Kaggle/SeaLionPopulation/softmax_model_just_fit_28_28_28_16_12.h5"
    filename_list = glob.glob(data_dir + "*.npz")

    print(filename_list[0])


    loaded_data = np.load(filename_list[0])
    image = loaded_data["image"]
    labels = loaded_data["labels"]



    test_size = 0.3
    validation_size = 0.5

    random_state = random.randint(1, int(math.pow(2,32)))

    x_data, y_data = dhap.load_lion_files(filename_list, fraction=0.5, shuffle=1)


    # There are 16160 images of subadult males.
    # To have a uniform data distribution we limit the
    # data to a subset from 0 to 16160.
    # The data are sorted so we take all the subadult males
    # into account.
    #x_data = x_data[y_data[:,1].argsort()[::-1]]
    #y_data = y_data[y_data[:,1].argsort()[::-1]]

    #x_data = x_data[0:16160, :]
    #y_data = y_data[0:16160, :]

    #print(y_data[1:10,:])


    print("y_data.shape: " + str(y_data.shape))
    s = np.sum(y_data, axis=1)
    print("s: " + str(s))
    s = np.sum(y_data, axis=0)
    print("s: " + str(s))

    # Divide data in to a train set and a test set.
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_size, random_state=random_state)
    # A chunk of the test set will play the role of a validation set.
    x_validation, x_test, y_validation, y_test = train_test_split(x_test, y_test, test_size=validation_size, random_state=random_state)
    
    # ["RED", "MAGENTA", "BROWN", "BLUE", "GREEN"]
    #col_del = [1, 2, 3, 4] # red
    # col_del = [0, 2, 3, 4] # mag
    # col_del = [0, 1, 3, 4] # bro
    # col_del = [0, 1, 2, 4] # blu
    # col_del = [0, 1, 2, 3] # gre    
    #y_train = np.delete(y_train, col_del, 1)
    #y_validation = np.delete(y_validation, col_del, 1)
    #y_test = np.delete(y_test, col_del, 1)
    
    #y_train[y_train[:,0] > 0] = 1.0
    #y_validation[y_validation[:,0] > 0] = 1.0
    #y_test[y_test[:,0] > 0] = 1.0


    _, ih, iw, ic = x_test.shape
    _, nl = y_test.shape


    print("x_train.shape: %s" % (str(x_train.shape)))
    print("x_validation.shape: %s" % (str(x_validation.shape)))
    print("x_test.shape: %s" % (str(x_test.shape)))
    print("y_train.shape: %s" % (str(y_train.shape)))
    print("y_validation.shape: %s" % (str(y_validation.shape)))
    print("y_test.shape: %s" % (str(y_test.shape)))

    K.get_session()

    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=5.0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False) # randomly flip images


    #model = krmd.RecognitionNeuralNetworkModelSmall(ih, iw, ic, nl)
    model = krmd.RecognitionNeuralNetworkModelLarge49(ih, iw, ic, nl)

    batch_size = 100
    model.fit_generator(datagen.flow(x_train, y_train,
                           batch_size=batch_size),
                        steps_per_epoch=x_train.shape[0] // batch_size,
                        epochs=300,
                        validation_data=(x_validation, y_validation))

    model.save(model_filename)

    K.clear_session()









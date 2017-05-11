import os
import sys
import glob
import random
import math

import cv2
import numpy as np

from keras import backend as K
from keras.models import load_model

from sklearn.model_selection import train_test_split

import keras_recogntion_model_definitions as krmd
import data_handling_and_preparation as dhap


if __name__ == "__main__":
    print("Preparing to train model...")

    data_dir = "/home/tadek/Coding/Kaggle/SeaLionPopulation/temp_data_28_28_28_16_12/"
    model_filename = "/home/tadek/Coding/Kaggle/SeaLionPopulation/softmax_model_just_fit.h5"
    filename_list = glob.glob(data_dir + "*.npz")

    print(filename_list[0])


    loaded_data = np.load(filename_list[0])
    image = loaded_data["image"]
    labels = loaded_data["labels"]
    ih, iw, ic = image.shape
    _, nl = labels.shape


    test_size = 0.2
    validation_size = 0.5

    random_state = random.randint(1, int(math.pow(2,32)))

    x_data, y_data = dhap.load_lion_files(filename_list, fraction=1.0)
    # Divide data in to a train set and a test set.
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_size, random_state=random_state)
    # A chunk of the test set will play the role of a validation set.
    x_validation, x_test, y_validation, y_test = train_test_split(x_test, y_test, test_size=validation_size, random_state=random_state)
    

    print("x_train.shape: %s" % (str(x_train.shape)))
    print("x_validation.shape: %s" % (str(x_validation.shape)))
    print("x_test.shape: %s" % (str(x_test.shape)))
    print("y_train.shape: %s" % (str(y_train.shape)))
    print("y_validation.shape: %s" % (str(y_validation.shape)))
    print("y_test.shape: %s" % (str(y_test.shape)))

    K.get_session()

    model = krmd.RecognitionNeuralNetworkModelSmall(ih, iw, ic, nl)
    #model = krmd.RecognitionNeuralNetworkModelLarge(ih, iw, ic, nl)
    model.fit(x_train, y_train,
              epochs=100,
              batch_size=400,
              shuffle=True,
              validation_data=(x_validation, y_validation))

    model.save(model_filename)

    K.clear_session()









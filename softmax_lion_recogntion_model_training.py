import os
import sys
import glob
import random
import math

import cv2
import numpy as np

from keras import backend as K
from keras.models import load_model

import keras_recogntion_model_definitions as krmd
import data_handling_and_preparation as dhap

def load_single_lion_file(filename):
    """
    Load a single files that has been dispatched by
    softmax_like_approach.py

    """

    file_exists = os.path.exists(filename)
    if (file_exists == False):
        sys.exit("ERROR! The file path you provided does not exist!")

    loaded_data = np.load(filename)

    image = loaded_data["image"]
    labels = loaded_data["labels"]

    return image, labels


def load_lion_files(filename_list, fraction=1.0):

    n_files = len(filename_list)
    if (n_files == 0):
        sys.exit("ERROR: filename_list is empty.")

    image, labels = load_single_lion_file(filename_list[0])
    ih, iw, ic = image.shape
    _, nl = labels.shape

    x_data = np.zeros((n_files, ih, iw, ic))
    y_data = np.zeros((n_files, nl))

    x_data[0, :, :, :] = image
    y_data[0, :] = labels

    for n in range(1, int(fraction*n_files)):
        image, labels = load_single_lion_file(filename_list[n])

        x_data[n, :, :, :] = image
        y_data[n, :] = labels

    return x_data, y_data


def train_model(model,
                x_train_fnl,
                n_epochs,
                n_image_to_load_at_once,
                mini_batch_size):

    if (mini_batch_size >= n_image_to_load_at_once):
        sys.exit("ERROR: mini_batch_size has to smaller then n_image_to_load_at_once")

    number_of_image_loads = round(len(x_train_fnl) / n_image_to_load_at_once)
    print("Number of image loads per global epoch: %d" % (number_of_image_loads))

    for i in range(n_epochs):
        ptr = 0
        print("Global epoch: %d" % (i))
        for b in range(number_of_image_loads):
            progress = (b + 1)/number_of_image_loads
            #print("Batch: %f" % (progress), end="\r")
            mini_batch_fnl = x_train_fnl[ptr:(ptr + n_image_to_load_at_once)]
            ptr = ptr + n_image_to_load_at_once

            # Read a chunk of the data into RAM.
            x_data, y_data = load_lion_files(mini_batch_fnl)

            mn, nl = y_data.shape

            n_batches_per_load = round(mn / mini_batch_size)
            print("n_batches_per_load: " + str(n_batches_per_load))
            batch_ptr = 0
            for k in range(n_batches_per_load):
                print("Global Epoch: %d, image load: %d, processes image load %f, k: %d, processes mini batch %f" % (i, b, progress, k, (k+1)/n_batches_per_load))
                # A chunk of the chunk will we loaded with fit into the GPU.
                x_train = x_data[batch_ptr:(batch_ptr + mini_batch_size), :, :]
                y_train = y_data[batch_ptr:(batch_ptr + mini_batch_size), :]

                #print("x_train: " + str(x_train.shape))
                #print("y_train: " + str(y_train.shape))

                batch_ptr = batch_ptr + n_batches_per_load
                #model.train_on_batch(x_train, y_train)
                model.fit(x_train, 
                          y_train,
                          epochs=1,
                          batch_size=1,
                          shuffle=False)

        #print()

    return model



if __name__ == "__main__":
    print("Preparing to train model...")

    #data_dir = "/home/tadek/Coding/Kaggle/SeaLionPopulation/temp_data/"
    #model_filename = "/home/tadek/Coding/Kaggle/SeaLionPopulation/softmax_model.h5"
    
    data_dir = "/home/tadek/Coding/Kaggle/SeaLionPopulation/temp_data_32_32_32_32_32/"
    model_filename = "/home/tadek/Coding/Kaggle/SeaLionPopulation/softmax_model.h5"
    filename_list = glob.glob(data_dir + "*.npz")

    print(filename_list[0])


    loaded_data = np.load(filename_list[0])
    image = loaded_data["image"]
    labels = loaded_data["labels"]
    ih, iw, ic = image.shape

    print(labels.shape)
    _, nl = labels.shape


    K.get_session()

    #model = None
    #if (os.path.isfile(model_filename) == True):
    #    model = load_model(model_filename)
    #else:
    #model = krmd.TestRecognitionNeuralNetworkModel(ih, iw, ic, nl)
    #model = krmd.RecognitionNeuralNetworkModelTrainSmall2(ih, iw, ic, nl)
    model = krmd.RecognitionNeuralNetworkModelSmall(ih, iw, ic, nl)

    model = train_model(model,
                        filename_list,
                        n_epochs=10,
                        n_image_to_load_at_once=2000,
                        mini_batch_size=100)




    model.save(model_filename)

    K.clear_session()









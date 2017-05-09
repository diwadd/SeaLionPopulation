import os
import glob
import random
import pickle
import time
import sys

import numpy as np

from keras import backend as K
from keras.models import load_model
import keras_detection_model_definitions as kdmd

import data_handling_and_preparation as dhap
from data_handling_and_preparation import SAP

import working_directory_definition as wdd
from sklearn.model_selection import train_test_split


def train_model(model,
                x_train_fnl,
                n_epochs,
                n_image_to_load_at_once,
                mini_batch_size):

    if (mini_batch_size >= n_image_to_load_at_once):
        sys.exit("ERROR: mini_batch_size has to smaller then n_image_to_load_at_once")

    number_of_image_loads = round(len(x_train_fnl) / n_image_to_load_at_once)
    print("Number of batches per epoch: %d" % (number_of_image_loads))

    for i in range(n_epochs):
        ptr = 0
        print("Epoch: %d" % (i))
        for b in range(number_of_image_loads):
            progress = (b + 1)/number_of_image_loads
            #print("Batch: %f" % (progress), end="\r")
            mini_batch_fnl = x_train_fnl[ptr:(ptr + n_image_to_load_at_once)]
            ptr = ptr + n_image_to_load_at_once

            # Read a chunk of the data into RAM.
            x_data, y_data = dhap.load_lion_detection_files(mini_batch_fnl)

            mn, mh, mw = y_data.shape
            y_data = np.reshape(y_data, (mn, mh * mw))

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
                model.fit(x_train, y_train,
                          epochs=1,
                          batch_size=1,
                          shuffle=False)

        #print()

    return model

top_dir = os.path.dirname(os.path.abspath(__file__)) + "/"
directories = dhap.get_current_version_directory(top_dir)

detection_model_filename = directories["PARAMETERS_DIRECTORY"] + "detection_model.h5"


print("Data directory:\n " + str(directories["PREPROCESSED_DETECTION_DATA_DIRECTORY"]))
filename_list = glob.glob(directories["PREPROCESSED_DETECTION_DATA_DIRECTORY"] + "*.npz")

train_test_filenames = dhap.filename_list_train_test_split(filename_list)
#train_test_filenames = dhap.load_train_test_filenames_trainsmall2(fraction=0.2, data_type="detection")

# fnl - filename list
x_train_fnl = train_test_filenames[0]
x_validation_fnl = train_test_filenames[1]
x_test_fnl = train_test_filenames[2]

print("x_train len: %s" % (str(len(x_train_fnl))))
print("x_validation len: %s" % (str(len(x_validation_fnl))))
print("x_test len: %s" % (str(len(x_test_fnl))))

print("\n\n ===> ---- <=== \n\n")


f = open(directories["PARAMETERS_FILENAME"], "rb")
parameters = pickle.load(f)
f.close()


ih = parameters["resize_image_patch_to_h"]
iw = parameters["resize_image_patch_to_w"]
ic = 3 # number of channels in image

mh = parameters["resize_mask_patch_to_h"]
mw = parameters["resize_mask_patch_to_w"]

print("ih: %d, iw: %d, ic: %d, mh: %d, mw: %d" % (ih, iw, ic, mh, mw))


K.get_session()

model = None
if (os.path.isfile(detection_model_filename) == True):
    model = load_model(detection_model_filename)
else:
    model = kdmd.DetectionNeuralNetworkModelTrainSmall2(ih, iw, ic, mh, mw)


model = train_model(model,
                    x_train_fnl,
                    n_epochs=30,
                    n_image_to_load_at_once=2000,
                    mini_batch_size=100)




model.save(detection_model_filename)

K.clear_session()



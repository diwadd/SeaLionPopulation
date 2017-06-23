import os
import glob
import random
import pickle
import time
import sys
import math

import numpy as np

from keras import backend as K
from keras.models import load_model
import keras_direct_approach_model_definitions as kdmd

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
            x_data, y_data = dhap.load_lion_direct_approach_files(mini_batch_fnl)

            mn, mh = y_data.shape
            y_data = np.reshape(y_data, (mn, mh))

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


def custom_generator(file_name_list, ew, ev, noli=10):
    
    """
    noli - number of loaded images per yield
    
    """

    while True:
        n = len(file_name_list)
        number_of_image_loads = round(n / noli)
        ptr = 0
        # print("n: " + str(n))
        # print("number_of_image_loads: " + str(number_of_image_loads))

        for i in range(number_of_image_loads):
            # print("We are a i: " + str(i))
            # create numpy arrays of input data
            # and labels, from each line in the file
            mini_batch_fnl = file_name_list[ptr:(ptr + noli)]
            ptr = ptr + noli

            x_data, y_data = dhap.load_lion_direct_approach_files(mini_batch_fnl)

            data_size, _, _, _, = x_data.shape

            for j in range(data_size):
                x_data[j,:,:,:] = dhap.color_augmentation_of_an_image(x_data[j,:,:,:], ew, ev, ca_std=0.2)

            # print("y_data: " + str(y_data))
            yield (x_data, y_data)



def evaluate_model(file_name_list, model, noli=10):
    n = len(file_name_list)
    number_of_image_loads = round(n / noli)
    ptr = 0
    # print("n: " + str(n))
    # print("number_of_image_loads: " + str(number_of_image_loads))

    s = 0
    n_rows = 0
    for i in range(number_of_image_loads):
        # print("We are a i: " + str(i))
        # create numpy arrays of input data
        # and labels, from each line in the file
        mini_batch_fnl = file_name_list[ptr:(ptr + noli)]
        ptr = ptr + noli

        x_data, y_true = dhap.load_lion_direct_approach_files(mini_batch_fnl)
        y_pred = model.predict(x_data)

        #print("y_true:")
        #print(y_true)
        #print("y_pred:")
        #print(y_pred)

        local_loss = np.sqrt(np.sum((y_true - y_pred)*(y_true - y_pred))/y_true.size)
        #print("local_loss: " + str(local_loss))
        
        s = s + local_loss
        n_rows = n_rows + 1
    return s/n_rows



if (__name__ == "__main__"):
    top_dir = os.path.dirname(os.path.abspath(__file__)) + "/"
    directories = dhap.get_current_version_directory(top_dir)

    detection_model_filename = directories["PARAMETERS_DIRECTORY"] + "counting_model.h5"


    print("Data directory:\n " + str(directories["PREPROCESSED_DETECTION_DATA_DIRECTORY"]))
    filename_list = glob.glob(directories["PREPROCESSED_DETECTION_DATA_DIRECTORY"] + "*.npz")

    train_test_filenames = dhap.filename_list_train_test_split(filename_list)
    #train_test_filenames = dhap.load_train_test_filenames_trainsmall2(fraction=0.2, data_type="detection")


    # fnl - filename list
    x_train_fnl = train_test_filenames[0]
    x_validation_fnl = train_test_filenames[1]
    x_test_fnl = train_test_filenames[2]

    print("Number of files for training: " + str(len(x_train_fnl)))
    print("Number of files for validation: " + str(len(x_validation_fnl)))
    print("Number of files for tesring: " + str(len(x_test_fnl)))

    #data = custom_generator(x_train_fnl)
    #data = tuple(data)

    #x_data = data[0][0]
    #y_data = data[0][1]
    #print("x_data.shape " + str(x_data.shape))
    #print("y_data.shape " + str(y_data.shape))

    train_data_dir = directories["TRAIN_DATA_DIRECTORY"]
    nonprocessed_images = glob.glob(train_data_dir + "*.jpg")
    ew, ev = dhap.get_data_eigenvalues_and_eigenvectors(nonprocessed_images, fraction=1/50)
    print("ew: " + str(ew) + " ev: " + str(ev))

    f = open(directories["PARAMETERS_FILENAME"], "rb")
    parameters = pickle.load(f)
    f.close()

    ih = parameters["resize_image_patch_to_h"]
    iw = parameters["resize_image_patch_to_w"]
    ic = 3 # number of channels in image
    mh = dhap.CONST_NUMBER_OF_CLASSES

    K.get_session()

    retrain = True
    if (retrain == False):
        #model = kdmd.DetectionNeuralNetworkModelTrainSmall2(ih, iw, ic, mh)
        model = kdmd.ImageNetTransferModel()
    else:
        model = load_model(directories["PARAMETERS_DIRECTORY"] + "counting_model_min_imgnet.h5")


    noli = 10
    n = len(x_train_fnl)
    number_of_image_loads = round(n / noli)
    n_epochs = 250
    n_sub_epochs = 1
    # Total number of epoch is equal to n_epochs x n_sub_epochs.

    min_loss = math.inf
    for i in range(n_epochs):
        print("Progress: " + str((i + 1)/n_epochs))
        model.fit_generator(custom_generator(x_train_fnl, ew, ev, noli=noli),
                            steps_per_epoch=number_of_image_loads,
                            epochs=n_sub_epochs)

        # model = train_model(model,
        #            x_train_fnl,
        #            n_epochs=30,
        #            n_image_to_load_at_once=3000,
        #            mini_batch_size=400)


        #x_data, y_data = dhap.load_lion_direct_approach_files(x_test_fnl)
        print("Validating model on custom test data...")
        # loss = model.evaluate(x_data, y_data)
        loss = evaluate_model(x_validation_fnl, model, noli=noli)
        #loss = evaluate_model(x_train_fnl, model, noli=noli)

        if (loss < min_loss):
            min_loss = loss
            model.save(detection_model_filename.replace(".h5", "_min_imgnet.h5"))
            

        print("Loss: " + str(loss))

    model.save(detection_model_filename)

    K.clear_session()





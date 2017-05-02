import os

import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

import data_handling_and_preparation as dhap
import working_directory_definition as wdd


def TestNeuralNetworkModel(ih, iw, ic, mh, mw):
    """
    A simple model used to test the machinery.
    ih, iw, ic - describe the dimensions of the input image
    mh, mw - describe the dimensions of the output mask


    """

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(ih, iw, ic)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    #model.add(Dropout(0.5))

    model.add(Dense((mh * mw), activation="sigmoid"))

    model.compile(loss='binary_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    return model



train_test_data = dhap.load_train_test_data_trainsmall2(fraction=0.2)

x_train = train_test_data[0]
x_validation = train_test_data[1]
x_test = train_test_data[2]
y_train = train_test_data[3]
y_validation = train_test_data[4]
y_test = train_test_data[5]

print("x_train.shape: %s" % (str(x_train.shape)))
print("x_validation.shape: %s" % (str(x_validation.shape)))
print("x_test.shape: %s" % (str(x_test.shape)))
print("y_train.shape: %s" % (str(y_train.shape)))
print("y_validation.shape: %s" % (str(y_validation.shape)))
print("y_test.shape: %s" % (str(y_test.shape)))

print("\n\n\n ===> ---- <=== \n\n\n")

ni, ih, iw, ic = x_train.shape
print("Single image size: %d, %d, %d" % (ih, iw, ic))

nm_train, mh_train, mw_train = y_train.shape
nm_test, mh_test, mw_test = y_test.shape
print("Single mask size: %d, %d" % (mh_train, mw_train))

y_train = np.reshape(y_train, (nm_train, mh_train * mw_train))
y_test = np.reshape(y_test, (nm_test, mh_test * mw_test))


model = TestNeuralNetworkModel(ih, iw, ic, mh_train, mw_train)

model.fit(x_train, y_train,
          epochs=10,
          batch_size=10,
          shuffle=True,
          validation_data=(x_test, y_test))

model.save('my_model.h5')




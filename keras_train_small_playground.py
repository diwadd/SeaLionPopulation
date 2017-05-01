import os

import numpy as np

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.datasets import mnist

import data_handling_and_preparation as dhap
import working_directory_definition as wdd




x_train, x_validation, x_test, y_train, y_validation, y_test = dhap.load_train_test_data_trainsmall2()

print("x_train.shape: %s" % (str(x_train.shape)))
print("x_validation.shape: %s" % (str(x_validation.shape)))
print("x_test.shape: %s" % (str(x_test.shape)))
print("y_train.shape: %s" % (str(y_train.shape)))
print("y_validation.shape: %s" % (str(y_validation.shape)))
print("y_test.shape: %s" % (str(y_test.shape)))


h, w, c = x_train[0,:,:,:].shape
print("Single image size: %d, %d, %d" % (h, w, c))

input_image = Input(shape=(128, 128, 3))

# Simple model
x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_image)
print("first conv2d - x - size: " + str(x.shape))

x = MaxPooling2D((2, 2), padding='same')(x)
print("first max pooling - x - size: " + str(x.shape))

x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
print("second conv2d - x - size: " + str(x.shape))


print("x size: " + str(x.shape))

(x_train, _), (x_test, _) = mnist.load_data()

print(type(x_train))
print(x_train.shape)
print(x_test.shape)




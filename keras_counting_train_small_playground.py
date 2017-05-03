import os

import numpy as np

from keras import backend as K
import keras_counting_model_definitions as kcmd

import data_handling_and_preparation as dhap
import working_directory_definition as wdd


directories = wdd.check_directory_structure_trainsmall2()
top_dir = directories["TOP_DIR"]
version_directory = dhap.get_current_version_directory(top_dir)

counting_model_filename = version_directory + "counting_model.h5"

train_test_data = dhap.load_train_test_data_trainsmall2(fraction=0.2, data_type="counting")

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

print("\n\n ===> ---- <=== \n\n")

ni, ih, iw, ic = x_train.shape
print("Single image size: %d, %d, %d" % (ih, iw, ic))

nl_train, sl_train = y_train.shape
nl_test, sl_test = y_test.shape
print("Single mask size: %d" % (nl_train))

y_train = np.reshape(y_train, (nl_train, sl_train))
y_test = np.reshape(y_test, (nl_test, sl_test))

print("y_train.shape: " + str(y_train.shape))
print("y_test.shape: " + str(y_test.shape))


K.get_session()
model = kcmd.TestCountingNeuralNetworkModel(ih, iw, ic, sl_train)

model.fit(x_train, y_train,
          epochs=10,
          batch_size=10,
          shuffle=True,
          validation_data=(x_test, y_test))

model.save(counting_model_filename)

K.clear_session()




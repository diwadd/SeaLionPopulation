import os

import numpy as np

from keras import backend as K
import keras_detection_model_definitions as kdmd

import data_handling_and_preparation as dhap
import working_directory_definition as wdd



top_dir = os.path.dirname(os.path.abspath(__file__)) + "/"
directories = dhap.get_current_version_directory(top_dir)

detection_model_filename = directories["PARAMETERS_DIRECTORY"] + "detection_model.h5"
train_test_data = dhap.load_train_test_data_trainsmall2(directories, fraction=1.0, data_type="detection")


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

nm_train, mh_train, mw_train = y_train.shape
nm_test, mh_test, mw_test = y_test.shape
print("Single mask size: %d, %d" % (mh_train, mw_train))

y_train = np.reshape(y_train, (nm_train, mh_train * mw_train))
y_test = np.reshape(y_test, (nm_test, mh_test * mw_test))

K.get_session()
#model = kdmd.TestDetectionNeuralNetworkModel(ih, iw, ic, mh_train, mw_train)
model = kdmd.DetectionNeuralNetworkModelTrainSmall2(ih, iw, ic, mh_train, mw_train)

model.fit(x_train, y_train,
          epochs=500,
          batch_size=100,
          shuffle=True,
          validation_data=(x_test, y_test))

model.save(detection_model_filename)

K.clear_session()




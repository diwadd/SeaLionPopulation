import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

def TestCountingNeuralNetworkModel(ih, iw, ic, sl):
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

    model.add(Dense((sl), activation="relu"))

    model.compile(loss="mean_squared_error",
                  optimizer="adadelta",
                  metrics=["accuracy"])

    return model

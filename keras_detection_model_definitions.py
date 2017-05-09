import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D


def TestDetectionNeuralNetworkModel(ih, iw, ic, mh, mw):
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

    model.compile(loss="binary_crossentropy",
                  optimizer="adadelta",
                  metrics=["accuracy"])

    print("\n ---> Model summary <--- \n")
    model.summary()

    return model


def DetectionNeuralNetworkModelTrainSmall2(ih, iw, ic, mh, mw):
    """
    A simple model used to test the machinery on TrainSmall2.
    ih, iw, ic - describe the dimensions of the input image
    mh, mw - describe the dimensions of the output mask


    """
    dropout = 0.8

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(ih, iw, ic)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout))
    

    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout))

    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout))

    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout))

    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))

    model.add(Dense((mh * mw), activation="sigmoid"))

    model.compile(loss="binary_crossentropy",
                  optimizer="adadelta",
                  metrics=["accuracy"])

    print("\n ---> Model summary <--- \n")
    model.summary()

    return model




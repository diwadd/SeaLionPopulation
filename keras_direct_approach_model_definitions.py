import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras import regularizers
from keras import backend as K


from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions



def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def TestDetectionNeuralNetworkModel(ih, iw, ic, mh, loss=root_mean_squared_error):
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
    model.add(Dense(32, activation="relu"))
    #model.add(Dropout(0.5))

    model.add(Dense((mh), activation="sigmoid"))

    model.compile(loss=loss,
                  optimizer="adadelta",
                  metrics=["accuracy"])

    print("\n ---> Model summary <--- \n")
    model.summary()

    return model


def DetectionNeuralNetworkModelTrainSmall2(ih, iw, ic, mh, loss=root_mean_squared_error):
    """
    A simple model used to test the machinery on TrainSmall2.
    ih, iw, ic - describe the dimensions of the input image
    mh, mw - describe the dimensions of the output mask


    """
    dropout = 0.9
    alpha = 0.001 # LeakyReLu parameter
    lm = 0.1 # reguralization lambda

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), 
                         activation="linear",
                         kernel_initializer="glorot_normal", 
                         kernel_regularizer=regularizers.l2(lm), 
                         input_shape=(ih, iw, ic)))
    model.add(LeakyReLU(alpha=alpha))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout))


    model.add(Conv2D(64, (3, 3), activation="linear",
                                 kernel_initializer="glorot_normal", 
                                 kernel_regularizer=regularizers.l2(lm)))
    model.add(LeakyReLU(alpha=alpha))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))
    

    model.add(Conv2D(64, (3, 3), activation="linear",
                                 kernel_initializer="glorot_normal", 
                                 kernel_regularizer=regularizers.l2(lm)))
    model.add(LeakyReLU(alpha=alpha))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout))


    model.add(Conv2D(64, (3, 3), activation="linear",
                                 kernel_initializer="glorot_normal", 
                                 kernel_regularizer=regularizers.l2(lm)))
    model.add(LeakyReLU(alpha=alpha))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))


    model.add(Conv2D(64, (3, 3), activation="linear",
                                 kernel_initializer="glorot_normal",
                                 kernel_regularizer=regularizers.l2(lm)))
    model.add(LeakyReLU(alpha=alpha))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout))

    model.add(Conv2D(128, (3, 3), activation="linear",
                                  kernel_initializer="glorot_normal", 
                                  kernel_regularizer=regularizers.l2(lm)))
    model.add(LeakyReLU(alpha=alpha))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout))

    model.add(Conv2D(128, (3, 3), activation="linear",
                                  kernel_initializer="glorot_normal", 
                                  kernel_regularizer=regularizers.l2(lm)))
    model.add(LeakyReLU(alpha=alpha))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout))

    #model.add(Conv2D(128, (3, 3), activation="linear",
    #                              kernel_initializer="glorot_normal", 
    #                              kernel_regularizer=regularizers.l2(lm)))
    #model.add(LeakyReLU(alpha=alpha))
    #model.add(BatchNormalization())
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(dropout))

    #model.add(Conv2D(64, (3, 3), activation="linear"))
    #model.add(LeakyReLU(alpha=alpha))
    #model.add(BatchNormalization())
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(dropout))

    #model.add(Conv2D(64, (3, 3), activation="relu"))
    #model.add(BatchNormalization())
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(dropout))

    model.add(Flatten())
    model.add(Dense(256, activation="linear", kernel_regularizer=regularizers.l2(lm)))
    model.add(LeakyReLU(alpha=alpha))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))

    model.add(Dense(mh))

    #sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    adam = keras.optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.01)

    model.compile(loss="mean_squared_error",
                  optimizer="adadelta")

    print("\n ---> Model summary <--- \n")
    model.summary()

    return model



def ImageNetTransferModel():
    """
    This model has a fixed input size (224 x 224 x 3).

    """

    in_model = ResNet50(weights='imagenet')
    in_model.layers.pop()

    o = Dense(5)(in_model.layers[-1].output)

    model = Model(inputs=in_model.layers[0].input, outputs=[o])

    adam = keras.optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.01)

    model.compile(loss="mean_squared_error",
                  optimizer="adadelta")
    model.summary()

    return model




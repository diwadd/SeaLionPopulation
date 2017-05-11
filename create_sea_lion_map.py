import os
import sys

import cv2
import numpy as np

import matplotlib.pyplot as plt

from keras import backend as K
from keras.models import load_model


def display_images_and_masks_in_patches_list(image,
                                             lion_map_red,
                                             lion_map_mag,
                                             lion_map_bro,
                                             lion_map_blu,
                                             lion_map_gre):
    """
    This functions is used internaly in prepare_single_image to plot
    the most relevant images (patches) in data preprosessing.
    
    """

    rows = 2
    cols = 4

    f, axs = plt.subplots(rows, cols, figsize=(20, 12))

    plt.subplot(rows, cols,1)
    plt.imshow(image)
    plt.axis("off")
    plt.title("image")

    plt.subplot(rows, cols,2)
    plt.imshow(lion_map_red)
    plt.axis("off")
    plt.title("lion_map_red")

    plt.subplot(rows, cols,3)
    plt.imshow(lion_map_mag)
    plt.axis("off")
    plt.title("lion_map_mag")

    plt.subplot(rows, cols,4)
    plt.imshow(lion_map_bro)
    plt.axis("off")
    plt.title("lion_map_bro")

    plt.subplot(rows, cols,5)
    plt.imshow(lion_map_blu)
    plt.axis("off")
    plt.title("lion_map_blu")

    plt.subplot(rows, cols,6)
    plt.imshow(lion_map_gre)
    plt.axis("off")
    plt.title("lion_map_gre")

    dummy_image = np.zeros(lion_map_gre.shape)
    plt.subplot(rows, cols,7)
    plt.imshow(dummy_image)
    plt.axis("off")
    plt.title("dummy_image")

    plt.subplot(rows, cols,8)
    plt.imshow(dummy_image)
    plt.axis("off")
    plt.title("dummy_image")

    plt.show()



def map_image(image,
              model):

    ih, iw, ic = image.shape
    
    lion_map_red = np.zeros((ih, iw))
    lion_map_mag = np.zeros((ih, iw))
    lion_map_bro = np.zeros((ih, iw))
    lion_map_blu = np.zeros((ih, iw))
    lion_map_gre = np.zeros((ih, iw))

    for i in range(16, ih - 16):
        for j in range(16, iw - 16):
            simg = image[(i - 16):(i + 16 + 1), (j - 16):(j + 16 + 1), :]
    
            #print("simg.shape: " + str(simg.shape))

            simg = np.reshape(simg, (1, 33, 33, 3))
            labels = model.predict(simg, batch_size=1, verbose=0)

            max_index = np.argmax(labels)

            #print(labels)
            #print(max_index)
            #print()

            if (max_index == 5):
                continue
            elif (max_index == 4):
                # labels shape is (1, 6)
                lion_map_gre[i][j] = labels[0, max_index]
                print(max_index)
            elif (max_index == 3):
                lion_map_blu[i][j] = labels[0, max_index]
                print(max_index)
            elif (max_index == 2):
                lion_map_bro[i][j] = labels[0, max_index]
                print(max_index)
            elif (max_index == 1):
                lion_map_mag[i][j] = labels[0, max_index]
                print(max_index)
            elif (max_index == 0):
                lion_map_red[i][j] = labels[0, max_index]
                print(max_index)
            else:
                pass

            return lion_map_red, lion_map_mag, lion_map_bro, lion_map_blu, lion_map_gre



if (__name__ == "__main__"):
    print("Creating sea lion map...")

    filename = "/home/tadek/Coding/Kaggle/SeaLionPopulation/TrainSmall2/Train/45.jpg"
    model_filename = "/home/tadek/Coding/Kaggle/SeaLionPopulation/softmax_model_just_fit_33x33_10epoch_small.h5"

    image = cv2.imread(filename)

    ih, iw, ic = image.shape

    lion_map = np.zeros((ih, iw))

    model = load_model(model_filename)

    lion_map_red, lion_map_mag, lion_map_bro, lion_map_blu, lion_map_gre = map_image(image,
                                                                                     model)

    cv2.imwrite("/home/tadek/Coding/Kaggle/SeaLionPopulation/image.jpg", image)
    cv2.imwrite("/home/tadek/Coding/Kaggle/SeaLionPopulation/lion_map_red.jpg", (255*lion_map_red).astype(np.uint8))
    cv2.imwrite("/home/tadek/Coding/Kaggle/SeaLionPopulation/lion_map_mag.jpg", (255*lion_map_mag).astype(np.uint8))
    cv2.imwrite("/home/tadek/Coding/Kaggle/SeaLionPopulation/lion_map_bro.jpg", (255*lion_map_bro).astype(np.uint8))
    cv2.imwrite("/home/tadek/Coding/Kaggle/SeaLionPopulation/lion_map_blu.jpg", (255*lion_map_blu).astype(np.uint8))
    cv2.imwrite("/home/tadek/Coding/Kaggle/SeaLionPopulation/lion_map_gre.jpg", (255*lion_map_gre).astype(np.uint8))

    display_images_and_masks_in_patches_list(image,
                                                 lion_map_red,
                                                 lion_map_mag,
                                                 lion_map_bro,
                                                 lion_map_blu,
                                                 lion_map_gre)



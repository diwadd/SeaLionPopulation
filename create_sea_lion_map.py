import os
import sys
import time

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

    MODEL_INPUT_SIZE_H = 33
    MODEL_INPUT_SIZE_W = 33
    IMAGE_OFFSET = 16

    ih, iw, ic = image.shape
    
    lion_map_red = np.zeros((ih, iw))
    lion_map_mag = np.zeros((ih, iw))
    lion_map_bro = np.zeros((ih, iw))
    lion_map_blu = np.zeros((ih, iw))
    lion_map_gre = np.zeros((ih, iw))


    # imagex = np.reshape(image, (1, ih, iw, ic))

    index = 0
    for i in range(IMAGE_OFFSET, ih - IMAGE_OFFSET):
        print("Processed: " + str((i + 1) / ih))

        img_stack = np.zeros((iw, MODEL_INPUT_SIZE_H, MODEL_INPUT_SIZE_W, 3))
        for j in range(IMAGE_OFFSET, iw - IMAGE_OFFSET):

            img_stack[j, :, :, :] = image[(i - IMAGE_OFFSET):(i + IMAGE_OFFSET + 1), (j - IMAGE_OFFSET):(j + IMAGE_OFFSET + 1), :]
            
            #simg = np.reshape(simg, (1, 33, 33, 3))
            #max_index = np.argmax(labels)

            #print(labels)
            #print(max_index)
            #print()

            """
            if (max_index == 5):
                continue
            elif (max_index == 4):
                # labels shape is (1, 6)
                lion_map_gre[i][j] = labels[0, max_index]
                #print(max_index)
            elif (max_index == 3):
                lion_map_blu[i][j] = labels[0, max_index]
                #print(max_index)
            elif (max_index == 2):
                lion_map_bro[i][j] = labels[0, max_index]
                #print(max_index)
            elif (max_index == 1):
                lion_map_mag[i][j] = labels[0, max_index]
                #print(max_index)
            elif (max_index == 0):
                lion_map_red[i][j] = labels[0, max_index]
                #print(max_index)
            else:
                pass
            """

        #print("img_stack.shape: " + str(img_stack.shape))
        labels = model.predict(img_stack, batch_size=4000, verbose=0)
        #print("labels.shape: " + str(labels.shape))

        for k in range(IMAGE_OFFSET, iw - IMAGE_OFFSET):
        
            max_index = np.argmax(labels[k,:])
            #print("labels[" + str(k) + "]: " + str(labels[k,:]) )
            #print("max_index: " + str(max_index))
            if (max_index == 5):
                continue
            elif (max_index == 4):
                # labels shape is (1, 6)
                lion_map_gre[i][k] = labels[k, max_index]
                #print(max_index)
            elif (max_index == 3):
                lion_map_blu[i][k] = labels[k, max_index]
                #print(max_index)
            elif (max_index == 2):
                lion_map_bro[i][k] = labels[k, max_index]
                #print(max_index)
            elif (max_index == 1):
                lion_map_mag[i][k] = labels[k, max_index]
                #print(max_index)
            elif (max_index == 0):
                lion_map_red[i][k] = labels[k, max_index]
                #print(max_index)
            else:
                pass

    return lion_map_red, lion_map_mag, lion_map_bro, lion_map_blu, lion_map_gre



if (__name__ == "__main__"):
    print("Creating sea lion map...")

    filename = "/home/tadek/Coding/Kaggle/SeaLionPopulation/TrainSmall2/Train/45.jpg"
    model_filename = "/home/tadek/Coding/Kaggle/SeaLionPopulation/softmax_model_just_fit.h5"    
    #model_filename = "/home/tadek/Coding/Kaggle/SeaLionPopulation/softmax_model_just_fit_33x33_10epoch_small.h5"

    image = cv2.imread(filename)/255.0

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



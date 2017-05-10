import os
import glob
import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt

import data_handling_and_preparation as dhap


def plot_image(img):
    plt.subplots()
    plt.imshow(img, cmap=plt.cm.get_cmap("Greys"))
    plt.colorbar()
    plt.show()


def dispatch_images(train_dir, 
                    train_dotted_dir, 
                    data_dir, 
                    nw, 
                    nh,
                    radious_list=[24, 24, 24, 12, 10],
                    take_train_dotted=False,
                    save_jpg=False):

    train_image_list = glob.glob(train_dir + "*.jpg")
    train_dotted_image_list = glob.glob(train_dotted_dir + "*.jpg")

    n_train_images = len(train_image_list)
    n_train_dotted_images = len(train_dotted_image_list)

    if (n_train_images != n_train_dotted_images):
        sys.exit("ERROR in dispatch_images: Number of loaded train and train dotted images do not agree!")

    for n in range(n_train_images):

        print("Processed: " + str( (n + 1) / n_train_images ))

        if (n > 2):
            continue

        train_stem = dhap.get_filename_stem(train_image_list[n])
        train_dotted_stem = dhap.get_filename_stem(train_dotted_image_list[n])

        if (train_stem != train_dotted_stem):
            sys.exit("ERROR in dispatch_images: File stems do not agree!")

        print("Loading " + train_image_list[n])
        train_image = cv2.imread(train_image_list[n])

        print("Loading " + train_dotted_image_list[n])
        train_dotted_image = cv2.imread(train_dotted_image_list[n])


        if (train_image.shape != train_dotted_image.shape):
            print("Image shapes are not compitable! Skiping!")
            continue

        NEAR_ZERO_THRESHOLD = 1
        gray_image = cv2.cvtColor(train_dotted_image.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_image, NEAR_ZERO_THRESHOLD, 255, cv2.THRESH_BINARY)
        mask = mask/255.0
        train_image = dhap.apply_mask(train_image, mask)

        sea_lion_images = dhap.softmax_dispatch_count_lions_in_a_single_lion_image(train_image,
                                                                                   train_dotted_image,
                                                                                   radious_list=radious_list,
                                                                                   take_train_dotted=take_train_dotted)


        for i in range(len(sea_lion_images)):
            image_for_save = sea_lion_images[i][0]
            image_for_save = cv2.resize(image_for_save, (nw, nh), interpolation = cv2.INTER_LINEAR)

            earth_image = sea_lion_images[i][2]
            earth_image = cv2.resize(earth_image, (nw, nh), interpolation = cv2.INTER_LINEAR)

            label_str = str(sea_lion_images[i][1][0]) + "_" + \
                        str(sea_lion_images[i][1][1]) + "_" + \
                        str(sea_lion_images[i][1][2]) + "_" + \
                        str(sea_lion_images[i][1][3]) + "_" + \
                        str(sea_lion_images[i][1][4]) + "_" + "0"

            if (save_jpg == True):
                cv2.imwrite(data_dir + "lion_image_filestem_" + train_stem + "_id_" + str(i) + "_label_" + label_str + ".jpg", image_for_save)
                cv2.imwrite(data_dir + "earth_image_filestem_" + train_stem + "_id_" + str(i) + "_label_0_0_0_0_0_1.jpg", earth_image)

            fn = data_dir + "lion_image_filestem_" + train_stem + "_id_" + str(i) + "_label_" + label_str + ".npz"
            labels = sea_lion_images[i][1]/np.sum(sea_lion_images[i][1])
            labels = np.reshape(labels, (1, -1))     
            np.savez_compressed(fn, 
                                image=image_for_save.astype(np.float32)/255.0, 
                                labels=labels)

            fn = data_dir + "earth_image_filestem_" + train_stem + "_id_" + str(i) + "_label_0_0_0_0_0_1.npz"
            np.savez_compressed(fn, 
                                image=earth_image.astype(np.float32)/255.0, 
                                labels=np.array([[0.0 , 0.0, 0.0, 0.0, 0.0, 1.0]]))



if (__name__ == "__main__"):

    top_dir = "/home/tadek/Coding/Kaggle/SeaLionPopulation/"
    train_dir = "/media/tadek/My_Passport/Kaggle.com/SeaLionPopulation/Kaggle-NOAA-SeaLions_FILES/Train/"
    train_dotted_dir = "/media/tadek/My_Passport/Kaggle.com/SeaLionPopulation/Kaggle-NOAA-SeaLions_FILES/TrainDotted/"
    data_dir = "/home/tadek/Coding/Kaggle/SeaLionPopulation/temp_data/"

    dhap.check_if_dir_exists_create_it_if_not_remove_content(data_dir)

    nw = 32
    nh = 32
    # radious_list=[24, 24, 24, 12, 10]
    radious_list=[28, 28, 28, 16, 12]

    dispatch_images(train_dir,
                    train_dotted_dir,
                    data_dir, 
                    nw=nw,
                    nh=nh,
                    radious_list=radious_list,
                    take_train_dotted=False,
                    save_jpg=False)






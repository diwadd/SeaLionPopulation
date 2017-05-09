import os
import glob
import sys

import cv2
import data_handling_and_preparation as dhap

import matplotlib.pyplot as plt

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
                    take_train_dotted=False):

    train_image_list = glob.glob(train_dir + "*.jpg")
    train_dotted_image_list = glob.glob(train_dotted_dir + "*.jpg")

    n_train_images = len(train_image_list)
    n_train_dotted_images = len(train_dotted_image_list)

    if (n_train_images != n_train_dotted_images):
        sys.exit("ERROR in dispatch_images: Number of loaded train and train dotted images do not agree!")

    for n in range(n_train_images):

        if (n != 1):
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
            cv2.imwrite(data_dir + "lion_image_filestem_" + train_stem + "_id_" + str(i) + "_label_" + label_str + "1.jpg", image_for_save)
            cv2.imwrite(data_dir + "earth_image_filestem_" + train_stem + "_id_" + str(i) + "_label_0_0_0_0_0_11.jpg", earth_image)



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
                    take_train_dotted=True)






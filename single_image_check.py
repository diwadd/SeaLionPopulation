import os
import sys
import glob

import cv2
import numpy as np

import matplotlib.pyplot as plt
import data_handling_and_preparation as dhap

def plot_image(img, title="Title"):
    plt.subplots()
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.colorbar()
    plt.title(title)
    plt.show()



def load_single_lion_file(filename,
                          y_label="labels"):
    """
    Load a single files that has been dispatched by
    softmax_like_approach.py

    """

    file_exists = os.path.exists(filename)
    if (file_exists == False):
        sys.exit("ERROR! The file path you provided does not exist!")

    loaded_data = np.load(filename)

    image = loaded_data["image"]
    labels = loaded_data[y_label]

    return image, labels




if (__name__ == "__main__"):

    #filename = "/home/tadek/Coding/Kaggle/SeaLionPopulation/temp_data_28_28_28_16_12/earth_image_filestem_271_id_455_label_0_0_0_0_0_1.npz"
    #filename = "/home/tadek/Coding/Kaggle/SeaLionPopulation/temp_data_24_24_24_24_24/lion_image_filestem_93_id_234_label_0_0_1_2_0_0.npz"
    #filename = "/home/tadek/Coding/Kaggle/SeaLionPopulation/temp_data_28_28_28_16_12/lion_image_filestem_411_id_154_label_0_0_4_3_0_0.npz"

    top_dir = os.path.dirname(os.path.abspath(__file__)) + "/"
    directories = dhap.get_current_version_directory(top_dir)

    data_dir = directories["PREPROCESSED_DETECTION_DATA_DIRECTORY"]
    filename_list = glob.glob(data_dir + "*npz")

    n = len(filename_list)
    for i in range(n):
        print(filename_list[i])
        image, labels = load_single_lion_file(filename_list[i], "mask")
        plot_image(image, str(labels))

    """
    # Compare a few images on one plot.

    f1 = "/home/tadek/Coding/Kaggle/SeaLionPopulation/Detection_data/476_prep_data_collection_c_0_patches_list_i_3_j_3.npz"
    f2 = "/home/tadek/Coding/Kaggle/SeaLionPopulation/Detection_data/476_prep_data_collection_c_1_patches_list_i_3_j_3.npz"
    f3 = "/home/tadek/Coding/Kaggle/SeaLionPopulation/Detection_data/476_prep_data_collection_c_2_patches_list_i_3_j_3.npz"
    
    image1, labels1 = load_single_lion_file(f1, "mask")
    image2, labels2 = load_single_lion_file(f2, "mask")
    image3, labels3 = load_single_lion_file(f3, "mask")
    

    f, axs = plt.subplots(1, 3, figsize=(20, 12))

    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title(str(labels1))

    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title(str(labels2))

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(image3, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title(str(labels3))

    plt.show()

    """



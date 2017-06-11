import os
import sys

import cv2
import numpy as np

import matplotlib.pyplot as plt


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



    image, labels = load_single_lion_file(sys.argv[1], "mask")

    plot_image(image, str(labels))

    print(image)
    print(labels)








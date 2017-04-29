import cv2
import numpy as np

import matplotlib.pyplot as plt
import data_handling_and_preparation as dhap


def plot_image(image):

    plt.imshow(image)
    plt.axis("off")
    plt.show()

count_filename = "/home/tadek/Coding/Kaggle/SeaLionPopulation/Counting_data/41_prep_data_lion_images_list_n_9.npz"
count_image, labels = dhap.load_single_lion_count_file(count_filename)

count_image = cv2.cvtColor(count_image, cv2.COLOR_BGR2RGB)
plot_image(count_image)
print("labels: " + str(labels))


detection_filename = "/home/tadek/Coding/Kaggle/SeaLionPopulation/Detection_data/41_prep_data_collection_c_0_patches_list_i_4_j_4.npz"
detection_image, mask = dhap.load_single_lion_detection_file(detection_filename)

detection_image = cv2.cvtColor(detection_image, cv2.COLOR_BGR2RGB)
detection_image = (detection_image*255).astype(np.uint8)
plot_image(detection_image)

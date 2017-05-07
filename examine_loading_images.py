import cv2
import numpy as np

import matplotlib.pyplot as plt
import data_handling_and_preparation as dhap


def plot_image(image):

    plt.imshow(image)
    plt.axis("off")
    plt.show()

#count_filename = "/home/tadek/Coding/Kaggle/SeaLionPopulation/TrainSmall2/Counting_data/41_prep_data_lion_images_list_n_4.npz"
#count_image, labels = dhap.load_single_lion_count_file(count_filename)

#count_image = cv2.cvtColor(count_image.astype(np.float32), cv2.COLOR_BGR2RGB)
#plot_image(count_image)
#print("labels: " + str(labels))


detection_filename = "/home/tadek/Coding/Kaggle/SeaLionPopulation/TrainSmall2/Detection_data/42_prep_data_collection_c_0_patches_list_i_3_j_7.npz"
print(detection_filename)
detection_image, mask = dhap.load_single_lion_detection_file(detection_filename)

detection_image = cv2.cvtColor(detection_image, cv2.COLOR_BGR2RGB)
plot_image(detection_image)

plot_image(mask)

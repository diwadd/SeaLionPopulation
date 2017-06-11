import pickle
import os

import cv2
import numpy as np

from keras import backend as K
from keras.models import load_model

from data_handling_and_preparation import SAP
import data_handling_and_preparation as dhap
import working_directory_definition as wdd

import matplotlib.pyplot as plt

# We are using a custom loss function.
# By default keras does not know this function.
# We have to define out custom loss within keras.
# Solution: https://github.com/fchollet/keras/issues/5916
from keras_direct_approach_model_definitions import root_mean_squared_error
import keras.losses
keras.losses.root_mean_squared_error = root_mean_squared_error


def apply_model_to_image_patches_list(image_patches_list, model):

    nh_slices, nw_slices = dhap.get_patches_list_dimensions(image_patches_list)
    output_patches_list = [[None for j in range(nw_slices)] for i in range(nh_slices)]

    for i in range(nh_slices):
        for j in range(nw_slices):
            ih, iw, ic = image_patches_list[i][j].shape

            x_image = np.reshape(image_patches_list[i][j], (1, ih, iw, ic))
            output_patches_list[i][j] = model.predict(x_image, batch_size=1, verbose=0)

    return output_patches_list


def reshape_patches_list(patches_list,
                         resize_mask_patch_to_h,
                         resize_mask_patch_to_w):
    """
    The patches in the patch list that is ouput by apply_model_to_image_patches_list
    has size (1, n). This is the output that is given by the model.
    This needs to be resized to (resize_mask_patch_to_h x resize_mask_patch_to_w) as
    in the original mask.

    """

    nh_slices, nw_slices = dhap.get_patches_list_dimensions(patches_list)
    for i in range(nh_slices):
        for j in range(nw_slices):

            patch = np.reshape(patches_list[i][j], (resize_mask_patch_to_h, resize_mask_patch_to_w))
            patches_list[i][j] = patch

    return patches_list


def count_sea_lions_in_image(filename,
                              model,
                              patch_h,
                              patch_w,
                              resize_image_patch_to_h, 
                              resize_image_patch_to_w,
                              resize_mask_patch_to_h,
                              resize_mask_patch_to_w,
                              display_mask=False):

    """
    Take a filename, reads the image, slices the image into a
    patches list, applies the model to each patch, recombined the
    image from the patches list, outputs a image with detected
    sea lions. The model is assumed to be a detection model.

    """

    train_image = cv2.imread(filename)
    image_patches_list = dhap.slice_the_image_into_patches(train_image, patch_h, patch_w)

    # Recombine the image from the patches (train_image.shape != image.shape)
    # bacause the size of the image is adjusted to be a multiple of patch_h and patch_w.    
    image = dhap.combine_pathes_into_image(image_patches_list)


    # Resize the patches to the ones used by the model.
    image_patches_list = dhap.resize_patches_in_patches_list(image_patches_list, 
                                                             resize_image_patch_to_h, 
                                                             resize_image_patch_to_w)

    counts_patches_list = apply_model_to_image_patches_list(image_patches_list, model)

    # Sum the lions from al the patches.
    lion_sum = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    nh_slices, nw_slices = dhap.get_patches_list_dimensions(counts_patches_list)

    for i in range( nh_slices ):
        for j in range( nw_slices ):  
            lion_sum = lion_sum + counts_patches_list[i][j]



    print("Number of lions in image: " + str(lion_sum))

    if (display_mask == True):
        fig, ax = plt.subplots()
        cax = ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        cbar = fig.colorbar(cax)
        plt.axis("off")
        plt.show()




if __name__ == '__main__':

    top_dir = os.path.dirname(os.path.abspath(__file__)) + "/"
    directories = dhap.get_current_version_directory(top_dir)

    parameters_directory = directories["PARAMETERS_DIRECTORY"]

    # To process the data we need the parameters with which the
    # data were generated. This should be stored under parameters_directory + "parameters_file.pkls".
    parameter_file = open(parameters_directory + "parameters_file.pkls", "rb")
    parameters = pickle.load(parameter_file)


    # Detection data parameters for dispatch.
    patch_h = parameters["patch_h"]
    patch_w = parameters["patch_w"]
    resize_image_patch_to_h = parameters["resize_image_patch_to_h"]
    resize_image_patch_to_w = parameters["resize_image_patch_to_w"]
    resize_mask_patch_to_h = parameters["resize_mask_patch_to_h"]
    resize_mask_patch_to_w = parameters["resize_mask_patch_to_w"]
    radious_list = parameters["radious_list"]
    sap_list = parameters["sap_list"]

    # Counting data parameters for dispatch.
    counting_radious=parameters["counting_radious"]
    nh=parameters["nh"] # final image size - height
    nw=parameters["nw"] # final image size - width
    counting_dot_threshold=parameters["counting_dot_threshold"]
    lions_contour_dot_threshold=parameters["lions_contour_dot_threshold"]
    h_threshold=parameters["h_threshold"] # minimal height (size) of a single lion that will be cropped
    w_threshold=parameters["w_threshold"] # minimal width (size) of a single lion that will be cropped
    rectangle_shape=parameters["rectangle_shape"]


    direct_approach_model_hdf5_filename = parameters_directory + "counting_model.h5"

    K.get_session()
    model = load_model(direct_approach_model_hdf5_filename)


    print("detection model type: " + str(type(model)))

    #filename = "/home/tadek/Coding/Kaggle/SeaLionPopulation/TrainSmall/Train/1.jpg"
    filename = "/home/tadek/Coding/Kaggle/SeaLionPopulation/TrainSmall2/Train/48.jpg"

    count_sea_lions_in_image(filename,
                             model,
                             patch_h,
                             patch_w,
                             resize_image_patch_to_h, 
                             resize_image_patch_to_w,
                             resize_mask_patch_to_h,
                             resize_mask_patch_to_w,
                             display_mask=True)

    K.clear_session()









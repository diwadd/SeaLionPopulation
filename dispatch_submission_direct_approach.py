import pickle
import os
import sys
import glob

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

            # Patches list is in uint8 format with
            # values from 0 to 255.
            # The model expects floats from 0.0 to 1.0.
            x_image = x_image.astype(np.float32)/255.0
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

    train_image = (cv2.imread(filename).astype(np.float32))
    image_patches_list = dhap.slice_the_image_into_patches(train_image, patch_h, patch_w)

    # Recombine the image from the patches (train_image.shape != image.shape)
    # bacause the size of the image is adjusted to be a multiple of patch_h and patch_w.    
    image = dhap.combine_pathes_into_image(image_patches_list)

    # Convert to uint8 for cv2.cvtColor.    
    #image = image.astype(np.uint8)

    # Resize the patches to the ones used by the model.
    image_patches_list = dhap.resize_patches_in_patches_list(image_patches_list, 
                                                             resize_image_patch_to_h, 
                                                             resize_image_patch_to_w)

    counts_patches_list = apply_model_to_image_patches_list(image_patches_list, model)

    # Sum the lions from al the patches.
    lion_sum = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    nh_slices, nw_slices = dhap.get_patches_list_dimensions(counts_patches_list)

    # Get the number of lions in each patch and
    # add them to the lion sum.
    # lion_sum is the predicted number of lions
    # in the image.
    for i in range( nh_slices ):
        for j in range( nw_slices ):
            # print(counts_patches_list[i][j])  
            lion_sum = lion_sum + counts_patches_list[i][j]



    # print("Number of lions in image: " + str(lion_sum))

    if (display_mask == True):
        fig, ax = plt.subplots()
        cax = ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        cbar = fig.colorbar(cax)
        plt.axis("off")
        plt.show()

    return lion_sum


def count_sea_lions_in_image_list(filename_list,
                                  model,
                                  patch_h,
                                  patch_w,
                                  resize_image_patch_to_h, 
                                  resize_image_patch_to_w,
                                  resize_mask_patch_to_h,
                                  resize_mask_patch_to_w,
                                  display_mask=False):
    
    n = len(filename_list)
    y_pred = np.zeros((n, dhap.CONST_NUMBER_OF_CLASSES))
    ids = [j for j in range(n)]    

    for i in range(n):
        print("i: " + str(i) + " out of " + str(n))
        lion_sum = count_sea_lions_in_image(filename_list[i],
                                            model=model,
                                            patch_h=patch_h,
                                            patch_w=patch_w,
                                            resize_image_patch_to_h=resize_image_patch_to_h, 
                                            resize_image_patch_to_w=resize_image_patch_to_w,
                                            resize_mask_patch_to_h=resize_mask_patch_to_h,
                                            resize_mask_patch_to_w=resize_mask_patch_to_w,
                                            display_mask=display_mask)

        ids[i] = dhap.get_filename_stem(filename_list[i])
        y_pred[i,:] = np.array(lion_sum)

    return y_pred, ids


def make_labels_data_dict(labels_filename):

    """
    Read the csv files that contains the labels and
    store them in a dict.

    """

    labels_dict = {}
    labels = dhap.read_csv(labels_filename)

    for i in range(len(labels)):

        row = [int(labels[i][j]) for j in range(len(labels[i]))]
        labels_dict[row[0]] = np.array(row[1:]).astype(np.float32)

    #for keys, values in labels_dict.items():
    #    print(str(keys) + " : " + str(values))      

    return labels_dict


def get_y_true(labels_dict, stem_list):

    
    n = len(stem_list)
    y_true = np.zeros((n, dhap.CONST_NUMBER_OF_CLASSES))    

    for i in range(n):
        y_true[i, :] = labels_dict[stem_list[i]]
        
    return y_true



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


    direct_approach_model_hdf5_filename = parameters_directory + "counting_model_min_imgnet.h5"

    K.get_session()
    model = load_model(direct_approach_model_hdf5_filename)


    print("detection model type: " + str(type(model)))

    #filename = "/home/tadek/Coding/Kaggle/SeaLionPopulation/TrainSmall/Train/1.jpg"
    filename_1 = "/home/tadek/Coding/Kaggle/SeaLionPopulation/TrainSmall2/Train/48.jpg"
    filename_2 = "/home/tadek/Coding/Kaggle/SeaLionPopulation/TrainSmall2/Train/47.jpg"
    filename_3 = "/home/tadek/Coding/Kaggle/SeaLionPopulation/TrainSmall2/Train/46.jpg"
    filename_4 = "/home/tadek/Coding/Kaggle/SeaLionPopulation/TrainSmall/Train/7.jpg"
    filename_5 = "/home/tadek/Coding/Kaggle/SeaLionPopulation/TrainSmall/Train/9.jpg"
    filename_list = [filename_1, filename_2, filename_3, filename_4, filename_5]

    stem_1 = int(dhap.get_filename_stem(filename_1))
    stem_2 = int(dhap.get_filename_stem(filename_2))
    stem_3 = int(dhap.get_filename_stem(filename_3))
    stem_4 = int(dhap.get_filename_stem(filename_4))
    stem_5 = int(dhap.get_filename_stem(filename_5))
    stem_list = [stem_1, stem_2, stem_3, stem_4, stem_5]


    labels_filename = "/home/tadek/Coding/Kaggle/SeaLionPopulation/TrainSmall2/Train/train.csv"
    labels_dict = make_labels_data_dict(labels_filename)

    """
    lion_sum = count_sea_lions_in_image(filename,
                                        model,
                                        patch_h,
                                        patch_w,
                                        resize_image_patch_to_h, 
                                        resize_image_patch_to_w,
                                        resize_mask_patch_to_h,
                                        resize_mask_patch_to_w,
                                        display_mask=True)
    """
    y_pred, ids = count_sea_lions_in_image_list(filename_list,
                                           model,
                                           patch_h,
                                           patch_w,
                                           resize_image_patch_to_h, 
                                           resize_image_patch_to_w,
                                           resize_mask_patch_to_h,
                                           resize_mask_patch_to_w,
                                           display_mask=False)

    y_true = get_y_true(labels_dict, stem_list)

    print("y_true:")
    print(y_true)
    print("y_pred: ")
    print(y_pred)


    loss = K.eval(root_mean_squared_error(y_true, y_pred))
    print("\nLoss for image: \n" + str(loss) + "\n")



    filename_list = glob.glob("/media/tadek/My_Passport/Kaggle.com/SeaLionPopulation/Kaggle-NOAA-SeaLions_FILES/Test/*.jpg")

    # start = 1499
    # stop = 13500

    start = 13499
    stop = 20000

    filename_list = filename_list[start:stop:1]


    y_pred, ids = count_sea_lions_in_image_list(filename_list,
                                                model,
                                                patch_h,
                                                patch_w,
                                                resize_image_patch_to_h, 
                                                resize_image_patch_to_w,
                                                resize_mask_patch_to_h,
                                                resize_mask_patch_to_w,
                                                display_mask=False)

    
    ids = np.array(ids)
    ids = np.reshape(ids, (-1, 1))

    print("Test predictions shape:" + str(y_pred.shape))
    print("Ids predictions shape:" + str(ids.shape))

    output_array = np.hstack((ids, y_pred)).astype(dtype=np.float32)
    print("output_array shape:" + str(output_array.shape))
    
    output_array[output_array < 0.0] = 0.0
    output_array = output_array[output_array[:, 0].argsort()]
    print(output_array)

    print("---")
    y_pred[y_pred < 0.0] = 0.0
    print(y_pred)
    n, m = y_pred.shape

    f = open("sub.csv", "a")
    f.write("test_id,adult_males,subadult_males,adult_females,juveniles,pups\n")
    for k in range(n):
        a = str(int(output_array[k,0]))
        b = str(output_array[k,1])
        c = str(output_array[k,2])
        d = str(output_array[k,3])
        e = str(output_array[k,4])
        g = str(output_array[k,5])


        f.write(a + "," + b + "," + c + "," + d + "," + e + "," + g + "\n")

    f.close()


    # a = np.array([ [1,2,3,4], [4,5,6,7], [2,3,4,50], [9,8,7,6], [2,2,2,2]]).astype(np.float32)
    # b = np.array([ [1,5,3,4], [3,5,6,7], [7,6,4,34], [4,6,7,8], [3,3,3,3]]).astype(np.float32)
    # loss = K.eval(root_mean_squared_error(a, b))
    # print(loss)


    K.clear_session()




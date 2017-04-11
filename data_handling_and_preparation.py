import sys
import math
import random
import os
import pickle
import glob
import time
import shutil
import re

import cv2
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

CONST_TRAIN_DOTTED_IMAGES_DIR = "/home/tadek/Coding/Kaggle/SeaLionPopulation/TrainSmall/TrainDotted/"
CONST_TRAIN_IMAGES_DIR = "/home/tadek/Coding/Kaggle/SeaLionPopulation/TrainSmall/Train/"
CONST_PREPROCESSED_DATA_DIR = "/home/tadek/Coding/Kaggle/SeaLionPopulation/PreProcessedTrainData/"
CONST_WHITE_COLOR = (255, 255, 255)
CONST_BLUE_COLOR = (255, 0, 0)
CONST_GREEN_COLOR = (0, 255, 0)
CONST_RED_COLOR = (0, 0, 255)


def measure_time(func):
    """
    Decorator to measure the execution time of a function.
    """
    def func_wrapper(*args, **kwargs):

        start = time.time()
        ret = func(*args, **kwargs)
        stop = time.time()
        print("Time spent in function " + func.__name__ + ": " + str(stop - start))
        
        return ret

    return func_wrapper



class SAP:
    """
    Static Augmentation Parameters.
    The training data is augmented and save on the hard drive.
    The save images can be rotated and scaled.
    It's static because these poeration will not be performed
    during training.

    This class is a short hand to represent the augmented
    parameters i.e. the rotation angle (ang) and scale (scl).

    """

    def __init__(self, ang, scl):
        self.rotation_angle = ang
        self.scale = scl

    def __str__(self):
        ra = round(self.rotation_angle, 2)
        s = round(self.scale, 2)
        return "rotation angle: %f deg, scale: %f" % (ra, s)


def detect_dots_in_image(image, color="MAGENTA"):
    """
    This function takes an RBG image with sea lions.
    The lions should be marked with colored dots.
    
    The colored dots of a given color are detected and an image
    with the detected dots is returned. The background is black.
    The color of the dots is not changed.

    The colors are detected with appropriate lower and upper RBG
    bounds and a bitwise_and function.

    color - color of the dots you want to detect.
    """

    # RBG in OpenCV is BGR = [BLUE, GREEN RED]
    # Postfixes are used only for testing purposses.
    DOTTED_IMAGE_MAGENTA_POSTFIX = "_dots_mag_x.jpg"
    MAGENTA_RGB_LOWER_BOUND = np.array([225,  0, 225], dtype=np.uint8)
    MAGENTA_RGB_UPPER_BOUND = np.array([255, 30, 255], dtype=np.uint8)

    DOTTED_IMAGE_RED_POSTFIX = "_dots_red_x.jpg"
    RED_RGB_LOWER_BOUND = np.array([ 0,  0, 225], dtype=np.uint8)
    RED_RGB_UPPER_BOUND = np.array([30, 30, 255], dtype=np.uint8)

    DOTTED_IMAGE_BLUE_POSTFIX = "_dots_blu_x.jpg"
    BLUE_RGB_LOWER_BOUND = np.array([140, 40, 15], dtype=np.uint8)
    BLUE_RGB_UPPER_BOUND = np.array([255, 80, 55], dtype=np.uint8)

    DOTTED_IMAGE_GREEN_POSTFIX = "_dots_gre_x.jpg"
    GREEN_RGB_LOWER_BOUND = np.array([5, 145, 20], dtype=np.uint8)
    GREEN_RGB_UPPER_BOUND = np.array([55, 195, 70], dtype=np.uint8)

    DOTTED_IMAGE_BROWN_POSTFIX = "_dots_bro_x.jpg"
    BROWN_RGB_LOWER_BOUND = np.array([ 0, 37,  70], dtype=np.uint8)
    BROWN_RGB_UPPER_BOUND = np.array([15, 55,  95], dtype=np.uint8)

    h, w, c = image.shape

    mask = None
    image_post_fix = None

    if color == "MAGENTA":
        mask = cv2.inRange(image,
                           MAGENTA_RGB_LOWER_BOUND,
                           MAGENTA_RGB_UPPER_BOUND)
        image_post_fix = DOTTED_IMAGE_MAGENTA_POSTFIX

    elif color == "RED":
        mask = cv2.inRange(image,
                           RED_RGB_LOWER_BOUND,
                           RED_RGB_UPPER_BOUND)
        image_post_fix = DOTTED_IMAGE_RED_POSTFIX

    elif color == "BLUE":
        mask = cv2.inRange(image,
                           BLUE_RGB_LOWER_BOUND,
                           BLUE_RGB_UPPER_BOUND)
        image_post_fix = DOTTED_IMAGE_BLUE_POSTFIX

    elif color == "GREEN":
        mask = cv2.inRange(image,
                           GREEN_RGB_LOWER_BOUND,
                           GREEN_RGB_UPPER_BOUND)
        image_post_fix = DOTTED_IMAGE_GREEN_POSTFIX

    elif color == "BROWN":
        mask = cv2.inRange(image,
                           BROWN_RGB_LOWER_BOUND,
                           BROWN_RGB_UPPER_BOUND)
        image_post_fix = DOTTED_IMAGE_BROWN_POSTFIX

    else:
        pass

    dotted_image = np.zeros((h,w,c), dtype=np.uint8)
    cv2.bitwise_and(image, image, dotted_image, mask=mask)

    #cv2.imwrite(image_filename.replace(".jpg", image_post_fix), dotted_image)

    return dotted_image


def plot_circles_return_mask(dotted_image, radious=40, dot_threshold=50):
    """
    This function takes as input the output from read_image_detect_dots.
    It detects contours of each dot. The contours are basically a list/numpy array of [x, y]
    coordinates of pixels that surround the dots.

    The first pixels coordinates of the contours are taken and a circle
    is drawn around this pixel (it suffices to take the first pixel
    because the dots can be treated as point like).

    A mask image is returned (0 for the background, 1 for the circle). 

    dotted_image - this is the output from read_image_detect_dots.

    """

    gray_image = cv2.cvtColor(dotted_image, cv2.COLOR_BGR2GRAY)

    _, thresholded_image = cv2.threshold(gray_image, dot_threshold, 255, cv2.THRESH_BINARY)
    _, contours, _ = cv2.findContours(thresholded_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    mask = thresholded_image
    for i in range(len(contours)):
        x = contours[i][0][0][0]
        y = contours[i][0][0][1]
        mask = cv2.circle(mask,(x, y), radious, CONST_WHITE_COLOR, -1)

    mask = mask/255.0

    return mask
    

def apply_mask(image, mask):
    """
    Applies a mask to an image.

    The size of the image should be N x M x 3.
    The mask should be a N x M array with 0.0 or 1.0 values.
    
    """

    image[:,:,0] = image[:,:,0]*mask
    image[:,:,1] = image[:,:,1]*mask
    image[:,:,2] = image[:,:,2]*mask

    return image


def draw_poly(image, contour, epsilon=0.01, color=CONST_GREEN_COLOR):
    arc = cv2.arcLength(contour, True)
    poly = cv2.approxPolyDP(contour, epsilon * arc, True)
    cv2.drawContours(image, [poly], -1, color, 1)


def get_detected_lions_list(masked_lion_image, 
                            dot_threshold=50, 
                            h_threshold=10, 
                            w_threshold=10,
                            rectangle_shape=True):

    gray_image = cv2.cvtColor(masked_lion_image, cv2.COLOR_BGR2GRAY)
    
    _, thresholded_image = cv2.threshold(gray_image, dot_threshold, 255, cv2.THRESH_BINARY)
    _, contours, _ = cv2.findContours(thresholded_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #cv2.imwrite("base_image.jpg", masked_lion_image)

    lions_image = None
    lion_images_list = []
    for c in range(len(contours)):
        # Find basic rectangle.
        x, y, w, h = cv2.boundingRect(contours[c])
        
        # Check if its big enough?
        if (h < h_threshold) or (w < w_threshold):
            continue

        if (rectangle_shape == True):
            lions_image = masked_lion_image[y:(y + h), x:(x + w)]
            lion_images_list.append(lions_image)
            continue

        # This part of the code will find the minimal
        # spanning eclipse.

        ellipse = cv2.fitEllipse(contours[c])
        eh, ew, _ = masked_lion_image.shape

        e_mask = np.zeros((eh, ew))
        cv2.ellipse(e_mask, ellipse, CONST_BLUE_COLOR, -1)

        e_masked_image = np.array(masked_lion_image)
        e_masked_image = apply_mask(e_masked_image, e_mask)

        e_gray_image = cv2.cvtColor(e_masked_image, cv2.COLOR_BGR2GRAY)
        
        _, e_thresholded_image = cv2.threshold(e_gray_image, dot_threshold, 255, cv2.THRESH_BINARY)
        _, e_contours, _ = cv2.findContours(e_thresholded_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for j in range(len(e_contours)):
            x, y, w, h = cv2.boundingRect(e_contours[j])

            # There should be only one e_contour.
            # However, sometimes small, flanky contours are detected.
            # We discard them here.
            if (h < h_threshold) or (w < w_threshold):
                continue

            # If the contour is big enough then we found the right one.
            lions_image = masked_lion_image[y:(y + h), x:(x + w)]
            lion_images_list.append(lions_image)
            break


        #cv2.imwrite("extracted_lions_c_" + str(c) + "_j_" + str(j) + ".jpg", lions_image)

        cv2.imwrite("y0_extracted_lions_c_" + str(c) + ".jpg", lions_image)

        #draw_poly(masked_lion_image, contours[c], color=CONST_GREEN_COLOR)
    #cv2.imwrite("contours.jpg", masked_lion_image)
    return lion_images_list


def count_lions_in_a_single_lion_image(lion_image, radious=15, dot_threshold=1):
    """
    Takes a sinle lion image and determins the number of
    lions in the image. 
    In principle there should be only one lion on each image.
    This is however not always true since some times the lions
    lie very close to one another.
    Returns an array with the corresponing counts.

    """

    color_list = ["MAGENTA", "RED", "BLUE", "GREEN", "BROWN"]
    lions_count = [0, 0, 0, 0, 0]

    for c in range(len(color_list)):

        processing_image = np.array(lion_image)
        processing_image = detect_dots_in_image(processing_image, color=color_list[c])

        # The dots that are detected can be fragmented.
        # We plot circles around them to make the into one solid block
        # of pixels. This can be done by the plot_circles_return_mask function.
        # We multiply by 255 because we want an image not a mask.
        processing_image = 255.0*plot_circles_return_mask(processing_image, radious=10)
        
        gray_image = processing_image.astype(np.uint8) # cv2.cvtColor(processing_image.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        
        _, thresholded_image = cv2.threshold(gray_image, dot_threshold, 255, cv2.THRESH_BINARY)
        _, contours, _ = cv2.findContours(thresholded_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        lions_count[c] = lions_count[c] + len(contours)

    return lions_count


def mask_the_lion_image(image, radious_list=[45, 50, 40, 22, 42], dot_threshold=50):
    """
    Takes an image in which the sea lions are marked with coloered dots.

    Calculates a mask and then applies this mask to the input image.
    As a result only the lions remain visible in the original image.
    Each lion is located within a circle whose radious is defined in radious_list.

    The masked_lion_image that is returned is an image on which only the
    lions are visible. All other things are black.

    Each lion is marked with a circle.
    The radious of the circle should be adjusted to the
    lion's size.

    """
    h, w, _ = image.shape

    color_list = ["MAGENTA", "RED", "BLUE", "GREEN", "BROWN"]
    mask = np.zeros((h, w))

    for c in range(len(color_list)):       

        dotted_image = detect_dots_in_image(image, color_list[c])
        
        mask_of_a_give_color = plot_circles_return_mask(dotted_image,
                                                        radious=radious_list[c],
                                                        dot_threshold=dot_threshold)

        mask = mask + mask_of_a_give_color
        
    mask[mask > 0] = 1.0

    # We want to avoid modyfing the original image.
    masked_lion_image = np.array(image)
    masked_lion_image = apply_mask(masked_lion_image, mask)

    return masked_lion_image, mask


def mask_a_few_lion_images(filename_list):
    """
    Takes a list with image filenames. For each image
    saves a masked_lion_image and its mask.

    """

    n_files = len(filename_list)
    for i in range(n_files):
        print("Processed: %f" % ((i + 1)/n_files), end="\r")

        image = cv2.imread(filename_list[i])
        masked_lion_image, mask = mask_the_lion_image(image)
    print()


def get_new_dimensions(h, w, patch_h, patch_w):
    """
    A helper function for slice_the_image_into_patches and slice_the_mask_into_patches.

    Calculates the new dimenstion of a resized image.

    The dimenstions are chosen in such a way so that they
    lie as close as possible to the multiples of patch_h and patch_w.

    For example if the image dimension are h = 3744, w = 5616 and we
    set patch_h = 400, patch_w = 400 then the new image will have dimensions
    nh = 3600, nw = 5600.

    """


    hd = (h % patch_h)
    wd = (w % patch_w)

    lower_h = h - hd
    lower_w = w - wd

    upper_h = lower_h + patch_h
    upper_w = lower_w + patch_w

    l_h_distance = abs(h - lower_h)
    l_w_distance = abs(w - lower_w)

    u_h_distance = abs(h - upper_h)
    u_w_distance = abs(w - upper_w)

    nh = None
    nw = None

    if (u_h_distance <= l_h_distance):
        nh = upper_h
    else:
        nh = lower_h

    if (u_w_distance <= l_w_distance):
        nw = upper_w
    else:
        nw = lower_w

    nh_slices = nh//patch_h
    nw_slices = nw//patch_w

    return nh, nw, nh_slices, nw_slices


def slice_the_image_into_patches(image,
                                 patch_h = 400,
                                 patch_w = 400):
    """
    Takes an image and cuts it into patches.
    The size of the patch is patch_h x patch_w x patch_c.

    The function resizes the image before cutting.
    The image dimensions are set to new values so that
    they are multiples of patch_h and patch_w.

    A list of the pathes is returned.
    
    """

    # Calculating the resized image dimensions.

    h = None
    w = None
    c = None
    nh = None
    nw = None
    nh_slices = None
    nw_slices = None
    patches_list = None
    try:
        h, w, c = image.shape
        nh, nw, nh_slices, nw_slices = get_new_dimensions(h, w, patch_h, patch_w)
        patches_list = [[np.zeros((patch_h, patch_w, c), dtype=np.uint8) for j in range(nw_slices)] for i in range(nh_slices)]
    except ValueError:
        h, w = image.shape
        nh, nw, nh_slices, nw_slices = get_new_dimensions(h, w, patch_h, patch_w)
        patches_list = [[np.zeros((patch_h, patch_w), dtype=np.uint8) for j in range(nw_slices)] for i in range(nh_slices)]


    

    # In the cv2.resize mathod the image shape is reverted (w, h) -> (h, w).
    resized_image = cv2.resize(image, (nw, nh), interpolation = cv2.INTER_LINEAR)

    for i in range( nh_slices ):
        for j in range( nw_slices ):

            try:  
                patches_list[i][j][:,:,:] = resized_image[(i*patch_h):(i*patch_h + patch_h), (j*patch_w):(j*patch_w + patch_w), :]
            except IndexError:
                patches_list[i][j][:,:] = resized_image[(i*patch_h):(i*patch_h + patch_h), (j*patch_w):(j*patch_w + patch_w)]   

    return patches_list


def slice_the_mask_into_patches(mask,
                                patch_h = 400,
                                patch_w = 400,
                                prefix_text = None):
    """
    Takes an mask and cuts it into patches.
    The size of the patch is patch_h x patch_w.

    The function resizes the mask before cutting.
    The mask dimensions are set to new values so that
    they are multiples of patch_h and patch_w.

    A list of the pathes is returned.
    
    """

    # Calculating the resized image dimensions.
    h, w = mask.shape

    nh, nw, nh_slices, nw_slices = get_new_dimensions(h, w, patch_h, patch_w)

    # In the cv2.resize mathod the image shape is reverted (w, h) -> (h, w).
    resized_mask = cv2.resize(mask, (nw, nh), interpolation = cv2.INTER_LINEAR)
    
    if (prefix_text != None):
        # The values in the mask are between 0.0 and 1.0, To save multiply by 255
        # to have a gray scale image.
        cv2.imwrite(prefix_text + "resized_mask.jpg", 255*resized_mask)

    patches_list = [[np.zeros((patch_h, patch_w), dtype=np.uint8) for j in range(nw_slices)] for i in range(nh_slices)]

    for i in range( nh_slices ):
        for j in range( nw_slices ):  
            patches_list[i][j][:,:] = resized_mask[(i*patch_h):(i*patch_h + patch_h), (j*patch_w):(j*patch_w + patch_w)]
            
            if (prefix_text != None):
                cv2.imwrite(prefix_text + str(patch_w) + "_" + str(patch_h) + "_%d_%d_mask.jpg" % (i, j), 255*patches_list[i][j])

    return patches_list


def get_patches_list_dimensions(patches_list):
    """
    Returns the dimensions of a patches_list.

    """

    nh_slices = len(patches_list)
    if (nh_slices == 0):
        sys.exit("ERROR: The provided patches_list has zero length!")
        

    nw_slices = len(patches_list[0])

    # This gives some security that an error will be spotted
    # but not full security because we use only the 0 index (patches_list[0]).
    if (nw_slices == 0):
        sys.exit("ERROR: The provided patches_list has zero length!")

    return nh_slices, nw_slices



def combine_pathes_into_image(patches_list, prefix_text = None):
    """
    Takes a patches_list returned by slice_the_image_into_patches and
    combines it back into a full image.

    """

    nh_slices, nw_slices = get_patches_list_dimensions(patches_list)
    patch_h, patch_w, patch_c = patches_list[0][0].shape

    h = nh_slices*patch_h
    w = nw_slices*patch_w

    image = np.zeros((h, w, patch_c), dtype=np.uint8)

    for i in range(nh_slices):
        for j in range(nw_slices):
            image[(i*patch_h):(i*patch_h + patch_h), (j*patch_w):(j*patch_w + patch_w), :] = patches_list[i][j][:,:,:]


    if (prefix_text != None):
        cv2.imwrite(prefix_text + "image_combined_from_patches.jpg", image)

    return image



def combine_pathes_into_mask(patches_list, prefix_text = None):
    """
    Takes a patches_list returned by slice_the_mask_into_patches and
    combines it back into a full mask.

    """
    
    nh_slices, nw_slices = get_patches_list_dimensions(patches_list)

    patch_h, patch_w = patches_list[0][0].shape

    h = nh_slices*patch_h
    w = nw_slices*patch_w

    mask = np.zeros((h, w), dtype=np.uint8)

    for i in range(nh_slices):
        for j in range(nw_slices):
            mask[(i*patch_h):(i*patch_h + patch_h), (j*patch_w):(j*patch_w + patch_w)] = patches_list[i][j][:,:]


    if (prefix_text != None):
        # The values in the mask are between 0.0 and 1.0, To save multiply by 255
        # to have a gray scale image.
        cv2.imwrite(prefix_text + "mask_combined_from_patches.jpg", 255*mask)

    return mask


def resize_patch(patch, nh, nw):
    """
    Takes a patch and resizes it.
    This works with any image i.e. instead of
    patch any image can be passed.

    """

    resized_mask_patch = cv2.resize(patch, (nw, nh), interpolation = cv2.INTER_LINEAR)

    return resized_mask_patch



def resize_patches_in_patches_list(patches_list, nh, nw):
    """
    Takes a list of mask patches and resizes each patch.
    The new size of the patch is nh x nw.
    
    """

    nh_slices, nw_slices = get_patches_list_dimensions(patches_list)

    patch_c = None
    resized_patches_list = None
    try:
        _, _, patch_c = patches_list[0][0].shape
        resized_patches_list = [[np.zeros((nh, nw, patch_c), dtype=np.uint8) for j in range(nw_slices)] for i in range(nh_slices)]
    except ValueError:
        resized_patches_list = [[np.zeros((nh, nw), dtype=np.uint8) for j in range(nw_slices)] for i in range(nh_slices)]

    for i in range(nh_slices):
        for j in range(nw_slices):
            resized_patches_list[i][j] = resize_patch(patches_list[i][j], nh, nw)

    return resized_patches_list


def diff_two_patches_lists_with_masks(patches_list_1, patches_list_2):
    """
    Takes two patches_lists and calcualtes 
    the difference between their elements i.e.
    patches_list_1[i][j] - patches_list_2[i][j]

    The difference is stored in diff_patches_list which
    is also returned from this function.

    """

    nh_slices_1, nw_slices_1 = get_patches_list_dimensions(patches_list_1)
    nh_slices_2, nw_slices_2 = get_patches_list_dimensions(patches_list_2)

    patch_h_1, patch_w_1 = patches_list_1[0][0].shape
    patch_h_2, patch_w_2 = patches_list_2[0][0].shape

    if (patch_h_1 != patch_h_2) or (patch_w_1 != patch_w_2):
        sys.exit("ERROR: Patch dimension mismatch!")


    diff_patches_list = [[np.zeros((patch_h_1, patch_w_1), dtype=np.uint8) for j in range(nw_slices_1)] for i in range(nh_slices_2)]

    for i in range(nh_slices_1):
        for j in range(nw_slices_1):
            diff_patches_list[i][j] = patches_list_1[i][j] - patches_list_2[i][j]

            #if (prefix_text != None):
                # The values in the mask are between 0.0 and 1.0, To save multiply by 255
                # to have a gray scale image.
            #    cv2.imwrite(prefix_text + str(patch_h_1) + "_" + str(patch_h_2) +  "_%d_%d_diff_two_patches.jpg" % (i, j), 255*diff_patches_list[i][j])

    return diff_patches_list


def apply_mask_patches_list_to_image_patches_list(mask_patches_list,
                                                  image_patches_list):

    """
    Take two patches_lists. 
    The first one should contain masks.
    The second one should contain images.

    The function applies the masks to the images i.e. invokes
    apply_mask(image_patches_list[i][j], mask_patches_list[i][j])

    It returns a new patches_list with the masked images.

    """

    nh_slices_m, nw_slices_m = get_patches_list_dimensions(mask_patches_list)
    nh_slices_i, nw_slices_i = get_patches_list_dimensions(image_patches_list)

    if (nh_slices_m != nh_slices_i) or (nw_slices_m != nw_slices_i):
        sys.exit("Patches_lists dimension mismatch!")

    patch_h_m, patch_w_m = mask_patches_list[0][0].shape
    patch_h_i, patch_w_i, patch_c = image_patches_list[0][0].shape

    if (patch_h_m != patch_h_i) or (patch_w_m != patch_w_i):
        sys.exit("ERROR: Patch dimension mismatch!")

    patches_list = [[np.zeros((patch_h_i, patch_w_i, patch_c), dtype=np.uint8) for j in range(nw_slices_i)] for i in range(nh_slices_i)]
    

    for i in range(nh_slices_m):
        for j in range(nw_slices_m):

            # We want to avoid modyfing the original image.
            patch = np.array(image_patches_list[i][j])
            patches_list[i][j] = apply_mask(patch, 
                                            mask_patches_list[i][j])

    return patches_list


def get_data_eigenvalues_and_eigenvectors(filename_list, fraction=1/10):
    """
    Calcualtes the eigen values (ew) and eigen vectors (ev) of an image set.
    The image set if passed as a list of filenames.
    When the set of images is too large to fit into the memory
    an approximate solution can be found be taking a fraction of
    the original data set.

    """

    n_files = len(filename_list)
    if (n_files == 0) or (fraction > 1.0) or (round(fraction, 3) <= 0.0):
        sys.exit("ERROR: The number of filenames passed to the function is zero or " + \
                 "the fraction has a value greater than 1.0 or less the 0.0.")

    sub_filename_list = random.sample(filename_list, int(fraction*n_files))

    n_files = len(sub_filename_list)
    if (n_files == 0):
        sys.exit("ERROR: The number of files after sampling is zero.")


    images_array = np.empty(n_files, dtype=object)
    for f in range(n_files):
        images_array[f] = cv2.imread(sub_filename_list[f])


    h, w, c = images_array[0].shape
    pixel_array = np.resize(images_array[0], (h * w, c))

    for i in range(1, n_files):
        h, w, c = images_array[i].shape
        pixel_array = np.concatenate((pixel_array, np.resize(pixel_array[i], (h * w, c))), axis=0)

    pixel_array = pixel_array/255.0
    C = np.cov(pixel_array.T)
    ew, ev = np.linalg.eig(C)

    return ew, ev


def color_augmentation_of_an_image(image, ew, ev, ca_std=0.2):
    """
    Apply color augmentation to an image as in Krizhevsky et al. 2012

    This function is meant to be used for one the fly (online) data
    augumentation during training.

    ew - eigen values returned by get_data_eigenvalues_and_eigenvectors.
    ev - eigen vectors returned by get_data_eigenvalues_and_eigenvectors.

    """

    image = image/255.0

    h, w, c = image.shape    
    
    r = np.random.randn(c)

    delta = np.dot(ev, np.transpose((ca_std * r) * ew))

    delta_image = np.zeros((h, w, c))
    for i in range(c):
        delta_image[:, :, i] = delta[i]

    image = image + delta_image
    
    # Adding the delta to the image might cause its
    # color values (RGB) to go bellow zero
    # or above one. Here be bring the outliners back into
    # the interval [0, 1].
    image[image < 0.0] = 0.0
    image[image > 1.0] = 1.0

    image = image*255.0

    return image


def rotate_and_scale_image(image, 
                           rotation_angle=180,
                           scale=1.0):
    """
    Rotate image about an angle equal to rotation_angle.
    Scale the rotate image according to scale.

    """
    h = None
    w = None
    try:
        h, w, _ = image.shape
    except ValueError:
        h, w = image.shape

    R = cv2.getRotationMatrix2D((w/2, h/2), rotation_angle, scale)
    rotated_image = cv2.warpAffine(image, R, (w, h))

    return rotated_image


def color_augment_patches_list(patches_list, ew, ev, ca_std):
    """
    Apply color augmentation to all images in a patches_list.
    This function takse only patches_lists with images!
    Passing masks will raise an ValueError and program termination.

    """

    nh_slices, nw_slices = get_patches_list_dimensions(patches_list)
    patch_h, patch_w, _ = patches_list[0][0].shape

    color_augmented_patches_list = [[np.zeros((patch_h, patch_w), dtype=np.uint8) for j in range(nw_slices)] for i in range(nh_slices)]
    for i in range(nh_slices):
        for j in range(nw_slices):
            color_augmented_patches_list[i][j] = color_augmentation_of_an_image(patches_list[i][j], ew, ev, ca_std)
    
    return color_augmented_patches_list


def rotate_patches_list(patches_list, rotation_angle=0.0, scale=1.0):
    """
    Apply rotation to all images in a patches_list.
    This function takse only patches_lists with images!
    Passing masks will raise an ValueError and program termination.

    """

    nh_slices, nw_slices = get_patches_list_dimensions(patches_list)

    rotated_patches_list = None
    try:
        patch_h, patch_w, patch_c = patches_list[0][0].shape
        rotated_patches_list = [[np.zeros((patch_h, patch_w, patch_c), dtype=np.uint8) for j in range(nw_slices)] for i in range(nh_slices)]
    except ValueError:
        patch_h, patch_w = patches_list[0][0].shape
        rotated_patches_list = [[np.zeros((patch_h, patch_w), dtype=np.uint8) for j in range(nw_slices)] for i in range(nh_slices)]  

    
    for i in range(nh_slices):
        for j in range(nw_slices):
            rotated_patches_list[i][j] = rotate_and_scale_image(patches_list[i][j], rotation_angle, scale)
    
    return rotated_patches_list


def create_collection_of_rotated_patches_lists(patches_list, 
                                               sap_list=[SAP(0.0, 1.0), SAP(90.0, 1.0)]):
    """
    Creates a collection of patches_lists.
    In each patches_lists the images are rotated according to
    the values in rotation_angle_array given in deg.

    """

    n_pl_in_collection = len(sap_list) # pl = patches_lists
    patches_lists_collection = [patches_list for i in range(n_pl_in_collection)]

    nh_slices, nw_slices = get_patches_list_dimensions(patches_list)

    for c in range(n_pl_in_collection):
        patches_lists_collection[c] = rotate_patches_list(patches_lists_collection[c], 
                                                          sap_list[c].rotation_angle, 
                                                          sap_list[c].scale)

    return patches_lists_collection


def color_augment_patches_lists_collection(patches_lists_collection, ew, ev, ca_std):
    """
    Creates a collection of patches_lists.
    In each patches_lists the images are color agumented 
    according to ew, ev, ca_std

    """

    n_pl_in_collection = len(patches_lists_collection) # pl = patches_lists

    for c in range(n_pl_in_collection):
        patches_lists_collection[c] = color_augment_patches_list(patches_lists_collection[c], ew, ev, ca_std)

    return patches_lists_collection


def is_mask_patches_list(patches_list):
    """
    Check if patches_list contains masks.
    Mask should contain only values between 0 and 1.
    Images on the otherhand values between 0 and 255.
    Images with values [0,1] will be considered as masks
    and rescaled to 255 when saving.

    """

    nh_slices, nw_slices = get_patches_list_dimensions(patches_list)

    is_mask = False
    max_value = -math.inf
    for i in range(nh_slices):
        for j in range(nw_slices):
            value = np.max(patches_list[i][j])
    
            if (value > max_value):
                max_value = value

    print("max_value: " + str(max_value))

    if (round(max_value, 1) <= 1.0):
        is_mask = True

    return is_mask



def save_images_in_patches_list(patches_list, image_filename):

    # Check if patches_list contains masks.
    # Mask should contain only values between 0 and 1.
    # Images on the otherhand values between 0 and 255.
    # Images with values [0,1] will be considered as masks
    # and rescaled to 255 when saving.
    is_mask = is_mask_patches_list(patches_list)
    nh_slices, nw_slices = get_patches_list_dimensions(patches_list)

    for i in range(nh_slices):
        for j in range(nw_slices):

            current_image_filename = image_filename + "_" + str(i) + "_" + str(j) + ".jpg"
            if (is_mask == True):      
                cv2.imwrite(current_image_filename, 255*patches_list[i][j])
            else:
                cv2.imwrite(current_image_filename, patches_list[i][j])


def save_collection_of_patches_lists(patches_lists_collection, image_filename_stem):

    n_pl_in_collection = len(patches_lists_collection)

    for c in range(n_pl_in_collection):
        save_images_in_patches_list(patches_lists_collection[c], "c" + str(c) + "from_collection_number_" + image_filename_stem)


def print_image_sizes(filename_list):

    for i in range(len(filename_list)):
        image = cv2.imread(filename_list[i])
        print(image.shape)


def pickle_patches_lists_collection(filename, patches_lists_collection):

    f = open(filename,"wb")
    pickle.dump(patches_lists_collection, f)
    f.close()


def savez_two_patches_list(filename_stem, mask_patches_list, image_patches_list):

    nh_slices_m, nw_slices_m = get_patches_list_dimensions(mask_patches_list)
    nh_slices_i, nw_slices_i = get_patches_list_dimensions(image_patches_list)

    for i in range(nh_slices_m):
        for j in range(nw_slices_m):
            # The image vaules are reduced to a [0, 1] interval.
            image_for_save = (image_patches_list[i][j]/255.0).astype(np.float32)
            mask_for_save = mask_patches_list[i][j].astype(np.float32)

            fn = filename_stem + "_i_" + str(i) + "_j_" + str(j) + ".npz"
            np.savez_compressed(fn, image=image_for_save, mask=mask_for_save)


def savez_patches_list_collection(filename_stem,
                                  mask_patches_lists_collection,
                                  image_patches_lists_collection):

    n_pl_in_collection = len(mask_patches_lists_collection)

    for c in range(n_pl_in_collection):
        fn = filename_stem + "_c_" + str(c)
        savez_two_patches_list(fn, 
                               mask_patches_lists_collection[c], 
                               image_patches_lists_collection[c])


def get_filename_list_in_dir(dir_name, file_type="jpg"):
    """
    Takes a direcotry and returns a list of all files in it
    that have the extension given by file_type.

    """

    tdd = os.path.isdir(dir_name)
    if (tdd == False):
        sys.exit("ERROR! Directory does not exist.")

    filename_list = glob.glob(dir_name + "*" + file_type)
    
    return filename_list


def get_filename_stem(filename):

    filename_stem = filename.split("/")
    filename_stem = filename_stem[-1].split(".")    
    filename_stem = filename_stem[0]

    return filename_stem


def display_images_and_masks_in_patches_list(image_patch,
                                             resized_image_patch,
                                             masked_dotted_image_patch,
                                             mask_patch,
                                             resized_mask_patch,
                                             back_resized_mask_patch,
                                             image_masked_with_resized_patch,
                                             sap):
    """
    This functions is used internaly in prepare_single_image to plot
    the most relevant images (patches) in data preprosessing.
    
    """

    rows = 2
    cols = 4

    f, axs = plt.subplots(rows, cols, figsize=(20, 12))

    plt.subplot(rows, cols,1)
    plt.imshow(cv2.cvtColor(image_patch, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title(str(image_patch.shape))

    plt.subplot(rows, cols,2)
    plt.imshow(cv2.cvtColor(resized_image_patch, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title(str(resized_image_patch.shape))

    plt.subplot(rows, cols,3)
    plt.imshow(cv2.cvtColor(masked_dotted_image_patch, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title(str(masked_dotted_image_patch.shape))

    plt.subplot(rows, cols,4)
    plt.imshow(mask_patch)
    plt.axis("off")
    plt.title(str(mask_patch.shape))

    plt.subplot(rows, cols,5)
    plt.imshow(resized_mask_patch)
    plt.axis("off")
    plt.title(str(resized_mask_patch.shape))

    plt.subplot(rows, cols,6)
    plt.imshow(back_resized_mask_patch)
    plt.axis("off")
    plt.title(str(back_resized_mask_patch.shape))

    plt.subplot(rows, cols,7)
    plt.imshow(cv2.cvtColor(image_masked_with_resized_patch, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title(str(image_masked_with_resized_patch.shape))

    dummy_image = np.zeros(back_resized_mask_patch.shape)
    plt.subplot(rows, cols,8)
    plt.imshow(dummy_image)
    plt.axis("off")
    plt.title(str(dummy_image.shape))

    plt.suptitle(str(sap))
    plt.show()



def prepare_single_full_input_image(train_image_filename, 
                                    train_dotted_image_filename,
                                    patch_h=500,
                                    patch_w=500,
                                    resize_image_patch_to_h=256,
                                    resize_image_patch_to_w=256,
                                    resize_mask_patch_to_h=64,
                                    resize_mask_patch_to_w=64,
                                    radious_list = [32, 32, 32, 16, 32],
                                    sap_list=[SAP(0.0, 1.0), SAP(90.0, 1.0)],
                                    interactive_plot=False,
                                    display_every=10):

    """
    This function prepares the data for the sea lion detection neural network.
    It takes two files as input:
    - train_image_filename - is a file from the Train folder.
    - train_dotted_image_filename - is a file to the corresponding image in the TrainDotted folder.
    The file names must be given with full paths e.g.
    train_image_filename = /full/path/to/image/image_name.jpg

    The function returns:
    - collection_of_resized_image_patches_lists - a collection of lists of image patches with 
                                                  dimensions (resize_image_patch_to_h x resize_image_patch_to_w) each.
    - collection_of_resized_mask_patches_lists - a collection of lists of mask patches with dimensions 
                                                 (resize_mask_patch_to_h x resize_mask_patch_to_w) each.

    """

    train_filename_stem = get_filename_stem(train_image_filename)
    train_dotted_filename_stem = get_filename_stem(train_dotted_image_filename)

    if (train_filename_stem != train_dotted_filename_stem):
        sys.exit("ERROR! Filename stems do not agree.")

    train_image = cv2.imread(train_image_filename)
    train_dotted_image = cv2.imread(train_dotted_image_filename)

    if (train_image.shape != train_dotted_image.shape):
        sys.exit("ERROR! Train and train dotted image shapes do not agree.")

    image_patches_list = slice_the_image_into_patches(train_image, patch_h, patch_w)
    resized_image_patches_list = resize_patches_in_patches_list(image_patches_list, 
                                                                resize_image_patch_to_h, 
                                                                resize_image_patch_to_w)
    masked_lion_image, mask = mask_the_lion_image(train_dotted_image, radious_list)


    mask_patches_list = slice_the_image_into_patches(mask, patch_h, patch_w)
    resized_mask_patches_list = resize_patches_in_patches_list(mask_patches_list, 
                                                               resize_mask_patch_to_h, 
                                                               resize_mask_patch_to_w)

    collection_of_image_patches_lists = create_collection_of_rotated_patches_lists(image_patches_list, sap_list)
    collection_of_resized_image_patches_lists = create_collection_of_rotated_patches_lists(resized_image_patches_list, sap_list)
    collection_of_mask_patches_lists = create_collection_of_rotated_patches_lists(mask_patches_list, sap_list)
    collection_of_resized_mask_patches_lists = create_collection_of_rotated_patches_lists(resized_mask_patches_list, sap_list)

    if (interactive_plot == True):

        back_resized_mask_patches_list = resize_patches_in_patches_list(resized_mask_patches_list, patch_h, patch_w)
        masked_dotted_image_patches_list = slice_the_image_into_patches(masked_lion_image, patch_h, patch_w)
        images_masked_with_resized_patches_list = apply_mask_patches_list_to_image_patches_list(back_resized_mask_patches_list,
                                                                                                image_patches_list)


        collection_of_back_resized_mask_patches_lists = create_collection_of_rotated_patches_lists(back_resized_mask_patches_list, sap_list)
        collection_of_masked_dotted_image_patches_lists = create_collection_of_rotated_patches_lists(masked_dotted_image_patches_list, sap_list)
        collection_of_images_masked_with_resized_patches_lists = create_collection_of_rotated_patches_lists(images_masked_with_resized_patches_list, sap_list)

        n_pl_in_collection = len(collection_of_images_masked_with_resized_patches_lists)
        print("Number of patches_lists: %d" % (n_pl_in_collection))
        for c in range(n_pl_in_collection):
            nh_slices, nw_slices = get_patches_list_dimensions(collection_of_images_masked_with_resized_patches_lists[c])

            print("Collection: %d, Number of patches: %d" % (c + 1, nh_slices*nw_slices))
            every_index = 0
            for index_i in range(nh_slices):
                for index_j in range(nw_slices):

                    # For a quick inspection we just need to see a few image.
                    if (every_index % display_every != 0):
                        every_index += 1
                        continue

                    ipl = collection_of_image_patches_lists[c][index_i][index_j]
                    ripl = collection_of_resized_image_patches_lists[c][index_i][index_j]
                    mdipl = collection_of_masked_dotted_image_patches_lists[c][index_i][index_j]
                    mpl = collection_of_mask_patches_lists[c][index_i][index_j]
                    rmpl = collection_of_resized_mask_patches_lists[c][index_i][index_j]
                    brmpl = collection_of_back_resized_mask_patches_lists[c][index_i][index_j]
                    imwrpl = collection_of_images_masked_with_resized_patches_lists[c][index_i][index_j]
                    sap = sap_list[c]

                    display_images_and_masks_in_patches_list(ipl, ripl, mdipl, mpl, rmpl, brmpl, imwrpl, sap)
                    every_index += 1
                
        #combined_image_masked_with_resized_patches_list = combine_pathes_into_image(images_masked_with_resized_patches_list)
        #get_detected_lions_list(combined_image_masked_with_resized_patches_list, dot_threshold=1)

    return collection_of_resized_mask_patches_lists, collection_of_resized_image_patches_lists


def prepare_and_dispatch_data(train_image_filename_list, 
                              train_dotted_image_filename_list,
                              preprocessed_data_dir,
                              patch_h=500,
                              patch_w=500,
                              resize_image_patch_to_h=256,
                              resize_image_patch_to_w=256,
                              resize_mask_patch_to_h=64,
                              resize_mask_patch_to_w=64,
                              radious_list = [32, 32, 32, 16, 32],
                              sap_list=[SAP(0.0, 1.0), SAP(90.0, 1.0)],
                              interactive_plot=False,
                              display_every=10):
    """
    Take a list of the train images and the train image with color dots.
    Prepares the data for training according to the provided parameters 
    and saves them in preprocessed_data_dir.
 
    """

    # Check is CONST_PREPROCESSED_DATA_DIR exists.
    pdd = os.path.isdir(preprocessed_data_dir)

    if (pdd == True):
        shutil.rmtree(preprocessed_data_dir)
        os.makedirs(preprocessed_data_dir)
    else:
        os.makedirs(preprocessed_data_dir)

    n_files = len(train_image_filename_list)
    print("Processed: 0.0 %% files.", end="\r")
    for i in range(n_files):
        #ci = collection_of_resized_image_patches_lists
        #cm = collection_of_resized_mask_patches_lists
        cm, ci = prepare_single_full_input_image(train_image_filename_list[i], 
                                      train_dotted_image_filename_list[i],
                                      patch_h,
                                      patch_w,
                                      resize_image_patch_to_h,
                                      resize_image_patch_to_w,
                                      resize_mask_patch_to_h,
                                      resize_mask_patch_to_w,
                                      radious_list,
                                      sap_list,
                                      interactive_plot,
                                      display_every)


        filename_stem = get_filename_stem(train_image_filename_list[i])
        filename_stem = preprocessed_data_dir + filename_stem + "_prep_data"

        savez_patches_list_collection(filename_stem, cm, ci)
        print("Processed: %f %% files." % ( 100.0*((i + 1)/n_files) ), end="\r")
    print()


@measure_time
def load_files_in_folder(folder_name):

    filename_list = glob.glob(folder_name + "*npz")
    for f in filename_list:    
        loaded = np.load(f)

def load_single_file(filename):
    
    file_exists = os.path.exists(filename)
    if (file_exists == False):
        sys.exit("ERROR! The file you provided does not exist!")


    loaded_data = np.load(filename)
    
    image = loaded_data["image"]
    mask = loaded_data["mask"]

    print("image.shape: " + str(image.shape))
    print("mask.shape: " + str(mask.shape))

    img_max = np.max(image)
    msk_max = np.max(mask)

    print("img_max: " + str(img_max))
    print("msk_max: " + str(msk_max))



def manual_test_of_lion_counting(train_image_filename, 
                                 train_dotted_image_filename,
                                 patch_h=500,
                                 patch_w=500,
                                 resize_image_patch_to_h=256,
                                 resize_image_patch_to_w=256,
                                 resize_mask_patch_to_h=64,
                                 resize_mask_patch_to_w=64,
                                 radious_list = [32, 32, 32, 16, 32],
                                 sap_list=[SAP(0.0, 1.0), SAP(90.0, 1.0)],
                                 interactive_plot=False,
                                 display_every=10):

    train_dotted_image = cv2.imread(train_dotted_image_filename)
    masked_lion_image, mask = mask_the_lion_image(train_dotted_image, radious_list)


    lion_images_list = get_detected_lions_list(masked_lion_image, dot_threshold=1, rectangle_shape=False)
    
    lion_count_sum = np.array([0,0,0,0,0])
    for gg in range(len(lion_images_list)):
        lion_count = count_lions_in_a_single_lion_image(lion_images_list[gg])
        lion_count_sum = lion_count_sum + np.array(lion_count)
        print("Lion count: " + str(lion_count))

    print(lion_count_sum)
    


def manual_testing():

    #image = cv2.imread(CONST_TRAIN_DOTTED_IMAGES_DIR + "0.jpg")
    #masked_lion_image = mask_the_lion_image(image)

    #cv2.imshow("image", masked_lion_image)
    #cv2.imwrite("image.jpg", masked_lion_image)

    filename_list = [CONST_TRAIN_DOTTED_IMAGES_DIR + str(i) + ".jpg" for i in range(10 + 1)]

    ew, ev = get_data_eigenvalues_and_eigenvectors(filename_list, fraction=1)
    ca_std=0.5

    print("ew: " + str(ew) + " ev: " + str(ev))

    image = cv2.imread(filename_list[8])
    augmented_image = color_augmentation_of_an_image(image, ew, ev, ca_std)
    rotated_image = rotate_and_scale_image(image, rotation_angle=180)

    cv2.imwrite("000_1image.jpg", image)
    cv2.imwrite("000_1auimg.jpg", augmented_image)
    #cv2.imwrite("000_1au_img_dif.jpg", image - augmented_image)
    #cv2.imwrite("000_1roimg.jpg", rotated_image)

    image = cv2.imread(filename_list[0])
    cv2.imwrite("original_image.jpg", image)

    masked_lion_image, mask = mask_the_lion_image(image)
    cv2.imwrite("original_mask.jpg", 255*mask)


    patch_h = 500
    patch_w = 500
    image_patches_list = slice_the_image_into_patches(masked_lion_image, patch_h, patch_w)
    rotated_patches_list = rotate_patches_list(image_patches_list, rotation_angle=90)
    color_augmented_patches_list = color_augment_patches_list(image_patches_list, ew, ev, ca_std)
    mask_patches_list = slice_the_mask_into_patches(mask, patch_h, patch_w)

    save_images_in_patches_list(image_patches_list, "image_patches_list")
    save_images_in_patches_list(rotated_patches_list, "rotated_patches_list")
    save_images_in_patches_list(color_augmented_patches_list, "color_augmented_patches_list")
    save_images_in_patches_list(mask_patches_list, "mask_patches_list")

    resized_image_patches_list = resize_patches_in_patches_list(image_patches_list, 256, 256)
    save_images_in_patches_list(resized_image_patches_list, "z_resized_image_patches_list")


    combined_image = combine_pathes_into_image(image_patches_list)
    cv2.imwrite("combined_image.jpg", combined_image)

    combined_mask = combine_pathes_into_mask(mask_patches_list)
    cv2.imwrite("combined_mask.jpg", 255*combined_mask)

    nh = 30 # Hight of resized patches (height of labels for the neural network)
    nw = 30 # Width of resized patches (width of labels for the neural network)
    resized_patches_list = resize_patches_in_patches_list(mask_patches_list, nh, nw)
    save_images_in_patches_list(resized_patches_list, "resized_patches_list")

    resized_back_patches_list = resize_patches_in_patches_list(resized_patches_list, patch_h, patch_w)
    save_images_in_patches_list(resized_back_patches_list, "resized_back_patches_list")



    diff_patches_list = diff_two_patches_lists_with_masks(mask_patches_list, resized_back_patches_list)
    save_images_in_patches_list(diff_patches_list, "diff_patches_list")


    images_masked_with_resized_patches_list = apply_mask_patches_list_to_image_patches_list(resized_back_patches_list,
                                                                                            image_patches_list)
    save_images_in_patches_list(images_masked_with_resized_patches_list, "1_images_masked_with_resized_patches_list")
    
    #cv2.imwrite("0_image_patches_list.jpg", image_patches_list[5][4])
    #cv2.imwrite("0_images_masked_with_resized_patches_list.jpg", images_masked_with_resized_patches_list[5][4])

    print_image_sizes(filename_list)
    mask_a_few_lion_images(filename_list)

    sap_list = [SAP(0.0, 1.0), 
                SAP(90.0, 1.0), 
                SAP(180.0, 1.0), 
                SAP(270.0, 1.0),
                SAP(0.0, 0.75), 
                SAP(90.0, 0.75), 
                SAP(180.0, 0.75), 
                SAP(270.0, 0.75),
                SAP(0.0, 0.25), 
                SAP(90.0, 0.25), 
                SAP(180.0, 0.25), 
                SAP(270.0, 0.25),
                SAP(0.0, 0.5), 
                SAP(90.0, 0.5), 
                SAP(180.0, 0.5),
                SAP(270.0, 0.5)]

    image_patches_lists_collection = create_collection_of_rotated_patches_lists(image_patches_list, sap_list)
    #save_collection_of_patches_lists(patches_lists_collection, "collection_check")

    mask_patches_lists_collection = create_collection_of_rotated_patches_lists(mask_patches_list, sap_list)

    image_patches_lists_collection = color_augment_patches_lists_collection(image_patches_lists_collection, ew, ev, ca_std)
    #save_collection_of_patches_lists(patches_lists_collection, "ca_collection_check")


    #pickle_patches_lists_collection("temp_pickle.pkl", patches_lists_collection)
    #size_of_collection = sys.getsizeof(patches_lists_collection)
    #print(size_of_collection)

    #savez_two_patches_list("/home/tadek/Coding/Kaggle/SeaLionPopulation/temp/checking_savez", mask_patches_list, image_patches_list)    
    


    savez_patches_list_collection("/home/tadek/Coding/Kaggle/SeaLionPopulation/temp/checking_savez",
                                  image_patches_lists_collection,
                                  mask_patches_lists_collection)

    load_files_in_folder("/home/tadek/Coding/Kaggle/SeaLionPopulation/temp/")



if __name__ == '__main__':

    train_image_filename_list = get_filename_list_in_dir(CONST_TRAIN_IMAGES_DIR, file_type="jpg")
    train_dotted_image_filename_list = get_filename_list_in_dir(CONST_TRAIN_DOTTED_IMAGES_DIR, file_type="jpg")

    print(train_image_filename_list)
    print(train_dotted_image_filename_list)

    train_image_filename = train_image_filename_list[9]
    train_dotted_image_filename = train_dotted_image_filename_list[9]

    print(train_image_filename)
    print(train_dotted_image_filename)

    filename_stem_train = get_filename_stem(train_image_filename)
    filename_stem_train_dotted = get_filename_stem(train_dotted_image_filename)
    
    print(filename_stem_train)
    print(filename_stem_train_dotted)

    #preprocessed_data_dir = "/home/tadek/Coding/Kaggle/SeaLionPopulation/temp/"
    #prepare_and_dispatch_data(train_image_filename_list, 
    #                          train_dotted_image_filename_list,
    #                          preprocessed_data_dir)


    #load_single_file("/home/tadek/Coding/Kaggle/SeaLionPopulation/temp/1_prep_data_c_1_i_4_j_5.npz")
    
    
    collection_of_resized_image_patches_lists, collection_of_resized_mask_patches_lists = prepare_single_full_input_image(train_image_filename, 
                                                                                                               train_dotted_image_filename,
                                                                                                               patch_h=500,
                                                                                                               patch_w=500,
                                                                                                               resize_image_patch_to_h=192,
                                                                                                               resize_image_patch_to_w=192,
                                                                                                               resize_mask_patch_to_h=48,
                                                                                                               resize_mask_patch_to_w=48,
                                                                                                               radious_list = [24, 24, 24, 16, 24],
                                                                                                               sap_list=[SAP(0.0, 1.0), SAP(90.0, 1.0)], 
                                                                                                               interactive_plot=False,
                                                                                                               display_every=87)
    

    manual_test_of_lion_counting(train_image_filename, 
                                 train_dotted_image_filename,
                                 patch_h=500,
                                 patch_w=500,
                                 resize_image_patch_to_h=192,
                                 resize_image_patch_to_w=192,
                                 resize_mask_patch_to_h=48,
                                 resize_mask_patch_to_w=48,
                                 radious_list = [24, 24, 24, 16, 24],
                                 sap_list=[SAP(0.0, 1.0), SAP(90.0, 1.0)], 
                                 interactive_plot=False,
                                 display_every=87)


    #filename_list = [CONST_TRAIN_DOTTED_IMAGES_DIR + str(i) + ".jpg" for i in range(10 + 1)]

    #dispatch_preprocessed_data(filename_list)

    #manual_testing()

    #filename_list = [CONST_TRAIN_DOTTED_IMAGES_DIR + str(i) + ".jpg" for i in range(10 + 1)]

    #ew, ev = get_data_eigenvalues_and_eigenvectors(filename_list, fraction=1)
    #ca_std=0.5

    #image = cv2.imread(filename_list[8])

    #patch_h = 500
    #patch_w = 500
    #image_patches_list = slice_the_image_into_patches(image, patch_h, patch_w)

















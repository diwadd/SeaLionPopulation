import sys
import math

import cv2
import numpy as np

TRAIN_DOTTED_DIR = "/home/tadek/Coding/Kaggle/SeaLionPopulation/TrainSmall/TrainDotted/"
WHITE_COLOR = (255, 255, 255)

def detect_dots_in_image(image, color="MAGENTA"):
    """
    This function takes an RBG image with sea lions.
    The lions should be marked with colored dots.
    
    The colored dots of a given color are detected and an image
    with the detected dots is returned. The background is black.
    The color of the dots is not changed.

    The colors are detected with appropriate lower and upper RBG
    bounds and a bitwise_and function.

    :param image:
    :param color: Color of the dots you want to detect.
    :return dotted_image:
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

    :param dotted_image: This is the output from read_image_detect_dots.
    :param radious:
    :param dot_threshold:
    :return mask:
    """

    gray_image = cv2.cvtColor(dotted_image, cv2.COLOR_BGR2GRAY)

    _, thresholded_image = cv2.threshold(gray_image, dot_threshold, 255, cv2.THRESH_BINARY)
    _, contours, _ = cv2.findContours(thresholded_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    mask = thresholded_image
    for i in range(len(contours)):
        x = contours[i][0][0][0]
        y = contours[i][0][0][1]
        mask = cv2.circle(mask,(x, y), radious, WHITE_COLOR, -1)

    mask = mask/255.0

    return mask
    

def apply_mask(image, mask):
    """
    Applies a mask to an image.

    The size of the image should be N x M x 3.
    The mask should be a N x M array with 0.0 or 1.0 values.
    
    :param image:
    :param mask:
    :return:
    """

    image[:,:,0] = image[:,:,0]*mask
    image[:,:,1] = image[:,:,1]*mask
    image[:,:,2] = image[:,:,2]*mask

    return image


def mask_the_lion_image(image):
    """
    Takes an image in which the sea lions are marked with coloered dots.

    Calculates a mask and then applies this mask to the input image.
    As a result only the lions remain visible in the original image.
    Each lion is located within a circle whose radious is defined in radious_list.

    The masked_lion_image that is returned is an image on which only the
    lions are visible. All other things are black.

    :params image:
    :return masked_lion_image:
    "return mask:
    """
    h, w, _ = image.shape

    color_list = ["MAGENTA", "RED", "BLUE", "GREEN", "BROWN"]

    # Each lion is marked with a circle.
    # The radious of the circle is adjusted to the
    # lion's size.
    radious_list = [45, 50, 40, 22, 42]
    mask = np.zeros((h, w))

    for c in range(len(color_list)):       

        dotted_image = detect_dots_in_image(image, color_list[c])
        
        mask_of_a_give_color = plot_circles_return_mask(dotted_image, radious=radious_list[c])
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

    :param filename_list:
    """

    n_files = len(filename_list)
    for i in range(n_files):
        print("Processed: %f" % ((i + 1)/n_files), end="\r")

        image = cv2.imread(filename_list[i])
        masked_lion_image, mask = mask_the_lion_image(image)

        cv2.imshow("masked_lion_image", masked_lion_image)
        masked_lion_image_filename = filename_list[i].replace(".jpg", "_x_masked_lion_image.jpg")
        cv2.imwrite(masked_lion_image_filename, masked_lion_image)

        cv2.imshow("mask", masked_lion_image)
        mask_filename = filename_list[i].replace(".jpg", "_x_mask.jpg")
        cv2.imwrite(mask_filename, 255*mask)
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

    :param w:
    :param h:
    :param patch_w:
    :param patch_h:
    :return nh:
    :return nw:
    :return nh_slices:
    :return nw_slices:
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
                                 patch_w = 400,
                                 prefix_text = None):
    """
    Takes an image and cuts it into patches.
    The size of the patch is patch_h x patch_w x patch_c.

    The function resizes the image before cutting.
    The image dimensions are set to new values so that
    they are multiples of patch_h and patch_w.

    A list of the pathes is returned.
    
    :param image:
    :param patch_h:
    :param patch_w:
    :return patches_list:
    """

    # Calculating the resized image dimensions.
    h, w, c = image.shape

    nh, nw, nh_slices, nw_slices = get_new_dimensions(h, w, patch_h, patch_w)

    # In the cv2.resize mathod the image shape is reverted (w, h) -> (h, w).
    resized_image = cv2.resize(image, (nw, nh), interpolation = cv2.INTER_LINEAR)
    
    if (prefix_text != None):
        cv2.imwrite(prefix_text + "resized_image.jpg", resized_image)

    patches_list = [[np.zeros((patch_h, patch_w, c), dtype=np.uint8) for j in range(nw_slices)] for i in range(nh_slices)]

    for i in range( nh_slices ):
        for j in range( nw_slices ):  
            patches_list[i][j][:,:,:] = resized_image[(i*patch_h):(i*patch_h + patch_h), (j*patch_w):(j*patch_w + patch_w), :]
            
            if (prefix_text != None):
                cv2.imwrite(prefix_text + str(patch_w) + "_" + str(patch_h) + "_%d_%d_image.jpg" % (i, j), patches_list[i][j])

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
    
    :param image:
    :param patch_h:
    :param patch_w:
    :return patches_list:
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

    :param patches_list:
    :return nh_slices, nw_slices:
    """

    nh_slices = len(patches_list)
    if (nh_slices == 0):
        sys.exit("The provided patches_list has zero length!")
        

    nw_slices = len(patches_list[0])

    return nh_slices, nw_slices



def combine_pathes_into_image(patches_list, prefix_text = None):
    """
    Takes a patches_list returned by slice_the_image_into_patches and
    combines it back into a full image.

    :param patches_list:
    :return image:
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

    :param patches_list:
    :return image:
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

    :param patch:
    :param nh:
    :param nw:
    :return resized_mask_patch:
    """

    resized_mask_patch = cv2.resize(patch, (nw, nh), interpolation = cv2.INTER_LINEAR)

    return resized_mask_patch


def resize_patches_list_with_masks(patches_list, nh, nw):
    """
    Takes a list of mask patches and resizes each patch.
    The new size of the patch is nh x nw.
    
    :param patches_list:
    :param nh:
    :param nw:
    :param prefix_text:
    :return:
    """

    nh_slices, nw_slices = get_patches_list_dimensions(patches_list)
    patch_h, patch_w = patches_list[0][0].shape
    
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

    :param patches_list_1:
    :param patches_list_2:
    :param prefix_text:   
    :return diff_patches_list:
    """

    nh_slices_1, nw_slices_1 = get_patches_list_dimensions(patches_list_1)
    nh_slices_2, nw_slices_2 = get_patches_list_dimensions(patches_list_2)

    if (nh_slices_1 != nh_slices_2) or (nw_slices_1 != nw_slices_2):
        sys.exit("Patches_lists dimension mismatch!")

    patch_h_1, patch_w_1 = patches_list_1[0][0].shape
    patch_h_2, patch_w_2 = patches_list_2[0][0].shape

    if (patch_h_1 != patch_h_2) or (patch_w_1 != patch_w_2):
        sys.exit("Patch dimension mismatch!")


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

    :param mask_patches_list:
    :param image_patches_list:
    :return patches_list:
    """

    nh_slices_m, nw_slices_m = get_patches_list_dimensions(mask_patches_list)
    nh_slices_i, nw_slices_i = get_patches_list_dimensions(image_patches_list)

    if (nh_slices_m != nh_slices_i) or (nw_slices_m != nw_slices_i):
        sys.exit("Patches_lists dimension mismatch!")

    patch_h_m, patch_w_m = mask_patches_list[0][0].shape
    patch_h_i, patch_w_i, patch_c = image_patches_list[0][0].shape

    if (patch_h_m != patch_h_i) or (patch_w_m != patch_w_i):
        sys.exit("Patch dimension mismatch!")

    patches_list = [[np.zeros((patch_h_i, patch_w_i, patch_c), dtype=np.uint8) for j in range(nw_slices_i)] for i in range(nh_slices_i)]
    

    for i in range(nh_slices_m):
        for j in range(nw_slices_m):

            # We want to avoid modyfing the original image.
            patch = np.array(image_patches_list[i][j])
            patches_list[i][j] = apply_mask(patch, 
                                            mask_patches_list[i][j])

    return patches_list


def save_images_in_patches_list(patches_list, image_filename):

    nh_slices, nw_slices = get_patches_list_dimensions(patches_list)

    if (nh_slices == 0) or (nw_slices == 0):
        sys.exit("The patches_list has zero dimenssions!")


    # Check is patches_list contains masks.
    # Mask should contain only values between 0 and 1.
    # Images on the otherhand values between 0 and 255.
    # Images with values [0,1] will be considered as masks
    # and rescaled to 255 when saving.
    is_mask = False
    max_value = -math.inf
    for i in range(nh_slices):
        for j in range(nw_slices):
            value = np.max(patches_list[i][j])
    
            if (value > max_value):
                max_value = value

    print(image_filename + " max_value: " + str(max_value))

    if (round(max_value, 1) <= 1.0):
        is_mask = True

    for i in range(nh_slices):
        for j in range(nw_slices):

            if (is_mask == True):      
                cv2.imwrite(image_filename + "_" + str(i) + "_" + str(j) + ".jpg", 255*patches_list[i][j])
            else:
                cv2.imwrite(image_filename + "_" + str(i) + "_" + str(j) + ".jpg", patches_list[i][j])



def print_image_sizes(filename_list):

    for i in range(len(filename_list)):
        image = cv2.imread(filename_list[i])
        print(image.shape)


if __name__ == '__main__':

    """
    for i in range(0,10 + 1):

        image = cv2.imread(TRAIN_DOTTED_DIR + str(i) + ".jpg")

        dotted_image_mag = detect_dots_in_image(image, "MAGENTA")
        #dotted_image_red = detect_dots_in_image(image, "RED")
        #dotted_image_blu = detect_dots_in_image(image, "BLUE")
        #dotted_image_gre = detect_dots_in_image(image, "GREEN")
        #dotted_image_bro = detect_dots_in_image(image, "BROWN")

    #plot_circles_prototype(dotted_image_mag)

    mask_mag = plot_circles(dotted_image_mag)

    image = apply_mask(image, mask_mag)

    cv2.imshow("image", image)
    #cv2.waitKey(0)
    cv2.imwrite("image.jpg", image)
    """

    #image = cv2.imread(TRAIN_DOTTED_DIR + "0.jpg")
    #masked_lion_image = mask_the_lion_image(image)

    #cv2.imshow("image", masked_lion_image)
    #cv2.imwrite("image.jpg", masked_lion_image)

    filename_list = [TRAIN_DOTTED_DIR + str(i) + ".jpg" for i in range(10 + 1)]

    image = cv2.imread(filename_list[0])
    cv2.imwrite("original_image.jpg", image)

    masked_lion_image, mask = mask_the_lion_image(image)
    cv2.imwrite("original_mask.jpg", 255*mask)


    patch_h = 500
    patch_w = 500
    image_patches_list = slice_the_image_into_patches(masked_lion_image, patch_h, patch_w)
    save_images_in_patches_list(image_patches_list, "image_patches_list")

    mask_patches_list = slice_the_mask_into_patches(mask, patch_h, patch_w)
    save_images_in_patches_list(mask_patches_list, "mask_patches_list")


    combined_image = combine_pathes_into_image(image_patches_list)
    cv2.imwrite("combined_image.jpg", combined_image)

    combined_mask = combine_pathes_into_mask(mask_patches_list)
    cv2.imwrite("combined_mask.jpg", 255*combined_mask)

    nh = 30
    nw = 30
    resized_patches_list = resize_patches_list_with_masks(mask_patches_list, nh, nw)
    save_images_in_patches_list(resized_patches_list, "resized_patches_list")

    resized_back_patches_list = resize_patches_list_with_masks(resized_patches_list, patch_h, patch_w)
    save_images_in_patches_list(resized_back_patches_list, "resized_back_patches_list")



    diff_patches_list = diff_two_patches_lists_with_masks(mask_patches_list, resized_back_patches_list)
    save_images_in_patches_list(diff_patches_list, "diff_patches_list")

    images_masked_with_resized_patches_list = apply_mask_patches_list_to_image_patches_list(resized_back_patches_list,
                                                                                            image_patches_list)
    save_images_in_patches_list(images_masked_with_resized_patches_list, "1_images_masked_with_resized_patches_list")
    
    cv2.imwrite("0_image_patches_list.jpg", image_patches_list[5][4])
    cv2.imwrite("0_images_masked_with_resized_patches_list.jpg", images_masked_with_resized_patches_list[5][4])

    print_image_sizes(filename_list)
    mask_a_few_lion_images(filename_list)










import cv2
import numpy as np

TRAIN_DOTTED_DIR = "/home/tadek/Coding/Kaggle/SeaLionPopulation/TrainSmall/TrainDotted/"
WHITE_COLOR = (255, 255, 255)

def read_image_detect_dots(image, color="MAGENTA"):
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

    w, h, c = image.shape

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

    dotted_image = np.zeros((w,h,c), dtype=np.uint8)
    cv2.bitwise_and(image, image, dotted_image, mask=mask)

    #cv2.imwrite(image_filename.replace(".jpg", image_post_fix), dotted_image)

    return dotted_image


def plot_circles_return_mask(dotted_image, radious=40):
    """
    This function takes as input the output from read_image_detect_dots.
    It detects contours of each dot. The contours are basically a list/numpy array of [x, y]
    coordinates of pixels that surround the dots.

    The first pixels coordinates of the contours are taken and a circle
    is drawn around this pixel (it suffices to take the first pixel
    because the dots can be treated as point like).

    A mask image is returned (0 for the background, 1 for the circle). 

    :param dotted_image: This is the output from read_image_detect_dots.
    :param: radious
    :return mask:
    """

    gray_image = cv2.cvtColor(dotted_image, cv2.COLOR_BGR2GRAY)

    _, thresholded_image = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY)
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

    :params image:
    :return masked_lion_image:
    "return mask:
    """
    w, h, _ = image.shape

    color_list = ["MAGENTA", "RED", "BLUE", "GREEN", "BROWN"]

    # Each lion is marked with a circle.
    # The radious of the circle is adjusted to the
    # lion's size.
    radious_list = [45, 50, 40, 22, 42]
    mask = np.zeros((w, h))

    for c in range(len(color_list)):       

        dotted_image = read_image_detect_dots(image, color_list[c])
        
        mask_of_a_give_color = plot_circles_return_mask(dotted_image, radious=radious_list[c])
        mask = mask + mask_of_a_give_color
        
    mask[mask > 0] = 1.0
    masked_lion_image = apply_mask(image, mask)

    return masked_lion_image, mask


def mask_a_few_lion_images(filename_list):

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


def plot_circles_prototype(dotted_image):

    gray_image = cv2.cvtColor(dotted_image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("gray_image.jpg", gray_image)

    retval, thresholded_image = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY)
    cv2.imwrite("thresholded_image.jpg", thresholded_image)

    im2, contours, hierarchy = cv2.findContours(thresholded_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    print("--- Contours ---")    
    print(contours)
    print(type(contours))
    print(len(contours))
    print(contours[0])
    print(contours[0][0])    
    print(contours[0][0][0])
    print(contours[0][0][0][0])
    print(contours[0][0][0][1])
    print(type(contours[0][0][0][0]))

    x = contours[0][0][0][0]
    y = contours[0][0][0][1]

    cv2.drawContours(dotted_image, contours, -1, (255, 0, 255), 1)

    cv2.imshow("Detected contours", dotted_image)
    cv2.imwrite("temp.jpg", dotted_image)

    for i in range(len(contours)):
        dotted_image = cv2.circle(dotted_image,(contours[i][0][0][0], contours[i][0][0][1]), 200, (255, 255, 255), -1)

    cv2.imshow("Detected contours", dotted_image)
    cv2.imwrite("circle.jpg", dotted_image)

"""
for i in range(0,10 + 1):

    image = cv2.imread(TRAIN_DOTTED_DIR + str(i) + ".jpg")

    dotted_image_mag = read_image_detect_dots(image, "MAGENTA")
    #dotted_image_red = read_image_detect_dots(image, "RED")
    #dotted_image_blu = read_image_detect_dots(image, "BLUE")
    #dotted_image_gre = read_image_detect_dots(image, "GREEN")
    #dotted_image_bro = read_image_detect_dots(image, "BROWN")

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

mask_a_few_lion_images(filename_list)










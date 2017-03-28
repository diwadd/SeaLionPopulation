import cv2
import numpy as np

TRAIN_DOTTED_DIR = "/home/tadek/Coding/Kaggle/SeaLionPopulation/TrainSmall/TrainDotted/"

# RBG in OpenCV is BGR = [BLUE, GREEN RED]

DOTTED_IMAGE_MAGENTA_POSTFIX = "_dots_mag.jpg"
MAGENTA_RGB_LOWER_BOUND = np.array([225,  0, 225], dtype=np.uint8)
MAGENTA_RGB_UPPER_BOUND = np.array([255, 30, 255], dtype=np.uint8)

DOTTED_IMAGE_RED_POSTFIX = "_dots_red.jpg"
RED_RGB_LOWER_BOUND = np.array([ 0,  0, 225], dtype=np.uint8)
RED_RGB_UPPER_BOUND = np.array([30, 30, 255], dtype=np.uint8)

DOTTED_IMAGE_BLUE_POSTFIX = "_dots_blu.jpg"
BLUE_RGB_LOWER_BOUND = np.array([140, 40, 15], dtype=np.uint8)
BLUE_RGB_UPPER_BOUND = np.array([255, 80, 55], dtype=np.uint8)

DOTTED_IMAGE_GREEN_POSTFIX = "_dots_gre.jpg"
GREEN_RGB_LOWER_BOUND = np.array([5, 145, 20], dtype=np.uint8)
GREEN_RGB_UPPER_BOUND = np.array([55, 195, 70], dtype=np.uint8)

DOTTED_IMAGE_BROWN_POSTFIX = "_dots_bro.jpg"
BROWN_RGB_LOWER_BOUND = np.array([ 0, 20,  55], dtype=np.uint8)
BROWN_RGB_UPPER_BOUND = np.array([20, 60, 105], dtype=np.uint8)


def read_image(image_filename, color="MAGENTA"):

    image = cv2.imread(image_filename)
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


    grayscale_dotted_image = cv2.cvtColor( dotted_image, cv2.COLOR_RGB2GRAY )
    #image_max = np.max(grayscale_dotted_image)
    #image_min = np.min(grayscale_dotted_image)
    #grayscale_dotted_image = ( grayscale_dotted_image > int( (image_max - image_min)/2.0 ) )
    #grayscale_dotted_image = grayscale_dotted_image.astype(np.uint8)

    cv2.imwrite(image_filename.replace(".jpg", image_post_fix), grayscale_dotted_image)


read_image(TRAIN_DOTTED_DIR + "0.jpg")
read_image(TRAIN_DOTTED_DIR + "0.jpg", "RED")
read_image(TRAIN_DOTTED_DIR + "0.jpg", "BLUE")
read_image(TRAIN_DOTTED_DIR + "0.jpg", "GREEN")
read_image(TRAIN_DOTTED_DIR + "0.jpg", "BROWN")



















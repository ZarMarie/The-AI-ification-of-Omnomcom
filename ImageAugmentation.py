import cv2
import numpy as np
import random
import os

def process_image(image_path):
    """
    :param image_path:
    None, but if an image path is given, it will be read and processed

    :param image:
    None, but if an image is given, it will take the place of the image from the image path

    :return:
    A resized version of the image to be compatible with a vector classifier
    """

    original_image = cv2.imread(image_path) # original refers to the title of the window

    sobel_kernel = np.array([[-1, 0, 1,
                            -2, 0, 2,
                            -1, 0, 1]])

    custom_convolution = cv2.filter2D(original_image, -1, sobel_kernel)
    custom_convolution = cv2.resize(custom_convolution, (224,224))
    return custom_convolution


def flip_image(image_path):
    """
    :param image_path:
    The filepath of the image to be flipped

    :return:
    A flipped version of the image
    """
    image = cv2.imread(image_path)
    return cv2.flip(image, 1)


def rotate_image(image, step):
    """
    :param image_path:
    The filepath of the image to be rotated

    :param step:
    The number of 90 degree steps to rotate the image

    :return:
    A rotated version of the image
    """

    if step == 1:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif step == 2:
        return cv2.rotate(image, cv2.ROTATE_180)
    elif step == 3:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)


def show_image(image):
    """
    :param image_path:
    Initially none if image array is given; the filepath of the image to be displayed

    :param image:
    Initially none if an image filepath is given; the image array to be displayed

    :return:
    A window displaying the image that is closed when a key is pressed
    """

    cv2.imshow(" original ", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




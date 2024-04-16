import cv2
import numpy as np
import random
import os

def process_image(image_path=None, image=1):
    """
    :param image_path:
    None, but if an image path is given, it will be read and processed

    :param image:
    None, but if an image is given, it will take the place of the image from the image path

    :return:
    A resized version of the image to be compatible with a vector classifier
    """

    if not type(image) == int:
        original_image = image
    else:
        original_image = cv2.imread(image_path) # original refers to the title of the window

    custom_convolution = cv2.resize(original_image, (224,224))

    # Split the image into its channel components
    channels = cv2.split(custom_convolution)


    sobel_channels = [cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=3) + cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=3) for channel in channels]
    sobel_channels = [np.uint8(np.absolute(channel)) for channel in sobel_channels]
    custom_convolution = cv2.merge(sobel_channels)

    return custom_convolution


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


def brightness_image(image, low, high):

    value = random.uniform(low, high)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype=np.float64)
    hsv[:, :, 1] = hsv[:, :, 1]*value
    hsv[:, :, 1][hsv[:, :, 1] > 255] = 255
    hsv[:, :, 2] = hsv[:, :, 2]*value
    hsv[:, :, 2][hsv[:, :, 2] > 255] = 255
    hsv = np.array(hsv, dtype=np.uint8)
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return image


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




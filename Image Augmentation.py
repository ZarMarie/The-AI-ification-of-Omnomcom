import cv2


def pre_process(image_path=None, image=None):
    """
    :param image_path:
    None, but if an image path is given, it will be read and processed

    :param image:
    None, but if an image is given, it will take the place of the image from the image path

    :return:
    A greyscaled, resized and flattened version of the image to be compatible with a vector classifier
    """

    if not isinstance(image, list) and image_path != None:
        original_image = cv2.imread(image_path) # original refers to the title of the window
    else:
        original_image = image

    sobel_kernel = np.array([[-1, 0, 1,
                            -2, 0, 2,
                            -1, 0, 1]])

    custom_convolution = cv2.filter2D(original_image, -1, sobel_kernel)
    custom_convolution = cv2.cvtColor(custom_convolution, cv2.COLOR_RGB2GRAY)
    custom_convolution = cv2.resize(custom_convolution, (128,128))
    custom_convolution = custom_convolution.flatten()
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


def rotate_image(image_path, step):
    """
    :param image_path:
    The filepath of the image to be rotated

    :param step:
    The number of 90 degree steps to rotate the image

    :return:
    A rotated version of the image
    """
    image = cv2.imread(image_path)

    if step == 1:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif step == 2:
        return cv2.rotate(image, cv2.ROTATE_180)
    elif step == 3:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)


def show_image(image_path=None, image=None):
    """
    :param image_path:
    Initially none if image array is given; the filepath of the image to be displayed

    :param image:
    Initially none if an image filepath is given; the image array to be displayed

    :return:
    A window displaying the image that is closed when a key is pressed
    """

    if image == None and image_path != None:
        image = cv2.imread(image_path)
    else:
        return None
    cv2.imshow(" original ", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()






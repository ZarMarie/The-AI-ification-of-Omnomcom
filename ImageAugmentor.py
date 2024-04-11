from ImageAugmentation import rotate_image, show_image, process_image, brightness_image
import os
import random
import cv2
import numpy as np

for folder in os.listdir('data'):
    for file in os.listdir(f'data/{folder}'):
        image_path = f'data/{folder}/{file}'

        # Split the image into its channel components
        channels = cv2.split(process_image(image_path=image_path))

        # Apply sobel filter to each color channel separately and then combine them to maintain important color data
        sobel_channels = [cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=3) + cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=3) for channel in channels]
        sobel_channels = [np.uint8(np.absolute(channel)) for channel in sobel_channels]
        sobel_colored = cv2.merge(sobel_channels)

        rotation1, rotation2 = random.sample(range(1, 4), 2)

        augmented_images = {
            'original': sobel_colored,
            'rotated1': rotate_image(sobel_colored, rotation1),
            'rotated2': rotate_image(sobel_colored, rotation2),
            'bright1': brightness_image(sobel_colored, 0.5, 1.5),
            'bright2': brightness_image(rotate_image(sobel_colored, rotation1), 0.5, 1.5),
            'bright3': brightness_image(rotate_image(sobel_colored, rotation2), 0.5, 1.5)
        }

        for key, value in augmented_images.items():
            cv2.imwrite(f"augmented data/{folder}/{file[:-4]}-{key}.png", value)


import pygame
from cv2 import VideoCapture, waitKey
import cv2
from pygame.locals import *
from tensorflow.keras import models
import numpy as np

pygame.init()

cam_port = 0
cam = VideoCapture(cam_port)

result, image = cam.read()

rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
rgb_image = cv2.resize(rgb_image, (960, 540))
rgb_image = cv2.flip(rgb_image, 1)


height, width = rgb_image.shape[:2]

# Create a Pygame window
window = pygame.display.set_mode((width, height))

# Load the model
# model = models.load_model('trained_models/third_model.keras')

running = True
while running:

    result, image = cam.read()

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb_image = cv2.resize(rgb_image, (960, 540))
    rgb_image = cv2.flip(rgb_image, 1)

    # Convert the RGB image into a Pygame surface
    pygame_image = pygame.surfarray.make_surface(rgb_image.transpose(1, 0, 2))

    # Blit the image onto the window
    window.blit(pygame_image, (0, 0))

    # show window
    pygame.display.flip()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

pygame.quit()
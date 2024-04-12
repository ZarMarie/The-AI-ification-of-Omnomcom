import pygame
from cv2 import VideoCapture, waitKey
from pygame.locals import *

pygame.init()

cam_port = 1
cam = VideoCapture(cam_port)

while True:
    k = waitKey(5)
    result, image = cam.read()

    # make window
    window = pygame.display.set_mode( (640,480), RESIZABLE )

    # copy image to window
    pygame.surfarray.blit_array(window, image)

    # show window
    pygame.display.flip()

    if k == ord('q'):
        pygame.quit()
        break
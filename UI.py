import pygame
from tkinter import messagebox
import threading

image_path = 'data/AppleJuice/1.png'
price = int


def make_popup(image_path, price, window, product_name):
    # make popup menu in white rectangle
    pygame.draw.rect(window, (255, 255, 255), (50, 50, 860, 440))

    # load and place image in the center of the popup
    image = pygame.image.load(image_path)
    window.blit(image, (480 - image.get_width() // 2, 60))

    # make and place price text
    font = pygame.font.Font(None, 36)
    text = font.render(f"Price: ${price}", True, (0, 0, 0))
    window.blit(text, (480 - text.get_width() // 2, image.get_height()))

    # adding green and red yes and no button with text
    yes_button = pygame.draw.rect(window, (0, 255, 0), (480 - 100, image.get_height() + text.get_height() + 20, 100, 40))
    no_button = pygame.draw.rect(window, (255, 0, 0), (480, image.get_height() + text.get_height() + 20, 100, 40))
    text = font.render("Yes", True, (0, 0, 0))
    window.blit(text, (480 - 100 + 50 - text.get_width() // 2, image.get_height() + 20 + text.get_height() + 10))
    text = font.render("No", True, (0, 0, 0))
    window.blit(text, (480 + 50 - text.get_width() // 2, image.get_height() + 20 + text.get_height() + 10))

    # display the window
    pygame.display.flip()

    # check if user clicked yes or no; close program and print the selected program on yes, and close the popup on no
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                if yes_button.collidepoint(x, y):
                    print(f"\n_________________________\nAdded {product_name} to cart.\n_________________________")
                    pygame.quit()
                    quit()
                elif no_button.collidepoint(x, y):
                    return None

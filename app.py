import pygame
from cv2 import VideoCapture, waitKey
import cv2
from pygame.locals import *
from tensorflow.keras import models
import numpy as np
from ImageAugmentation import process_image
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


encoding_dict = {
    0 : 'Doritos Honey BBQ',
    1 : 'Tosti',
    2 : 'Apple Bandit',
    3 : 'Proto Tie',
    4 : 'Apple Juice',
    5 : 'Smint',
    6 : 'Ice Tea Peach',
    7 : 'Kinder Bueno',
    8 : 'Frog',
    9 : 'Lolipop'
}

pygame.init()

cam_port = 0
cam = VideoCapture(cam_port)

result, image = cam.read()

rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
rgb_image = cv2.resize(rgb_image, (960, 540))
rgb_image = cv2.flip(rgb_image, 1)

height, width = rgb_image.shape[:2]

print(process_image(image=rgb_image).shape)

# Create a Pygame window
window = pygame.display.set_mode((width, height))

base_model = tf.keras.applications.VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

pooling_layer = tf.keras.layers.GlobalAveragePooling2D()
dense_layer_1 = tf.keras.layers.Dense(1024, activation='relu')
dense_layer_2 = tf.keras.layers.Dense(512, activation='relu')
output_layer = tf.keras.layers.Dense(10, activation='softmax')

model = tf.keras.models.Sequential([
    base_model,
    pooling_layer,
    dense_layer_1,
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.BatchNormalization(),
    dense_layer_2,
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.BatchNormalization(),
    output_layer
])

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Build the model
model.build((None, 224, 224, 3))

model.load_weights('trained_models/fifth_model.weights.h5')

print(model.layers[0].input_shape)

pred_encoding = []
actual_encoding = []

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

    processed_image = process_image(image=rgb_image)

    # cv2.imshow("current", processed_image)

    processed_image = np.expand_dims(processed_image, axis=0)

    encoding_pred = model.predict(processed_image)[0]

    if np.max(encoding_pred) > 0.9:
        print(encoding_dict[np.argmax(encoding_pred)])

    # pred_encoding.append(encoding_pred)

    # k = cv2.waitKey(0)
    # if k == ord(' '):
    #     break
    #
    # if chr(k).isalnum():
    #     actual_encoding.append(int(chr(k)))
    #     cv2.destroyAllWindows()
    #     print(encoding_dict[encoding_pred], int(chr(k)))
    #
    # for event in pygame.event.get():
    #     if event.type == pygame.QUIT:
    #         running = False

# Confusion Matrix
plt.figure()
conf_matrix = confusion_matrix(actual_encoding, pred_encoding)

sns.heatmap(conf_matrix, annot=True, fmt="d")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title('Confusion Matrix')
plt.show()

cv2.destroyAllWindows()
pygame.quit()
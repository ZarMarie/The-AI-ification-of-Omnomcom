import pygame
from cv2 import VideoCapture
import cv2
import numpy as np
from ImageAugmentation import process_image
import tensorflow as tf
from UI import make_popup

# dictionary to for what number class is which product with its price
encoding_dict = {
    0 : ('Doritos Honey BBQ', "0.50"),
    1 : ('Tosti', "1.10"),
    2 : ('Apple Bandit', "1.35"),
    3 : ('Proto Tie', "12.00"),
    4 : ('Apple Juice', "0.70"),
    5 : ('Smint', "2.00"),
    6 : ('Ice Tea Peach', "1.10"),
    7 : ('Kinder Bueno', "1.00"),
    8 : ('Frog', "0.75"),
    9 : ('Lollipop', "0.15")
}

pygame.init()

# setup camera and get first image to base window and size off of it
cam_port = 0
cam = VideoCapture(cam_port)

result, image = cam.read()

# Convert the image to RGB, resize it, and flip it so that it is displayed correctly
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
rgb_image = cv2.resize(rgb_image, (960, 540))
rgb_image = cv2.flip(rgb_image, 1)

height, width = rgb_image.shape[:2]

print(process_image(image=rgb_image).shape)

# Create a Pygame window
window = pygame.display.set_mode((width, height))

# Create a new model with the exact same layers as the trained one in main.py
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

# Load the weights of the trained model onto the model
model.load_weights('trained_models/fifth_model.weights.h5')

# Print the model summary
model.summary()

print(model.layers[0].input_shape)

# Variables to keep track of the last item predicted and the count of the same item predicted
count = 0
last = ""
popup = False

running = True
while running:
    # Get the image from the camera
    result, image = cam.read()

    # Convert the image to RGB, resize it, and flip it so that it is displayed correctly
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb_image = cv2.resize(rgb_image, (960, 540))
    rgb_image = cv2.flip(rgb_image, 1)

    # Convert the RGB image into a Pygame surface
    pygame_image = pygame.surfarray.make_surface(rgb_image.transpose(1, 0, 2))

    # Blit the image onto the window
    window.blit(pygame_image, (0, 0))

    # show window
    pygame.display.flip()

    # Process the image and make a prediction
    processed_image = process_image(image=rgb_image)
    processed_image = np.expand_dims(processed_image, axis=0)

    encoding_pred = model.predict(processed_image, verbose=0)[0]

    item_prediction = encoding_dict[np.argmax(encoding_pred)][0]

    # If the prediction is above 95%, print the prediction and increment the same in a row counter if it is the same
    if np.max(encoding_pred) > 0.95:
        print(item_prediction)
        if item_prediction != last:
            count = 0
        else:
            count += 1
        last = item_prediction

    # If the same item has been predicted 10 times in a row, make a popup with the image of the item and its price
    if count == 10:
        make_popup(f"product images/{item_prediction}.jpeg", encoding_dict[np.argmax(encoding_pred)][1], window, item_prediction)
        count = 0
        last_item = ""

    # Check for events and exit out of the application properly if the window is close
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

# Close the camera and the Pygame window
cv2.destroyAllWindows()
pygame.quit()
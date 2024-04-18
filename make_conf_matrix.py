#
#
#
#
#
#
#
#
# _____________________________________________________________________________________________________________________
# This code makes the bad confusion matrix that just isn't correct :((
# _____________________________________________________________________________________________________________________
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#



from sklearn.metrics import confusion_matrix
import seaborn as sns
import tensorflow as tf
import pandas as pd
import os
import cv2
from ImageAugmentation import process_image, show_image
import numpy as np
import matplotlib.pyplot as plt

# rebuild the model exactly how it was made originally
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

# load the trained weights onto the built model
model.load_weights('trained_models/fifth_model.weights.h5')

# make a dataframe to store the images and their corresponding classes
df = pd.DataFrame(columns=['class', 'image'])

for encoding, folder in enumerate(os.listdir('augmented data')):
    for file in os.listdir(f'augmented data/{folder}'):
        df.loc[len(df.index)] = [encoding, cv2.imread(f"augmented data/{folder}/{file}")]


real_classes = []
pred_classes = []
# X = []
# y = []
processed_images = []

# loop through the dataframe and predict the class of each image, and store it in real classes and pred classes
for row in df.iterrows():
    image = row[1]['image']
    # processed_image = np.expand_dims(processed_image, axis=0)
    # pred_classes.append(np.argmax(model.predict(processed_image)))
    # X.append(processed_image)
    # y.append(row[1]['class'])
    real_classes.append(row[1]['class'])
    processed_image = process_image(image=image)
    show_image(processed_image)
    # processed_image = np.expand_dims(processed_image, axis=0)
    processed_images.append(processed_image)
    # pred_classes.append(np.argmax(model.predict(processed_image)))
    # print(f"Real Class: {real_classes[-1]}, Predicted Class: {pred_classes[-1]}" )

pred_classes = model.predict(np.array(processed_images))
pred_labels = []

# print all of the predictions and real classes
for m in range(len(pred_classes)):
    pred_labels.append(np.argmax(pred_classes[m]))
    print(f"Real Class: {real_classes[m]}, Predicted Class: {pred_labels[m]}")

# Make the confusion matrix seaborn heatmap as a pyplot figure and display it
plt.figure()
conf_matrix = confusion_matrix(real_classes, pred_labels)

sns.heatmap(conf_matrix, annot=True, fmt="d")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title('Confusion Matrix')
plt.show()

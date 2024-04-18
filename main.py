import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

# Create a dataframe to store the images and their corresponding classes
df = pd.DataFrame(columns=['class', 'image'])

for encoding, folder in enumerate(os.listdir('augmented data')):
    for file in os.listdir(f'augmented data/{folder}'):
        df.loc[len(df.index)] = [encoding, cv2.imread(f"augmented data/{folder}/{file}")]

# load the VGG16 model
base_model = tf.keras.applications.VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

# Create the layers for the model
pooling_layer = tf.keras.layers.GlobalAveragePooling2D()
dense_layer_1 = tf.keras.layers.Dense(1024, activation='relu')
dense_layer_2 = tf.keras.layers.Dense(512, activation='relu')
output_layer = tf.keras.layers.Dense(10, activation='softmax')

# Create and compile the model with previously defined layers as well as new dropout and batch normalization layers as
# these are dependent on the previous layers
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

# define features (X) and labels (y), and convert the labels to categorical so the model does not interpret them as
# numbers
X = np.array(df['image'].tolist())
y = np.array(df['class'].tolist())
y = tf.keras.utils.to_categorical(y, num_classes=10)

# Split the data into training, validation, and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

# Define the callbacks for the model (early stopping (es) and learning rate reduction (lr_reduction))
es = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', patience=20,  restore_best_weights=True)
lr_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=20, verbose=1, factor=0.5)

# Train the model and save the history of the training
history = model.fit(X_train, y_train, epochs=150, validation_split=0.2, verbose=2, callbacks=[lr_reduction, es])

# Save the model and its weights
model.save('trained_models/sixth_model.keras')
model.save_weights('trained_models/sixth_model.weights.h5')

# Evaluate the model on the test set
test = model.evaluate(X_test, y_test)
print(test)

# Get the training and validation loss from the history
loss = history.history['loss']

# Plot the training loss over the epochs
plt.plot(history.history['loss'], label='Train Loss', color='orange')
plt.plot(history.history['val_loss'], label='Validation Loss', color='blue')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.yscale('log')
plt.title('Training and Validation Loss Over Time')

# Display Legend and show the plot
plt.legend()
plt.show()



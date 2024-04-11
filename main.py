import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

df = pd.DataFrame(columns=['class', 'image'])

# Build dataframe
for encoding, folder in enumerate(os.listdir('augmented data')):
    for file in os.listdir(f'augmented data/{folder}'):
        df.loc[len(df.index)] = [encoding, cv2.imread(f"augmented data/{folder}/{file}")]

base_model = tf.keras.applications.VGG16(weights="imagenet", include_top=False)
base_model.trainable = False


pooling_layer = tf.keras.layers.GlobalAveragePooling2D()
dense_layer_1 = tf.keras.layers.Dense(1024, activation='relu')
dropout_layer = tf.keras.layers.Dropout(0.3)
dense_layer_2 = tf.keras.layers.Dense(512, activation='relu')
output_layer = tf.keras.layers.Dense(10, activation='softmax')

model = tf.keras.models.Sequential([
    base_model,
    pooling_layer,
    dense_layer_1,
    dropout_layer,
    tf.keras.layers.BatchNormalization(),
    dense_layer_2,
    dropout_layer,
    tf.keras.layers.BatchNormalization(),
    output_layer
])

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

X = np.array(df['image'].tolist())
y = np.array(df['class'].tolist())
y = tf.keras.utils.to_categorical(y, num_classes=10)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

es = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', patience=5,  restore_best_weights=True)
lr_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=5, verbose=1, factor=0.5)

history = model.fit(X_train, y_train, epochs=150, validation_split=0.2, verbose=2, callbacks=[lr_reduction, es])

model.save('trained_models/actual_first_model.keras')

test = model.evaluate(X_test, y_test)
print(test)

loss = history.history['loss']

# -- Plot the training loss over the epochs
plt.plot(history.history['loss'], label='Train Loss', color='orange')
plt.plot(history.history['val_loss'], label='Validation Loss', color='blue')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.yscale('log')
plt.title('Training and Validation Loss Over Time')

# -- Display Legend and show the plot
plt.legend()
plt.show()



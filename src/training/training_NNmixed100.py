from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard, LambdaCallback
from tensorflow.keras.layers import Input, Dropout, Dense, GlobalAveragePooling2D

from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, 
                                     GlobalAveragePooling2D, Input, Add, Activation, Multiply, Reshape)
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
import numpy as np
import itertools 
import datetime
import os
import cv2
import io
from tqdm import tqdm

from common.config import TRAINING_DIRECTORY, TESTING_DIRECTORY

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Initialize directories for saving logs
logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

# Load the labels and initialize other variables (similar to your original code)
labels = ['glioma', 'meningioma', 'notumor', 'pituitary']
image_size = 150

# Initialize lists for images and labels
x_train = []  # Training images
y_train = []  # Training labels
x_test = []   # Testing images
y_test = []   # Testing labels

# Your image loading and processing code (as in the original script)
for label in labels:
    trainPath = os.path.join(TRAINING_DIRECTORY, label)
    for file in tqdm(os.listdir(trainPath)):
        image = cv2.imread(os.path.join(trainPath, file), 0)  # Load images in gray scale
        image = cv2.bilateralFilter(image, 2, 50, 50)  # Remove image noise
        image = cv2.applyColorMap(image, cv2.COLORMAP_BONE)  # Apply pseudocolor
        image = cv2.resize(image, (image_size, image_size))  # Resize to 150x150
        x_train.append(image)
        y_train.append(labels.index(label))

    testPath = os.path.join(TESTING_DIRECTORY, label)
    for file in tqdm(os.listdir(testPath)):
        image = cv2.imread(os.path.join(testPath, file), 0)
        image = cv2.bilateralFilter(image, 2, 50, 50)
        image = cv2.applyColorMap(image, cv2.COLORMAP_BONE)
        image = cv2.resize(image, (image_size, image_size))
        x_test.append(image)
        y_test.append(labels.index(label))

# Normalize image data
x_train = np.array(x_train) / 255.0
x_test = np.array(x_test) / 255.0

# Shuffle the training data
x_train, y_train = shuffle(x_train, y_train)

# One-hot encode the labels
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

# Split the data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)

# Initialize ImageDataGenerator for data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    horizontal_flip=True
)
datagen.fit(x_train)

# === Squeeze-and-Excitation Block ===
def squeeze_excite_block(input_tensor, ratio=16):
    """ Squeeze-and-Excitation Block """
    filters = input_tensor.shape[-1]
    se = GlobalAveragePooling2D()(input_tensor)
    se = Dense(filters // ratio, activation='relu')(se)
    se = Dense(filters, activation='sigmoid')(se)
    se = Reshape((1, 1, filters))(se)
    return Multiply()([input_tensor, se])

# === Residual Block ===
def residual_block(x, filters, downsample=False):
    """ Residual Block with optional downsampling """
    shortcut = x
    if downsample:
        shortcut = Conv2D(filters, (1, 1), padding='same', strides=2)(shortcut)

    x = Conv2D(filters, (3, 3), activation='relu', padding='same', strides=(2 if downsample else 1))(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters, (3, 3), activation=None, padding='same')(x)  # No activation yet
    x = BatchNormalization()(x)
    x = Add()([x, shortcut])  # Residual Connection
    x = tf.keras.activations.relu(x)  # Apply activation after addition
    x = squeeze_excite_block(x)  # SE Block
    return x

# === Load Pretrained ResNet50 ===
resnet = ResNet101(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
#resnet.trainable = False  # Freeze ResNet50 layers

# === Keep the spatial dimensions for Residual Blocks ===
x = resnet.output 

# === Add Custom Residual + SE Blocks ===
x = residual_block(x, 128, downsample=True)
x = Dropout(0.3)(x)
x = residual_block(x, 256, downsample=True)
x = Dropout(0.4)(x)
x = residual_block(x, 512, downsample=True)
x = Dropout(0.5)(x)

# === Now Apply Global Pooling ===
x = GlobalAveragePooling2D()(x)

# === Fully Connected Layers ===
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)

# === Output Layer ===
outputs = Dense(4, activation='softmax')(x)

# === Define Final Model ===
model = Model(inputs=resnet.input, outputs=outputs)

# === Compile Model ===
adam = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

# === Model Summary ===
model.summary()

class_names = list(labels)

# Callbacks setup
EPOCHS = 50

Checkpoint = ModelCheckpoint(filepath='model-{epoch:02d}-{val_accuracy:.2f}-{val_loss:.2f}.h5', 
                              monitor='val_loss', verbose=1, save_best_only=True, mode='min')

ES = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10, mode='min', restore_best_weights=True, verbose=1)

RL = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=10, verbose=1, mode='min')

callbacks = [ES, RL, Checkpoint]

# Train the model and save logs for TensorBoard
history = model.fit(datagen.flow(x_train, y_train, batch_size=20), 
                    validation_data=(x_val, y_val),
                    epochs=EPOCHS, 
                    callbacks=callbacks)


# Predictions and evaluation
predicted_classes = np.argmax(model.predict(x_test), axis=1)  # Get predicted classes
confusionmatrix = confusion_matrix(np.argmax(y_test, axis=1), predicted_classes)  # Compare to true labels

# Save confusion matrix image
plt.figure(figsize=(10, 8))
sns.heatmap(confusionmatrix, cmap='Blues', annot=True, cbar=True, xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Save to file (e.g., as a PNG image)
confusion_matrix_file = 'confusion_matrix.png'
plt.savefig(confusion_matrix_file)
plt.close()

print(f'Confusion matrix saved as {confusion_matrix_file}')

# Print classification report
print(classification_report(np.argmax(y_test, axis=1), predicted_classes))

loss,acc = model.evaluate(x_test,y_test)
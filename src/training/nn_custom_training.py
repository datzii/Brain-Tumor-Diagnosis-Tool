import os
import numpy as np
import tensorflow as tf
import cv2
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, 
                                     GlobalAveragePooling2D, Input, Add, Activation, Multiply, Reshape)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, classification_report

from common.config import TRAINING_DIRECTORY, TESTING_DIRECTORY


print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)

# Training variables definition
labels = ['glioma', 'meningioma', 'notumor', 'pituitary']
image_size = 150 
num_classes = len(labels)

# Load and preprocess images
x_train, y_train, x_test, y_test = [], [], [], []

for label in labels:
    trainPath = os.path.join(TRAINING_DIRECTORY, label)
    for file in tqdm(os.listdir(trainPath)):
        image = cv2.imread(os.path.join(trainPath, file), 0)  # Load grayscale
        image = cv2.bilateralFilter(image, 5, 50, 50)
        image = cv2.applyColorMap(image, cv2.COLORMAP_BONE)  # Convert to pseudo-color
        image = cv2.resize(image, (image_size, image_size))
        x_train.append(image)
        y_train.append(labels.index(label))

    testPath = os.path.join(TESTING_DIRECTORY, label)
    for file in tqdm(os.listdir(testPath)):
        image = cv2.imread(os.path.join(testPath, file), 0)
        image = cv2.bilateralFilter(image, 5, 50, 50)
        image = cv2.applyColorMap(image, cv2.COLORMAP_BONE)
        image = cv2.resize(image, (image_size, image_size))
        x_test.append(image)
        y_test.append(labels.index(label))

# Normalize images
x_train, x_test = np.array(x_train) / 255.0, np.array(x_test) / 255.0
x_train, y_train = shuffle(x_train, y_train)

print(x_train.shape)
print(x_test.shape)

# One-hot encoding
y_train, y_test = tf.keras.utils.to_categorical(y_train, num_classes), tf.keras.utils.to_categorical(y_test, num_classes)

# Train-validation split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)

# Data augmentation
datagen = ImageDataGenerator(rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
datagen.fit(x_train)

# Custom CNN with Residual Connections
def squeeze_excite_block(input_tensor, ratio=16):
    """ Squeeze-and-Excitation Block """
    filters = input_tensor.shape[-1]
    se = GlobalAveragePooling2D()(input_tensor)
    se = Dense(filters // ratio, activation='relu')(se)
    se = Dense(filters, activation='sigmoid')(se)
    se = Reshape((1, 1, filters))(se)
    return Multiply()([input_tensor, se])

def residual_block(x, filters, downsample=False):
    """ Residual Block with optional downsampling """
    shortcut = x
    if downsample:
        shortcut = Conv2D(filters, (1, 1), padding='same', strides=2)(shortcut)

    x = Conv2D(filters, (3, 3), activation='relu', padding='same', strides=(2 if downsample else 1))(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters, (3, 3), activation=None, padding='same')(x) 
    x = BatchNormalization()(x)
    x = Add()([x, shortcut])  # Residual Connection
    x = tf.keras.activations.relu(x)  # Apply activation after addition
    x = squeeze_excite_block(x)  # SE Block
    return x

def custom_cnn(input_shape=(150, 150, 3), num_classes=4):
    inputs = Input(shape=input_shape)

    # Initial Conv Block
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    # Residual Blocks (with increasing filters)
    x = residual_block(x, 128, downsample=True)
    x = Dropout(0.3)(x)
    x = residual_block(x, 256, downsample=True)
    x = Dropout(0.4)(x)
    x = residual_block(x, 512, downsample=True)
    x = Dropout(0.5)(x)

    # Fully Connected Layers
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model


# Build model
model = custom_cnn()

# Compile model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

model.summary()

# Callbacks
logdir = os.path.join("logs_custom", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
callbacks = [
    ModelCheckpoint(filepath="best_custom_model.h5", monitor="val_loss", save_best_only=True, mode="min", verbose=1),
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, verbose=1, mode="min")
]

EPOCHS = 50

# Train model
history = model.fit(datagen.flow(x_train, y_train, batch_size=20),
                    validation_data=(x_val, y_val),
                    epochs=EPOCHS,
                    callbacks=callbacks)

# Evaluate model
predicted_classes = np.argmax(model.predict(x_test), axis=1)
confusionmatrix = confusion_matrix(np.argmax(y_test, axis=1), predicted_classes)

# Save Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(confusionmatrix, cmap="Blues", annot=True, xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.savefig("confusion_matrix.png")
plt.close()

print(classification_report(np.argmax(y_test, axis=1), predicted_classes))
loss, acc = model.evaluate(x_test, y_test)
print(f"Loss: {loss}, Accuracy: {acc}")

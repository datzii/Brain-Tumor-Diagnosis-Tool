from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard, LambdaCallback
from tensorflow.keras.layers import Input, Dropout, Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.applications import EfficientNetB6
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
    trainPath = os.path.join('../../../cleaned/Training', label)
    for file in tqdm(os.listdir(trainPath)):
        image = cv2.imread(os.path.join(trainPath, file), 0)  # Load images in gray scale
        image = cv2.bilateralFilter(image, 2, 50, 50)  # Remove image noise
        image = cv2.applyColorMap(image, cv2.COLORMAP_BONE)  # Apply pseudocolor
        image = cv2.resize(image, (image_size, image_size))  # Resize to 150x150
        x_train.append(image)
        y_train.append(labels.index(label))

    testPath = os.path.join('../../../cleaned/Testing', label)
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

# Define the ResNet50 model
net = EfficientNetB6(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))

model = net.output
model = GlobalAveragePooling2D()(model)
model = Dropout(0.4)(model)
model = Dense(4, activation="softmax")(model)
model = Model(inputs=net.input, outputs=model)

# Compile the model
adam = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Define TensorBoard callback
tensorboard_callback = TensorBoard(log_dir=logdir, histogram_freq=1)

# Create a writer variable for writing into the log folder (for confusion matrix)
file_writer_cm = tf.summary.create_file_writer(logdir)

class_names = list(labels)
# Function to convert confusion matrix figure to image
def plot_to_image(figure):    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)

    digit = tf.image.decode_png(buf.getvalue(), channels=4)
    digit = tf.expand_dims(digit, 0)

    return digit

# Function to plot the confusion matrix
def plot_confusion_matrix(cm, class_names): 
    figure = plt.figure(figsize=(8, 8)) 
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Accent) 
    plt.title("Confusion matrix") 
    plt.colorbar() 
    tick_marks = np.arange(len(class_names)) 
    plt.xticks(tick_marks, class_names, rotation=45) 
    plt.yticks(tick_marks, class_names)

    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)  
    threshold = cm.max() / 2. 

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):   
        color = "white" if cm[i, j] > threshold else "black"   
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)  
    
    plt.tight_layout() 
    plt.ylabel('True label') 
    plt.xlabel('Predicted label') 

    return figure

# Following function will make predictions from the model and log the confusion matrix as an image
def log_confusion_matrix(epoch, logs):
    predictions = np.argmax(model.predict(x_test), axis=1)
    cm = confusion_matrix(np.argmax(y_test, axis=1), predictions)
    figure = plot_confusion_matrix(cm, class_names=class_names)
    cm_image = plot_to_image(figure)
    
    with file_writer_cm.as_default():
        tf.summary.image("Confusion Matrix", cm_image, step=epoch)

# Callbacks setup
BATCH_SIZE = 64
EPOCHS = 50

Checkpoint = ModelCheckpoint(filepath='model-{epoch:02d}-{val_accuracy:.2f}-{val_loss:.2f}.h5', 
                              monitor='val_loss', verbose=1, save_best_only=True, mode='min')

ES = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5, mode='min', restore_best_weights=True, verbose=1)

RL = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=5, verbose=1, mode='min')

callbacks = [ES, RL, tensorboard_callback, Checkpoint, LambdaCallback(on_epoch_end=log_confusion_matrix)]

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
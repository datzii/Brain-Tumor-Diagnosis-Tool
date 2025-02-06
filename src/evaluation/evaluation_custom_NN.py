from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import cv2
from tqdm import tqdm
import tensorflow as tf
import os

# Specify the path to the saved model
model_path = '../../models/NN_custom/best_custom_model.h5'

# Load the trained model
model = load_model(model_path)

# Print model summary to check if it's loaded properly
model.summary()

# Load the labels and initialize other variables
labels = ['glioma', 'meningioma', 'notumor', 'pituitary']
image_size = 150

# Initialize lists for images and labels
x_train = []  # Training images
y_train = []  # Training labels
x_test = []   # Testing images
y_test = []   # Testing labels

# Your image loading and processing code
for label in labels:

    testPath = os.path.join('../../../cleaned/Testing', label)
    for file in tqdm(os.listdir(testPath)):
        image = cv2.imread(os.path.join(testPath, file), 0)
        image = cv2.bilateralFilter(image, 2, 50, 50)
        image = cv2.applyColorMap(image, cv2.COLORMAP_BONE)
        image = cv2.resize(image, (image_size, image_size))
        x_test.append(image)
        y_test.append(labels.index(label))

# Normalize image data
x_test = np.array(x_test) / 255.0

# If y_test is not one-hot encoded, convert it to one-hot encoding
y_test = tf.keras.utils.to_categorical(y_test, num_classes=4)

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
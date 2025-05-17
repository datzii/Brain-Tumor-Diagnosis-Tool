import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
from joblib import dump
import pandas as pd  # for showing the table of results
from common.config import TRAINING_DIRECTORY, TESTING_DIRECTORY

# Load the labels and initialize variables
labels = ['glioma', 'meningioma', 'notumor', 'pituitary']
image_size = 150

# Initialize lists for images and labels
x_train, y_train, x_test, y_test = [], [], [], []

# Load and process images
for label in labels:
    trainPath = os.path.join(TRAINING_DIRECTORY, label)
    for file in tqdm(os.listdir(trainPath)):
        image = cv2.imread(os.path.join(trainPath, file), 0)  # Load grayscale image
        image = cv2.bilateralFilter(image, 2, 50, 50)  # Noise reduction
        image = cv2.applyColorMap(image, cv2.COLORMAP_BONE)  # Apply pseudocolor
        image = cv2.resize(image, (image_size, image_size)).flatten()  # Flatten image
        x_train.append(image)
        y_train.append(labels.index(label))  # Convert label to numerical value

    testPath = os.path.join(TESTING_DIRECTORY, label)
    for file in tqdm(os.listdir(testPath)):
        image = cv2.imread(os.path.join(testPath, file), 0)
        image = cv2.bilateralFilter(image, 2, 50, 50)
        image = cv2.applyColorMap(image, cv2.COLORMAP_BONE)
        image = cv2.resize(image, (image_size, image_size)).flatten()
        x_test.append(image)
        y_test.append(labels.index(label))

# Convert to NumPy arrays and normalize
x_train = np.array(x_train) / 255.0
x_test = np.array(x_test) / 255.0
y_train = np.array(y_train)
y_test = np.array(y_test)

# Train the SVM model using GridSearchCV
param_grid = {
    'kernel': ['linear', 'rbf', 'poly'],
    'C': [0.1, 1],
    'gamma': [0.001, 0.01],
    'degree': [2, 3]  # Only used for 'poly'
}

svc = GridSearchCV(svm.SVC(), param_grid, cv=3, verbose=2)
svc.fit(x_train, y_train)

# Display GridSearchCV results in a table
results_df = pd.DataFrame(svc.cv_results_)
print("\nGridSearchCV Results:")
print(results_df[['param_kernel', 'param_C', 'param_gamma', 'param_degree', 'mean_test_score', 'std_test_score']])

# Test the model with the best parameters
print("Testing the SVM model...")
y_pred = svc.predict(x_test)

# Confusion matrix for the best model
confusionmatrix = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(confusionmatrix, cmap='Blues', annot=True, cbar=True, xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix (Best Model)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Save confusion matrix
confusion_matrix_path = 'best_confusion_matrix.png'
plt.savefig(confusion_matrix_path)
plt.close()
print(f"Confusion matrix saved to {confusion_matrix_path}")

# Print evaluation metrics
print("\nConfusion Matrix:")
print(confusionmatrix)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=labels))

# Save the best trained model
best_model_path = 'best_svm_model.joblib'
dump(svc.best_estimator_, best_model_path)
print(f"Best model saved to {best_model_path}")
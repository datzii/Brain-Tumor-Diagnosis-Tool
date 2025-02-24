import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV  # CPU GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump

# Labels and Image Size
labels = ['glioma', 'meningioma', 'notumor', 'pituitary']
image_size = 150

# Lists to store images and labels
x_train, y_train, x_test, y_test = [], [], [], []

# Load data
for label in labels:
    trainPath = os.path.join('../../../cleaned/Training', label)
    for file in tqdm(os.listdir(trainPath)):
        image = cv2.imread(os.path.join(trainPath, file), 0)  # Grayscale
        image = cv2.bilateralFilter(image, 2, 50, 50)  # Noise reduction
        image = cv2.applyColorMap(image, cv2.COLORMAP_BONE)  # Apply pseudocolor
        image = cv2.resize(image, (image_size, image_size)).flatten()  # Flatten
        x_train.append(image)
        y_train.append(labels.index(label))  # Convert label to index

    testPath = os.path.join('../../../cleaned/Testing', label)
    for file in tqdm(os.listdir(testPath)):
        image = cv2.imread(os.path.join(testPath, file), 0)
        image = cv2.bilateralFilter(image, 2, 50, 50)
        image = cv2.applyColorMap(image, cv2.COLORMAP_BONE)
        image = cv2.resize(image, (image_size, image_size)).flatten()
        x_test.append(image)
        y_test.append(labels.index(label))

# Normalize the data
x_train = np.array(x_train) / 255.0
x_test = np.array(x_test) / 255.0
y_train = np.array(y_train)
y_test = np.array(y_test)

# Initialize Random Forest
rf_cpu = RandomForestClassifier(max_depth=30, random_state=42, n_estimators=200)

# Hyperparameter Grid for Tuning
#Best Hyperparameters: {'max_depth': 30, 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 200}
param_grid = {
    'n_estimators': [50, 100, 200],  # Number of trees
    'max_depth': [10, 20, 30],     # Tree depth
    'min_samples_split': [2, 5, 10], # Minimum samples to split
    'min_samples_leaf': [1, 2, 4],   # Minimum samples per leaf
}

# Fit the data to the different models
grid_search = GridSearchCV(estimator=rf_cpu, param_grid=param_grid, cv=3, n_jobs=1, verbose=2)
rf_cpu.fit(x_train, y_train)

# Get Best Model
#best_rf_cpu = grid_search.best_estimator_

# Prediction on Test Data
y_pred_cpu = rf_cpu.predict(x_test)

# Evaluate Model with Test Data
conf_matrix = confusion_matrix(y_test, y_pred_cpu)
#print("\nBest Hyperparameters:", grid_search.best_params_)
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", classification_report(y_test, y_pred_cpu))

# Plot Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, cmap='Blues', annot=True, cbar=True, xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('rf_cpu_confusion_matrix.png')
plt.close()

# Save the Best Model
dump(rf_cpu, 'rf_model.joblib')
print("Best CPU-accelerated model saved as best_rf_cpu_model.joblib")

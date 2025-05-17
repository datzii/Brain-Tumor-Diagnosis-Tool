import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd  # For tabular summary
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump
from common.config import TRAINING_DIRECTORY, TESTING_DIRECTORY

# Labels and Image Size
labels = ['glioma', 'meningioma', 'notumor', 'pituitary']
image_size = 150

# Lists to store images and labels
x_train, y_train, x_test, y_test = [], [], [], []

# Load data
for label in labels:
    trainPath = os.path.join(TRAINING_DIRECTORY, label)
    for file in tqdm(os.listdir(trainPath)):
        image = cv2.imread(os.path.join(trainPath, file), 0)
        image = cv2.bilateralFilter(image, 2, 50, 50)
        image = cv2.applyColorMap(image, cv2.COLORMAP_BONE)
        image = cv2.resize(image, (image_size, image_size)).flatten()
        x_train.append(image)
        y_train.append(labels.index(label))

    testPath = os.path.join(TESTING_DIRECTORY, label)
    for file in tqdm(os.listdir(testPath)):
        image = cv2.imread(os.path.join(testPath, file), 0)
        image = cv2.bilateralFilter(image, 2, 50, 50)
        image = cv2.applyColorMap(image, cv2.COLORMAP_BONE)
        image = cv2.resize(image, (image_size, image_size)).flatten()
        x_test.append(image)
        y_test.append(labels.index(label))

# Normalize data
x_train = np.array(x_train) / 255.0
x_test = np.array(x_test) / 255.0
y_train = np.array(y_train)
y_test = np.array(y_test)

# Initialize Random Forest
rf_cpu = RandomForestClassifier(random_state=42)

# Hyperparameter Grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

# Grid Search with Cross-Validation
grid_search = GridSearchCV(estimator=rf_cpu, param_grid=param_grid, cv=3, n_jobs=1, verbose=2)
grid_search.fit(x_train, y_train)

# Get best model and parameters
best_rf_cpu = grid_search.best_estimator_
print("\n✅ Best Hyperparameters:", grid_search.best_params_)

# Show GridSearchCV results as a table
results_df = pd.DataFrame(grid_search.cv_results_)
results_table = results_df[[
    'param_n_estimators', 'param_max_depth', 'param_min_samples_split',
    'param_min_samples_leaf', 'mean_test_score', 'std_test_score'
]].sort_values(by='mean_test_score', ascending=False)

print("\n📊 GridSearchCV Summary:")
print(results_table.to_string(index=False))

# Predict with best model
y_pred_cpu = best_rf_cpu.predict(x_test)

# Evaluation
conf_matrix = confusion_matrix(y_test, y_pred_cpu)
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

# Save the best model
dump(best_rf_cpu, 'best_rf_cpu_model.joblib')
print("💾 Best model saved as best_rf_cpu_model.joblib")

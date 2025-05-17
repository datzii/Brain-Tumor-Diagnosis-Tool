import joblib
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn import svm
from sklearn.calibration import label_binarize
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score
)
import seaborn as sns   
from tqdm import tqdm
from joblib import dump, load
from common.config import TESTING_DIRECTORY

# Load the saved model

model = load("best_rf_cpu_model.joblib")  # Replace with your file name


# Load the labels and initialize variables
labels = ['glioma', 'meningioma', 'notumor', 'pituitary']
image_size = 150

# Initialize lists for images and labels
x_train, y_train, x_test, y_test = [], [], [], []

# Load and process images
for label in labels:

    testPath = os.path.join(TESTING_DIRECTORY, label)
    for file in tqdm(os.listdir(testPath)):
        image = cv2.imread(os.path.join(testPath, file), 0)
        image = cv2.bilateralFilter(image, 2, 50, 50)
        image = cv2.applyColorMap(image, cv2.COLORMAP_BONE)
        image = cv2.resize(image, (image_size, image_size)).flatten()
        x_test.append(image)
        y_test.append(labels.index(label))

# Convert to NumPy arrays and normalize
x_test = np.array(x_test) / 255.0
y_train = np.array(y_train)
y_test = np.array(y_test)


# Example: Assuming you have test data X_test and y_test
y_pred = model.predict(x_test)  # Get predictions
accuracy = accuracy_score(y_test, y_pred)  # Compute accuracy

print(f"Model Accuracy: {accuracy:.2%}")
print(classification_report(y_test, y_pred))


# Confusion matrix
confusionmatrix = confusion_matrix(y_test, y_pred)

# Basic metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')  # 'macro' averages over classes
recall = recall_score(y_test, y_pred, average='macro')        # Sensitivity
f1 = f1_score(y_test, y_pred, average='macro')

# Specificity calculation (per class)
# Specificity = TN / (TN + FP) for each class
specificities = []
for i in range(len(confusionmatrix)):
    TP = confusionmatrix[i, i]
    FN = np.sum(confusionmatrix[i, :]) - TP
    FP = np.sum(confusionmatrix[:, i]) - TP
    TN = np.sum(confusionmatrix) - (TP + FP + FN)
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    specificities.append(specificity)

specificity = np.mean(specificities)

# AUC-ROC (for multiclass, using One-vs-Rest strategy)
try:
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3])
    y_pred_scores = model.decision_function(x_test)
    auc_roc = roc_auc_score(y_test_bin, y_pred_scores, average='macro', multi_class='ovr')
except:
    auc_roc = "N/A (model must support decision_function or predict_proba)"

# Print all metrics
print("\n🔹 Evaluation Metrics:")
print(f"Accuracy:     {accuracy:.2%}")
print(f"Precision:    {precision:.2%}")
print(f"Recall:       {recall:.2%} (Sensitivity)")
print(f"Specificity:  {specificity:.2%}")
print(f"F1 Score:     {f1:.2%}")
print(f"AUC-ROC:      {auc_roc}")

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(confusionmatrix, cmap='Blues', annot=True, cbar=True, xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Save confusion matrix
plt.savefig('confusion_matrix.png')
plt.close()

# Print evaluation metrics
print("\nConfusion Matrix:")
print(confusionmatrix)
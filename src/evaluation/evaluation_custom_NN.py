from tensorflow.keras.models import load_model
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import cv2
from tqdm import tqdm
import tensorflow as tf
import os
from common.config import TESTING_DIRECTORY

# Load the trained model
model_path = '../models/NN_custom/best_custom_model.h5'
model = load_model(model_path)

# Labels and image size
labels = ['glioma', 'meningioma', 'notumor', 'pituitary']
image_size = 150

# Load and preprocess test data
x_test = []
y_test = []

for label in labels:
    test_path = os.path.join(TESTING_DIRECTORY, label)
    for file in tqdm(os.listdir(test_path)):
        image = cv2.imread(os.path.join(test_path, file), 0)
        image = cv2.bilateralFilter(image, 2, 50, 50)
        image = cv2.applyColorMap(image, cv2.COLORMAP_BONE)
        image = cv2.resize(image, (image_size, image_size))
        x_test.append(image)
        y_test.append(labels.index(label))

# Normalize and convert data
x_test = np.array(x_test) / 255.0
y_test = np.array(y_test)
y_test_bin = tf.keras.utils.to_categorical(y_test, num_classes=4)

# Predict
y_pred_prob = model.predict(x_test)
y_pred = np.argmax(y_pred_prob, axis=1)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Specificity calculation
specificities = []
for i in range(len(conf_matrix)):
    TP = conf_matrix[i, i]
    FN = np.sum(conf_matrix[i, :]) - TP
    FP = np.sum(conf_matrix[:, i]) - TP
    TN = np.sum(conf_matrix) - (TP + FP + FN)
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    specificities.append(specificity)

specificity = np.mean(specificities)

# AUC-ROC
try:
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3])
    auc_roc = roc_auc_score(y_test_bin, y_pred_prob, average='macro', multi_class='ovr')
except:
    auc_roc = "N/A (AUC requires probability outputs or decision scores)"

# Print all metrics
print("\n🔹 Evaluation Metrics:")
print(f"Accuracy:     {accuracy:.2%}")
print(f"Precision:    {precision:.2%}")
print(f"Recall:       {recall:.2%} (Sensitivity)")
print(f"Specificity:  {specificity:.2%}")
print(f"F1 Score:     {f1:.2%}")
print(f"AUC-ROC:      {auc_roc}")

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=labels))

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, cmap='Blues', annot=True, cbar=True, xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('confusion_matrix.png')
plt.close()

print("\nConfusion matrix saved as confusion_matrix.png")

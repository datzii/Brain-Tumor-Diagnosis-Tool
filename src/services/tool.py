from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import numpy as np
import cv2

from services.Preprocessing import crop_img



def classify_brain_tumor_from_MRI(file_path: str, model) -> str:    
    ''' Classifies the type of tumor of a brain '''

    # Define class labels
    labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

    image_size = 150  # Model's expected input size

    image = cv2.imread(file_path, 0)  # Read in grayscale
    image = crop_img(image)
    cv2.imwrite('/mnt/c/Users/Usuario/Desktop/MASTER/TFM/cleaned/Testing'+'/'+'prova.jpg', image)
    image = cv2.bilateralFilter(image, 2, 50, 50)  # Apply bilateral filtering
    image = cv2.applyColorMap(image, cv2.COLORMAP_BONE)  # Apply color mapping
    image = cv2.resize(image, (image_size, image_size))  # Resize to match model input
    image = image / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Predict the class
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)  # Get class index
    predicted_label = labels[predicted_class]  # Get class name

    # Display result
    return f"Predicted Class: {predicted_label} (Confidence: {prediction[0][predicted_class]:.2%})"
import cv2
import json
import numpy as np
from services.Preprocessing import crop_img
from collections import Counter



def start_brain_tumor_classifier(data, model):
    file_path = data.get('file_path')
    try:
        response = classify_brain_tumor_from_MRI(file_path, model)
        return {
            "response": response
        }
    
    except Exception as err:
        print("Error executing NN ", err)


def classify_brain_tumor_from_MRI(file_path: str, models) -> str:    
    ''' Classifies the type of tumor of a brain '''

    image = load_image(file_path)

    predicted_labels = []
    confidence_values = []

    # Predict the class
    for model in models:
        pred, conf = make_prediction(image, model)
        print(f"Predicted Class: {pred} (Confidence: {conf})")
        predicted_labels.append(pred)
        confidence_values.append(conf)    

    # Count occurrences of each label
    label_counts = Counter(predicted_labels)

    # Find the mode (most common label), handling the tie case by choosing the first occurrence
    max_count = max(label_counts.values())  # Highest frequency
    mode_pred = next(label for label in predicted_labels if label_counts[label] == max_count)

    # Compute the mean confidence of predictions that match the mode
    mean_conf = np.mean([conf for pred, conf in zip(predicted_labels, confidence_values) if pred == mode_pred])

    # Display result
    result = f"Predicted Class: {mode_pred} (Confidence: {mean_conf:.2%})"
    print(f'-- Final Result {result}')
    return result


def make_prediction(image, model):    
    # Define class labels
    labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

    # Make prediction
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)
    predicted_label = labels[predicted_class]

    # Return prediction and confidence
    return predicted_label, prediction[0][predicted_class]


def load_image(file_path: str):
    image_size = 150  # Model's expected input size

    image = cv2.imread(file_path)  
    print('cropping')
    image = crop_img(image)
    image = cv2.bilateralFilter(image, 2, 50, 50)  
    image = cv2.applyColorMap(image, cv2.COLORMAP_BONE) 
    image = cv2.resize(image, (image_size, image_size))   
    image = image / 255.0 
    image = np.expand_dims(image, axis=0) 

    return image
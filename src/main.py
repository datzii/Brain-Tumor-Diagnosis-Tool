
from tensorflow.keras.models import load_model
from services.tool import classify_brain_tumor_from_MRI

if __name__ == "__main__":

    model_path = '../models/resNet/model-14-0.99-0.05.h5'
    model = load_model(model_path)
    print('model loaded')

    print(classify_brain_tumor_from_MRI('/mnt/c/Users/Usuario/Desktop/MASTER/TFM/cleaned/Testing/images/Te-no_0012.jpg'))

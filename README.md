# Brain Tumor Diagnosis Tool
Brain Tumor Diagnosis Tool based on trained Deep and Machine Learning models over a dataset of MRI images.

## 🚀 Getting Started

### 1. Create a Virtual Environment and Install Dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Crop and save images

```bash
cd src/services
```
Edit the `Preprocessing.py` file with the appropriate values in the `main` section:
- `from_path`: The path where the original dataset is
- `to_path`: The path where the cropped images want to be saved

Start the preprocessing:
```bash
cd python Preprocessing.py
```


### 3. Modify configuration variables
```bash
cd src/common
```

Edit the `config.py` file with the appropriate values:
- `TRAINING_DIRECTORY`: The directory where the cropped training images of the dataset are
- `TESTING_DIRECTORY`: The directory where the cropped testing images of the dataset are

### 4. Train the deep and machine learning models

To train the different models with the MRI images training dataset, the scripts are found at `src/training`.

To train the model defined in the file `model_to_train.py` in the folder `src/training`, for example:

```bash
cd src
python -m training.model_to_train
```

The output will be the trained model saved and the confusion matrix of the evaluation of the model.

### 5. Evaluate a deep and machine learning model

To evaluate the different models with the MRI images testing dataset, the scripts are found at `src/evaluation`.

To evaluate the model defined in the file `model_to_evaluate.py` in the folder `src/evaluation`, for example:

Edit the `model_to_evaluate.py` file with the appropriate values:
- `model_path`: The path were the saved model during training is located

```bash
cd src
python -m evaluation.model_to_evaluate
```

The output will be the metrics report and the confusion matrix of the model.


### 6. Start the Brain Tumor Diagnosis Tool micro-service

```bash
cd src
```
Edit the `main.py` file with the appropriate values:
- `paths`: The paths of the saved models that will be used in the classification of the Brain Tumor Diagnosis Tool

```bash
python main.py
```


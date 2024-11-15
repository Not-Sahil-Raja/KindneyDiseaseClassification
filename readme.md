---
title: Kidney Disease Classification
emoji: 🩺
colorFrom: blue
colorTo: purple
sdk: docker
pinned: true
license: apache-2.0
short_description: A deep learning-based project to classify kidney diseases from medical images, utilizing a sequential model and FastAPI for serving predictions.
tags:
  - deep learning
  - medical imaging
  - kidney disease
  - classification
  - FastAPI
  - TensorFlow
  - healthcare AI
  - Docker
long_description: |
  This project leverages deep learning techniques to classify kidney disease from medical images. It includes a binary classifier to identify kidney CT scans and a multi-class classifier to differentiate types of kidney diseases, such as cysts, stones, tumors, or normal conditions. The backend is built with FastAPI and containerized using Docker, while the frontend, developed with Next.js, provides a user-friendly interface for uploading images and viewing results.
---

# Kidney Disease Classification

This project aims to classify kidney disease using a deep learning model based on the VGG16 architecture. The project involves data ingestion, preprocessing, model training, and evaluation using TensorFlow and MLflow for experiment tracking.

## Table of Contents

- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Data Ingestion](#data-ingestion)
- [Model Preparation](#model-preparation)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Logging and Tracking](#logging-and-tracking)
- [License](#license)

## Project Structure

    project_root/
    │
    ├── artifacts/
    │   └── (directories created by the script)
    │
    ├── config/
    │   └── config.yaml
    │
    ├── src/
    │   └── Kidney_Disease_Classification/
    │       ├── __init__.py
    │       ├── components/
    │       │   ├── data_ingestion.py
    │       │   ├── model_evaluation.py
    │       │   ├── model_training.py
    │       │   └── prepare_base_model.py
    │       ├── config/
    │       │   └── configurations.py
    │       ├── entity/
    │       │   └── config_entity.py
    │       ├── pipeline/
    │       │   ├── stage_01_data_ingestion.py
    │       │   ├── stage_02_prepare_base_model.py
    │       │   ├── stage_03_model_training.py
    │       │   └── stage_04_model_evaluation.py
    │       └── utils/
    │           └── common.py
    │
    ├── params.yaml
    ├── requirements.txt
    └── README.md

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Not-Sahil-Raja/KindneyDiseaseClassification.git
   ```
2. **Create a virtual environment Using Conda & Activate that (Make Sure Conda is installed):**
   ```bash
   conda create -n kidney_CNN python=3.8.20 -y
   conda activate  kidney_CNN
   ```
3. **Install the necessary requirements:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the Main.py Only For the Deep Learning Pipeline [Optional]:**
   ```bash
   python main.py
   ```
5. **Run the App.py For the application:**
   ```bash
   python App.py
   ```

## Configuration

The configuration files are located in the `config` directory.
The `config.yaml` file contains the paths and parameters for data ingestion, model preparation, training, and evaluation.
The `params.yaml` file contains the hyperparameters for model training.

## Data Ingestion

The data ingestion pipeline downloads and extracts the dataset. The dataset is already uploaded into my google drive.

1.  **Download Data**: Fetch the dataset from the specified URL.
2.  **Extract Data**: Unzip the downloaded dataset to the specified directory.
3.  **Save the Data Path**: Save the path to the extracted dataset.

## Model Preparation

The model preparation pipeline loads the VGG16 base model and updates it for the specific task.

1.  **Load Configuration**: The configuration dictionary contains parameters such as image size, batch size, number of classes, weights, learning rate, and paths to save the base and updated models.

2.  **Load Pre-trained Model**: The VGG16 model is loaded with pre-trained weights from ImageNet, excluding the top classification layer (`include_top=False`).
3.  **Add Custom Layers**: Custom layers are added on top of the pre-trained model, including a Flatten layer, a Dropout layer for regularization, and a Dense layer with the number of units equal to the number of classes.
4.  **Compile the Model**: The model is compiled with the Adam optimizer, categorical cross-entropy loss, and accuracy as the evaluation metric.
5.  **Save the Models**: The base model and the updated model are saved to the specified paths.

## Model Training

The model training pipeline trains the model using the prepared data.

1.  **Load the Base Model**: Load the pre-trained and updated VGG16 model.

2.  **Prepare Data Generators**: Create data generators for training and validation data.
3.  **Compile the Model**: Compile the model with an appropriate optimizer, loss function, and metrics.
4.  **Train the Model**: Train the model using the training data and validate it using the validation data.
5.  **Save the Trained Model**: Save the trained model to a specified path.

## Model Evaluation

1.  **Load the Trained Model**: Load the trained model from the specified path.

2.  **Prepare Data Generators**: Create data generators for validation data.
3.  **Evaluate the Model**: Evaluate the model using the validation data.
4.  **Save the Evaluation Metrics**: Save the evaluation metrics to a JSON file.

## Logging and Tracking

MLflow is used for logging and tracking experiments. Ensure that the MLflow tracking URI and credentials are set correctly (currently not using MLflow because some of the error coming while logging into MLflow). For tracking the files and skipping the unnecessary files currently using DVC.

For setting Dagshub Run this commands in your windows bash or Linux :

```bash
export MLFLOW_TRACKING_URI=<your clone link>
export MLFLOW_TRACKING_USERNAME=<your usename>
export MLFLOW_TRACKING_PASSWORD=<your password>
```

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

## Resources

- [VisoAi VGG Blog](https://viso.ai/deep-learning/vgg-very-deep-convolutional-networks/#:~:text=The%20VGG16%20model%20achieves%20almost,models%20submitted%20to%20ILSVRC%2D2014.)
- [GFG VGG-16](https://www.geeksforgeeks.org/vgg-16-cnn-model/)
- [Kaggle Kidney Disease Dataset](https://www.kaggle.com/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone)
- [VGG16 Model Visualitation](https://youtu.be/RNnKtNrsrmg?si=7W3P2XSfgR5KWbvg)

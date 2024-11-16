<h1 align= "center" id="top"> Kidney Disease Ai </h1>

<h3 align="center">This project classifies kidney disease using Deep Learning Model.</h3>

<!-- <img
src="https://img.shields.io/github/license/Not-Sahil-Raja/KindneyDiseaseClassification"
alt="Logo"
style="boreder-radius: 8px"
/> -->

<div align="center" style="display: flex; justify-content: center; gap: 10px; flex-wrap: wrap;">
   <a href="https://kidneyhealthai.vercel.app/">
      <img src="https://img.shields.io/badge/Deployed-Frontend-blue?style=for-the-badge&logo=vercel" alt="Deployed Frontend">
   </a>
   <a href="https://huggingface.co/spaces/SahilRaja/Kidney_Disease_Classifier">
      <img src="https://img.shields.io/badge/Deployed-Backend-orange?style=for-the-badge&logo=huggingface" alt="Deployed Backend">
   </a>
</div>

<div align="center" style="margin-top: 10px;">
  <a href="https://github.com/Not-Sahil-Raja/KindneyDiseaseClassificationFrontend">
    <img src="https://img.shields.io/badge/Frontend-Repository-olive?style=for-the-badge&logo=github" alt="Frontend Repository">
  </a>
</div>

## What is the Aim of this Project?

**The aim of this project is to classify kidney disease using a binary classifier and a TensorFlow sequential model.**

## Challenges Faced 📉

- First i tried to use the VGG16 model but the accuracy was not upto the mark.That's why i used the Sequential model.

- At first the used single model to classify images, although it was able to classify the CT-Scan images but when i tried to classify the normal images it was not able to classify them properly.

- Deploying the model was also a challenge, as i was not able to deploy the model on the Render because there wasn't enough memory to use the model.

- The model size was also a challenge, as the model size was too large to deploy on the huggingface without lfs.

## Features 🌟

- **The dataset includes images of normal kidneys as well as those with cysts, tumors, and stones. Additionally, the model is trained with random non-kidney images to enhance its ability to distinguish between kidney and non-kidney images.**

- **After using much higher epoch we managed to get almost 99.5% accuracy on the validation set.**

- **A FastAPI endpoint is provided, enabling seamless predictions from anywhere with ease.**

## Installation 🛠️

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Not-Sahil-Raja/KindneyDiseaseClassification.git
   ```
2. **Create a virtual environment Using Conda & Activate that:**

   ```bash
   conda create -n kidney_CNN python=3.10.12 -y
   conda activate  kidney_CNN
   ```

   <span style="color: gold; opacity: 0.7;"> \*\* make sure you have conda installed in your system.</span>

3. **Install the necessary requirements:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Create the model directory:**

   ```bash
   mkdir model
   ```

   <span style="color: gold; opacity: 0.7;"> \*\* make sure you are at the root directory of the project.</span>

5. **Download the model:**

   ```bash
   python download_model.py
   ```

6. **Run the application:**
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 7860
   ```
7. **Run the App.py For the application:**
   ```bash
   python App.py
   ```
   <span style="color: skyblue ; opacity: 1;"> \*\* this might throw some error currently working on the fix !!</span>

## Configuration

The configuration files are located in the `config` directory.
The `config.yaml` file contains the paths and parameters for data ingestion, model preparation, training, and evaluation.
The `params.yaml` file contains the hyperparameters for model training.

## Data Ingestion

The data ingestion pipeline downloads and extracts the dataset. The dataset is already uploaded into my google drive.

1.  **Download Data**: Fetch the dataset from the specified URL.
2.  **Extract Data**: Unzip the downloaded dataset to the specified directory.
3.  **Save the Data Path**: Save the path to the extracted dataset.

Certainly! Here is a detailed explanation of data preprocessing and model preparation for your README, including corresponding code snippets:

## Data Preprocessing

Data preprocessing is a crucial step to ensure the model receives data in a suitable format for training. This involves data loading, normalization, and splitting the dataset into training and validation sets.

- Loading Data

  Load the dataset from the specified path and display the first few rows to understand the data structure.

- Data Normalization

  Normalization scales the input data to a range of [0, 1], which helps in faster convergence during training.

- Data Splitting

  Split the dataset into training and validation sets to evaluate the model's performance on unseen data.

## Model Preparation

We use a Sequential model architecture to build a deep learning model for kidney disease classification. The model consists of multiple layers, including input, hidden, and output layers.

- Building the Model

  Build the model using the Sequential API in TensorFlow. Add dense layers with ReLU activation and dropout for regularization. The output layer uses the softmax activation function for multi-class classification.

- Compiling the Model

  Compile the model with an appropriate loss function, optimizer, and evaluation metric. The loss function is categorical cross-entropy for multi-class classification.

- Training the Model

  Train the model using the training data and validate it using the validation data. The training process involves multiple epochs, where the model learns to minimize the loss function.

  ```python
  # Train the model
  history = model.fit(
     X_train, y_train,
     epochs=50,
     batch_size=32,
     validation_data=(X_val, y_val)
  )

  # Save the model
  model.save('kidney_disease_model.h5')
  ```

- **Model Summary**: The model consists of several dense layers with ReLU activation and dropout for regularization.

  Model: Sequential

  | Layer (type)                               | Output Shape          | Param #   |
  | ------------------------------------------ | --------------------- | --------- |
  | conv2d (Conv2D)                            | (None, 224, 224, 64)  | 1,792     |
  | batch_normalization (BatchNormalization)   | (None, 224, 224, 64)  | 256       |
  | conv2d_1 (Conv2D)                          | (None, 224, 224, 64)  | 36,928    |
  | batch_normalization_1 (BatchNormalization) | (None, 224, 224, 64)  | 256       |
  | max_pooling2d (MaxPooling2D)               | (None, 112, 112, 64)  | 0         |
  | dropout (Dropout)                          | (None, 112, 112, 64)  | 0         |
  | conv2d_2 (Conv2D)                          | (None, 112, 112, 128) | 73,856    |
  | batch_normalization_2 (BatchNormalization) | (None, 112, 112, 128) | 512       |
  | conv2d_3 (Conv2D)                          | (None, 112, 112, 128) | 147,584   |
  | batch_normalization_3 (BatchNormalization) | (None, 112, 112, 128) | 512       |
  | max_pooling2d_1 (MaxPooling2D)             | (None, 56, 56, 128)   | 0         |
  | dropout_1 (Dropout)                        | (None, 56, 56, 128)   | 0         |
  | conv2d_4 (Conv2D)                          | (None, 56, 56, 256)   | 295,168   |
  | batch_normalization_4 (BatchNormalization) | (None, 56, 56, 256)   | 1,024     |
  | conv2d_5 (Conv2D)                          | (None, 56, 56, 256)   | 590,080   |
  | batch_normalization_5 (BatchNormalization) | (None, 56, 56, 256)   | 1,024     |
  | conv2d_6 (Conv2D)                          | (None, 56, 56, 256)   | 590,080   |
  | batch_normalization_6 (BatchNormalization) | (None, 56, 56, 256)   | 1,024     |
  | max_pooling2d_2 (MaxPooling2D)             | (None, 28, 28, 256)   | 0         |
  | dropout_2 (Dropout)                        | (None, 28, 28, 256)   | 0         |
  | conv2d_7 (Conv2D)                          | (None, 28, 28, 512)   | 1,180,160 |
  | batch_normalization_7 (BatchNormalization) | (None, 28, 28, 512)   | 2,048     |
  | conv2d_8 (Conv2D)                          | (None, 28, 28, 512)   | 2,359,808 |
  | batch_normalization_8 (BatchNormalization) | (None, 28, 28, 512)   | 2,048     |
  | conv2d_9 (Conv2D)                          | (None, 28, 28, 512)   | 2,359,808 |
  | batch_normalization_9 (BatchNormalization) | (None, 28, 28, 512)   | 2,048     |
  | max_pooling2d_3 (MaxPooling2D)             | (None, 14, 14, 512)   | 0         |
  | dropout_3 (Dropout)                        | (None, 14, 14, 512)   |

  Total params: 16,342,212 (62.34 MB)

  Trainable params: 16,333,508 (62.31 MB)

  Non-trainable params: 8,704 (34.00 KB)

  ***

## Model Training

The model training pipeline trains the model using the prepared data.
There are two types of model we are using in this project, one is the Sequential model for multi-class classification and the other is the binary classifier model.

- **Training the Model**: The model is trained using the training data and validated using the validation data. The training process involves multiple epochs, where the model learns to minimize the loss function.
- **Model Checkpoint**: Model checkpoint saves the best model based on the validation loss. This ensures that the best model is saved and used for evaluation.

- **Early Stopping**: Early stopping is used to prevent overfitting. This technique stops the training process if the validation loss does not improve for a specified number of epochs.

## Model Evaluation

In this section, we evaluate the performance of our deep learning model for kidney disease classification. The evaluation involves measuring the model's accuracy and loss on the validation dataset, and visualizing the results using plots.

- **Accuracy and Loss**:
  First, we calculate the accuracy and loss on the validation dataset. High accuracy and low loss indicate good model performance.

- **Confusion Matrix**:
  To gain further insights into the model's performance, we use a confusion matrix to visualize the classification results.

- **Classification Report**:
  The classification report provides a summary of the model's performance, including precision, recall, and F1 score for each class.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

## Resources

- [VisoAi VGG Blog](https://viso.ai/deep-learning/vgg-very-deep-convolutional-networks/#:~:text=The%20VGG16%20model%20achieves%20almost,models%20submitted%20to%20ILSVRC%2D2014.)
- [GFG VGG-16](https://www.geeksforgeeks.org/vgg-16-cnn-model/)
- [Kaggle Kidney Disease Dataset](https://www.kaggle.com/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone)
- [VGG16 Model Visualitation](https://youtu.be/RNnKtNrsrmg?si=7W3P2XSfgR5KWbvg)

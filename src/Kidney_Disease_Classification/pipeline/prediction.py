import numpy as np
import tensorflow as tf
import os
import base64
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from io import BytesIO
from PIL import Image


class PredictionPipeline:
    def __init__(self, image_data):
        self.image_data = image_data
        print(image_data)

    def decode_image(self):
        # Decode the base64 image data
        img_data = base64.b64decode(self.image_data)
        img = Image.open(BytesIO(img_data))
        return img

    def predict(self):
        # Load and preprocess the image
        img = Image.open(BytesIO(self.image_data)).convert("RGB")
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize the image

        # Load the binary classification model
        binary_model_path = "model/ct_scan_classification_model.h5"
        binary_model = tf.keras.models.load_model(binary_model_path)

        prediction_prob = binary_model.predict(img_array)
        ctScanConfidence = prediction_prob[0][0]

        if ctScanConfidence < 0.5:
            return {
                "result": "This is not a CT scan image",
                "confidence": "N/A",
                "ctScanConfidence": ctScanConfidence * 100,
            }

        # Load the classification model
        classification_model_path = "model/test/model.h5"
        classification_model = tf.keras.models.load_model(classification_model_path)

        # Make a prediction And Calculate the confidence
        predictions = classification_model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)
        confidence = np.max(predictions, axis=1)

        # Define class labels
        class_labels = {0: "CYST", 1: "NORMAL", 2: "STONE", 3: "TUMOR"}
        predicted_label = class_labels[predicted_class[0]]

        # Convert confidence to percentage
        confidence_percentage = confidence[0] * 100

        return {
            "result": predicted_label,
            "confidence": confidence_percentage,
            "ctScanConfidence": ctScanConfidence * 100,
        }

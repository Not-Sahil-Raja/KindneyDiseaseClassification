import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import base64
from io import BytesIO
from PIL import Image


class PredictionPipeline:
    def __init__(self, image_data):
        self.image_data = image_data

    def decode_image(self):
        # Decode the base64 image data
        img_data = base64.b64decode(self.image_data)
        img = Image.open(BytesIO(img_data))
        return img

    def predict(self):
        model_path = "model/test/model.h5"
        model = tf.keras.models.load_model(model_path)

        # Load and preprocess the image
        img = Image.open(BytesIO(self.image_data)).convert("RGB")
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize the image

        # Make a prediction
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)

        # Define class labels
        class_labels = {0: "CYST", 1: "NORMAL", 2: "STONE", 3: "TUMOR"}
        predicted_label = class_labels[predicted_class[0]]

        return {"result": predicted_label}

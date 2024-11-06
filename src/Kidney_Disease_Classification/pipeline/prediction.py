import numpy as np
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
        model = load_model(os.path.join("model", "model.h5"))
        # model = load_model(os.path.join("artifacts", "training", "model.h5"))

        img = self.decode_image()
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        result = np.argmax(model.predict(img_array), axis=1)
        print(result)

        if result[0] == 0:
            prediction = "Kidney Disease"
            return [{"image": prediction}]
        elif result[0] == 1:
            prediction = "Normal"
            return [{"image": prediction}]
        elif result[0] == 2:
            prediction = "Stone"
            return [{"image": prediction}]
        else:
            prediction = "Tumor"
            return [{"image": prediction}]

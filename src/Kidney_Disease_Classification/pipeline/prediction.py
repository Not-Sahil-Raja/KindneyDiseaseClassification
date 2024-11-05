import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import jsonify


class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    def predict(self):
        model = load_model(os.path.join("model", "model.h5"))
        # model = load_model(os.path.join("artifacts", "training", "model.h5"))

        imagename = self.filename
        test_image = image.load_img(imagename, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = np.argmax(model.predict(test_image), axis=1)
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

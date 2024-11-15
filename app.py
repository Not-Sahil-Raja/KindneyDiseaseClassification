from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from Kidney_Disease_Classification.pipeline.prediction import PredictionPipeline
import os

app = FastAPI()

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ClientApp:
    def __init__(self):
        self.classifier = None


clApp = ClientApp()


@app.get("/")
async def home():
    return {"message": "Welcome to Kidney Disease Classification API"}


@app.post("/train")
async def train_route():
    os.system("python main.py")
    # os.system("dvc repro")
    return {"message": "Training done successfully!"}


@app.post("/predict")
async def predict_route(request: Request):
    data = await request.json()
    image_data = data["image"]
    clApp.classifier = PredictionPipeline(image_data)
    result = clApp.classifier.predict()
    return result


@app.post("/predict/file")
async def predict_file_route(request: Request):
    data = await request.form()
    image_data = await data["file"].read()
    clApp.classifier = PredictionPipeline(image_data)
    result = clApp.classifier.predict()
    return result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)

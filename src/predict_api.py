from fastapi import FastAPI
from src.schema import PredictRequest
import mlflow.pyfunc
import pandas as pd

app = FastAPI(title="Housing Price Prediction API", version="1.0")

# Load model from MLflow Model Registry
MODEL_NAME = "Decision tree classifier"
MODEL_VERSION = "1"
MODEL_URI = f"models:/{MODEL_NAME}/{MODEL_VERSION}"

model = mlflow.pyfunc.load_model(MODEL_URI)

@app.get("/")
def read_root():
    return {"message": "Welcome to Housing Price Prediction API"}

@app.post("/predict")
def predict(request: PredictRequest):
    input_data = pd.DataFrame([request.dict()])

    prediction = model.predict(input_data)

    return {"prediction": prediction.tolist()}

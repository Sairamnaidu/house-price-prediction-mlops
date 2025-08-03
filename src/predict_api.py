from fastapi import FastAPI
from src.schema import PredictRequest
import mlflow.pyfunc
import pandas as pd
from logs.main import logger

app = FastAPI(title="Housing Price Prediction API", version="1.0")

# Load model from MLflow Model Registry
# MODEL_NAME = "Decision tree classifier"
# MODEL_VERSION = "1"
# MODEL_URI = f"models:/{MODEL_NAME}/{MODEL_VERSION}"

model = mlflow.pyfunc.load_model("mlruns/480688790370800283/models/m-4c4eaa154b62400c87d966daf1b0eebb/artifacts")

@app.get("/")
def read_root():
    return {"message": "Welcome to Housing Price Prediction API"}

@app.post("/predict")
def predict(request: PredictRequest):
    global request_count
    request_count += 1

    input_data = pd.DataFrame([request.dict()])
    logger.info(f"Received input: {input_data.to_dict(orient='records')}")

    prediction = model.predict(input_data)
    logger.info(f"Prediction result: {prediction.tolist()}")

    return {"prediction": prediction.tolist()}

@app.get("/metrics")
def metrics():
    return {"total_requests": request_count}

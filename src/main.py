from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd

from .learning.feature_engineering import process_dataset_for_training, process_dataset_for_prediction
from .learning.predicting import predict_target
from .learning.splitter import split_dataset
from .learning.training import train_rfc

import io
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

app = FastAPI(title="Stanford University Prediction API")

@app.post("/split")
async def split_dataset_endpoint(file: UploadFile = File(...), predict_size: float = Form(...)):
    if not (0.0 < predict_size < 1.0):
        raise HTTPException(status_code=400, detail="predict_size must be between 0 and 1")

    contents = await file.read()
    try:
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')), delimiter=",")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading CSV file: {e}")

    train_count, predict_count = split_dataset(df, predict_size)
    return {
        "message": "Dataset successfully split",
        "train_count": train_count,
        "predict_count": predict_count
    }

@app.post("/train")
async def train_endpoint(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')), delimiter=",")
        processed_df = process_dataset_for_training(df)
        report = train_rfc(processed_df)
        return {"message": "Model trained successfully!", "model_report": report}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')), delimiter=",")
        X_pred = process_dataset_for_prediction(df)
        y_pred = predict_target(X_pred)
        return {"message": "Predicted target values", "y_pred": y_pred}
    except Exception as e:
        print(e)
        return JSONResponse(status_code=404, content={"error": str(e)})

# run: uvicorn src.main:app --reload
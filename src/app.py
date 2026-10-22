import os
import time
import joblib
import pandas as pd
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from datetime import datetime

# Load models and pipelines
try:
    feature_engineer = joblib.load('models/feature_engineer.joblib')
    model_data = joblib.load('models/best_model.joblib')
    model = model_data['model']
    threshold = model_data['threshold']
    print(f"Loaded model with threshold {threshold:.4f}")
except FileNotFoundError:
    print("WARNING: Model files not found. Using dummy models for testing.")
    feature_engineer = None
    model = None
    threshold = 0.5

app = FastAPI(title="FraudLens API")

class Transaction(BaseModel):
    timestamp: datetime = Field(..., description="Transaction timestamp")
    user_id: int = Field(..., description="User ID")
    merchant_id: int = Field(..., description="Merchant ID")
    amount: float = Field(..., gt=0, description="Transaction amount")

class BatchTransaction(BaseModel):
    transactions: List[Transaction]

class FraudResponse(BaseModel):
    fraud_probability: float = Field(..., description="Probability of fraud")
    is_fraud: bool = Field(..., description="Boolean indicating if it's classified as fraud")
    threshold_used: float = Field(..., description="The decision threshold used")
    inference_time_ms: float = Field(..., description="Inference time in ms")

class BatchFraudResponse(BaseModel):
    results: List[FraudResponse]
    total_inference_time_ms: float

@app.get("/health")
def health_check():
    status = "healthy" if model is not None else "degraded (no model)"
    return {"status": status, "model_loaded": model is not None}

def _process_prediction(transactions: List[Transaction]) -> List[FraudResponse]:
    start_time = time.time()

    # convert to dataframe
    df = pd.DataFrame([t.model_dump() for t in transactions])

    # feature engineering
    if feature_engineer is not None:
        df_feat = feature_engineer.transform(df)
        X = df_feat.drop(columns=['is_fraud'], errors='ignore')
    else:
        # Dummy behavior for tests
        X = df[['amount']]

    # prediction
    if model is not None:
        y_prob = model.predict_proba(X)[:, 1]
    else:
        # Dummy predictions for tests
        y_prob = [0.1] * len(df)

    end_time = time.time()
    inference_time = (end_time - start_time) * 1000.0

    # build response
    responses = []
    # If multiple transactions, divide inference time equally for response
    ind_inf_time = inference_time / len(y_prob) if len(y_prob) > 0 else 0

    for prob in y_prob:
        is_fraud = bool(prob >= threshold)
        responses.append(FraudResponse(
            fraud_probability=float(prob),
            is_fraud=is_fraud,
            threshold_used=float(threshold),
            inference_time_ms=ind_inf_time
        ))

    return responses

@app.post("/predict", response_model=FraudResponse)
def predict(transaction: Transaction):
    try:
        results = _process_prediction([transaction])
        return results[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=BatchFraudResponse)
def predict_batch(batch: BatchTransaction):
    start_time = time.time()
    try:
        results = _process_prediction(batch.transactions)
        total_time = (time.time() - start_time) * 1000.0
        return BatchFraudResponse(
            results=results,
            total_inference_time_ms=total_time
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

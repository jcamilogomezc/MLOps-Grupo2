from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
import os
import numpy as np
import mlflow
import mlflow.pyfunc
from fastapi import FastAPI, HTTPException
from fastapi.responses import ORJSONResponse
from typing import List
import asyncio
from functools import lru_cache

# ---------- Config ----------
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
REGISTERED_MODEL = os.getenv("REGISTERED_MODEL_NAME", "linear_regression_model")
MODEL_STAGE_OR_VERSION = os.getenv("MODEL_STAGE_OR_VERSION", "Production")

# ---------- Pydantic Models ----------
class PredictionInput(BaseModel):
    feature1: float
    feature2: float
    feature3: float

    model_config = {"json_schema_extra": {"example": {"feature1": 0.5, "feature2": 1.2, "feature3": -0.3}}}

class PredictionOutput(BaseModel):
    prediction: float

# ---------- Model Storage ----------
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown"""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    try:
        uri = f"models:/{REGISTERED_MODEL}/{MODEL_STAGE_OR_VERSION}"
        ml_models["model"] = mlflow.pyfunc.load_model(uri)
        print(f"✅ Model loaded successfully from {uri}")
    except Exception as e:
        print(f"⚠️  Warning: Could not load model on startup: {e}")
        ml_models["model"] = None
    
    yield
    ml_models.clear()

app = FastAPI(
    lifespan=lifespan,
    default_response_class=ORJSONResponse,
    title="ML Inference API",
    version="3.0.0"
)

@app.get("/health")
async def health():
    """Fast health check"""
    return {"status": "ok", "model_loaded": ml_models.get("model") is not None}

@app.post("/predict", response_model=PredictionOutput)
async def predict(item: PredictionInput) -> PredictionOutput:
    """Optimized single prediction with NumPy"""
    model = ml_models.get("model")
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model unavailable")
    
    try:
        # CRITICAL: Use NumPy directly (10-50x faster than pandas)
        input_array = np.array([[item.feature1, item.feature2, item.feature3]], dtype=np.float64)
        y_pred = model.predict(input_array)
        return PredictionOutput(prediction=float(y_pred[0]))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
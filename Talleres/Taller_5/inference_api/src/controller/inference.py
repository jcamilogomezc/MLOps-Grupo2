from src.app import app
from pydantic import BaseModel, Field
import os
import pandas as pd
import mlflow
import mlflow.pyfunc
from fastapi import FastAPI, HTTPException

# ---------- Config ----------
# Se pueden sobreescribir por variables de entorno
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
REGISTERED_MODEL = os.getenv("REGISTERED_MODEL_NAME", "linear_regression_model")
MODEL_STAGE_OR_VERSION = os.getenv("MODEL_STAGE_OR_VERSION", "Production")

class PredictionInput(BaseModel):
    feature1: float = Field(..., description="Feature 1")
    feature2: float = Field(..., description="Feature 2")
    feature3: float = Field(..., description="Feature 3")

    class Config:
        json_schema_extra = {
            "example": {
                "feature1": 0.5,
                "feature2": 1.2,
                "feature3": -0.3
            }
        }

class PredictionOutput(BaseModel):
    prediction: float = Field(..., description="Predicted value")

# ---------- Carga de modelo ----------
_model: mlflow.pyfunc.PyFuncModel | None = None

def _load_model():
    global _model, _label_names
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # Preferimos el Model Registry
    # - Stage: models:/<name>/<Stage>  (p.ej. Production)
    # - Versión concreta: models:/<name>/<version>
    uri = f"models:/{REGISTERED_MODEL}/{MODEL_STAGE_OR_VERSION}"
    _model = mlflow.pyfunc.load_model(uri)

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(item: PredictionInput) -> PredictionOutput:
    global _model
    try:
        _load_model()
    except:
        raise HTTPException(status_code=404, detail="Model not found in production yet")

    if _model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    df = pd.DataFrame([item.dict()])  # una fila
    y_pred = _model.predict(df)

    return PredictionOutput(prediction=y_pred)
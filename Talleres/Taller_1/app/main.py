from fastapi import FastAPI, HTTPException
from app.schemas import PredictRequest, PredictResponse
from app.models_loader import load_models
import pandas as pd

app = FastAPI(title="Penguins Serving", version="1.0.0", docs_url="/")

MODELS, TARGET_ENCODER = load_models()

@app.get("/health")
def health():
    return {
        "status": "ok",
        "available_models": list(MODELS.keys()),
        "classes": list(TARGET_ENCODER.classes_),
    }

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if req.model_name not in MODELS:
        raise HTTPException(status_code=400, detail=f"Modelo {req.model_name} no disponible")
    df = pd.DataFrame([x.model_dump() for x in req.items])
    pred_ids = MODELS[req.model_name].predict(df)
    pred_labels = TARGET_ENCODER.inverse_transform(pred_ids)
    return PredictResponse(model_name=req.model_name, predictions=pred_labels.tolist(), count=len(pred_labels))
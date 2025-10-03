# START API!

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import os
import mlflow
import mlflow.pyfunc
import pandas as pd

# ---------- Config ----------
# Se pueden sobreescribir por variables de entorno
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
REGISTERED_MODEL = os.getenv("REGISTERED_MODEL_NAME", "CovertypeClassifier")
MODEL_STAGE_OR_VERSION = os.getenv("MODEL_STAGE_OR_VERSION", "production")  # "Production" o "Staging" o "1", "2", ...

# Necesario para bajar artefactos desde MinIO (S3 compatible)
os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "admin")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "supersecret")

app = FastAPI(title="Covertype Inference API", version="1.0.0")

# ---------- Esquemas ----------

class CovertypeInput(BaseModel):
    """Input schema for Covertype classification predictions"""
    elevation: int = Field(..., description="Elevation in meters")
    aspect: int = Field(..., ge=0, le=360, description="Aspect in degrees azimuth")
    slope: int = Field(..., ge=0, description="Slope in degrees")
    horizontal_distance_to_hydrology: int = Field(..., description="Horizontal distance to nearest surface water features")
    vertical_distance_to_hydrology: int = Field(..., description="Vertical distance to nearest surface water features")
    horizontal_distance_to_roadways: int = Field(..., description="Horizontal distance to nearest roadway")
    hillshade_9am: int = Field(..., ge=0, le=255, description="Hillshade index at 9am, summer solstice")
    hillshade_noon: int = Field(..., ge=0, le=255, description="Hillshade index at noon, summer solstice")
    hillshade_3pm: int = Field(..., ge=0, le=255, description="Hillshade index at 3pm, summer solstice")
    horizontal_distance_to_fire_points: int = Field(..., description="Horizontal distance to nearest wildfire ignition points")
    wilderness_area: str = Field(..., description="Wilderness area designation")
    soil_type: str = Field(..., description="Soil type designation")

    class Config:
        json_schema_extra = {
            "example": {
                "elevation": 2500,
                "aspect": 180,
                "slope": 15,
                "horizontal_distance_to_hydrology": 250,
                "vertical_distance_to_hydrology": 50,
                "horizontal_distance_to_roadways": 1500,
                "hillshade_9am": 200,
                "hillshade_noon": 220,
                "hillshade_3pm": 150,
                "horizontal_distance_to_fire_points": 3000,
                "wilderness_area": "Rawah",
                "soil_type": "C2702"
            }
        }


class PredictionOutput(BaseModel):
    cover_type: str = Field(..., description="Predicted forest cover type (1-7)")
    probabilities: dict = Field(..., description="Probability distribution over all cover types")

# ---------- Carga de modelo ----------
_model: mlflow.pyfunc.PyFuncModel | None = None
_label_names: list[str] | None = None

def _load_model():
    global _model, _label_names
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # Preferimos el Model Registry
    # - Stage: models:/<name>/<Stage>  (p.ej. Production)
    # - Versión concreta: models:/<name>/<version>
    uri = f"models:/{REGISTERED_MODEL}/{MODEL_STAGE_OR_VERSION}"
    _model = mlflow.pyfunc.load_model(uri)

    # Intentamos extraer labels si el submodelo es sklearn classifier
    try:
        sk_pipeline = _model._model_impl.python_model.model  # Pipeline sklearn
        clf = sk_pipeline.named_steps.get("classifier", None)
        if clf is not None and hasattr(clf, "classes_"):
            _label_names = [str(c) for c in clf.classes_]
    except Exception:
        _label_names = None

@app.on_event("startup")
def startup_event():
    _load_model()

# ---------- Endpoints ----------
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionOutput)
def predict(item: CovertypeInput):
    if _model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    df = pd.DataFrame([item.dict()])  # una fila
    y_pred = _model.predict(df)

    # probabilidades (si el clasificador las soporta)
    probs = {}
    try:
        proba = _model.predict_proba(df)[0]
        if _label_names is not None and len(_label_names) == len(proba):
            probs = {str(lbl): float(p) for lbl, p in zip(_label_names, proba)}
        else:
            probs = {str(i): float(p) for i, p in enumerate(proba)}
    except Exception:
        pass

    return PredictionOutput(cover_type=str(y_pred[0]), probabilities=probs)
# app/models_loader.py
import os, joblib, sys, types
from typing import Dict, Tuple
from pathlib import Path

# 1) Importa tus clases
from app.custom_transformers import CategoryCleaner#, FeatureBuilder

# 2) Regístralas bajo __main__ para compatibilidad con modelos pickled desde notebooks/scripts
_main = sys.modules.get("__main__")
if _main is None:
    _main = types.ModuleType("__main__")
    sys.modules["__main__"] = _main
setattr(_main, "CategoryCleaner", CategoryCleaner)
#setattr(_main, "FeatureBuilder", FeatureBuilder)

MODEL_DIR = Path(os.getenv("MODEL_DIR", "/models"))

MODEL_FILES = {
    "MLP":    "penguins_pipeline_MLP.joblib",
    "HGB":    "penguins_pipeline_HGB.joblib",
    "LogReg": "penguins_pipeline_LogReg.joblib",
}
TARGET_ENCODER_FILE = "penguins_target_encoder.joblib"

def load_models() -> Tuple[Dict[str, object], object]:
    models = {}
    for name, fname in MODEL_FILES.items():
        path = MODEL_DIR / fname
        if not path.exists():
            raise FileNotFoundError(f"No se encontró {fname} en {MODEL_DIR}")
        models[name] = joblib.load(path)
    enc_path = MODEL_DIR / TARGET_ENCODER_FILE
    if not enc_path.exists():
        raise FileNotFoundError(f"No se encontró {TARGET_ENCODER_FILE} en {MODEL_DIR}")
    target_encoder = joblib.load(enc_path)
    return models, target_encoder
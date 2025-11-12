# START API!


from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field
from typing import Dict, Any
import os
import mlflow
import mlflow.pyfunc
import mlflow.tracking
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np
import json
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import asyncio
from concurrent.futures import ThreadPoolExecutor


# ---------- Config ----------
# Se pueden sobreescribir por variables de entorno
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
REGISTERED_MODEL = os.getenv("REGISTERED_MODEL_NAME", "diabetes_readmission_model")
MODEL_STAGE_OR_VERSION = os.getenv("MODEL_STAGE_OR_VERSION", "Production")  # "Production" o "Staging" o "1", "2", ...


app = FastAPI(title="Diabetes Readmission Inference API", version="2.3.0")


# ---------- Thread Pool for CPU-bound operations ----------
# Use a thread pool executor for CPU-bound ML inference tasks
# This allows async endpoints to handle multiple requests concurrently
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ml_inference")


# ---------- Prometheus Metrics ----------
REQUEST_COUNT = Counter(
    'predict_requests_total', 
    'Total number of prediction requests',
    ['method', 'endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'predict_latency_seconds', 
    'Prediction request latency in seconds',
    ['endpoint']
)

PREDICTION_COUNTER = Counter(
    'predictions_total', 
    'Total number of predictions', 
    ['prediction_class', 'status']
)

ERROR_COUNTER = Counter(
    'predict_errors_total', 
    'Total number of prediction errors', 
    ['error_type', 'endpoint']
)

IN_PROGRESS = Gauge(
    'predict_requests_in_progress', 
    'Number of prediction requests currently being processed',
    ['endpoint']
)


# ---------- Esquemas ----------


class DiabetesInput(BaseModel):
    """Input schema for Diabetes readmission prediction"""
    # Accept features as a flexible dictionary since there are 50+ features
    # Common features include: race, gender, age, weight, admission_type_id, 
    # discharge_disposition_id, medical_specialty, num_lab_procedures, medications, etc.
    features: Dict[str, Any] = Field(..., description="Dictionary of patient features for readmission prediction")


    class Config:
        json_schema_extra = {
            "example": {
                "features": {
                    "race": "Caucasian",
                    "gender": "Female",
                    "age": "[70-80)",
                    "weight": "?",
                    "admission_type_id": 1,
                    "discharge_disposition_id": 1,
                    "admission_source_id": 7,
                    "time_in_hospital": 1,
                    "payer_code": "?",
                    "medical_specialty": "Emergency/Trauma",
                    "num_lab_procedures": 41,
                    "num_procedures": 0,
                    "num_medications": 1,
                    "number_outpatient": 0,
                    "number_emergency": 0,
                    "number_inpatient": 0,
                    "diag_1": "250.83",
                    "diag_2": "?",
                    "diag_3": "?",
                    "number_diagnoses": 1,
                    "max_glu_serum": "None",
                    "A1Cresult": "None",
                    "metformin": "No",
                    "repaglinide": "No",
                    "nateglinide": "No",
                    "chlorpropamide": "No",
                    "glimepiride": "No",
                    "acetohexamide": "No",
                    "glipizide": "No",
                    "glyburide": "No",
                    "tolbutamide": "No",
                    "pioglitazone": "No",
                    "rosiglitazone": "No",
                    "acarbose": "No",
                    "miglitol": "No",
                    "troglitazone": "No",
                    "tolazamide": "No",
                    "examide": "No",
                    "citoglipton": "No",
                    "insulin": "No",
                    "glyburide-metformin": "No",
                    "glipizide-metformin": "No",
                    "glimepiride-pioglitazone": "No",
                    "metformin-rosiglitazone": "No",
                    "metformin-pioglitazone": "No",
                    "change": "No",
                    "diabetesMed": "No"
                }
            }
        }



class PredictionOutput(BaseModel):
    readmission_prediction: str = Field(..., description="Predicted readmission status (NO, <30, >30)")
    probabilities: Dict[str, float] = Field(..., description="Probability distribution over all readmission classes")


# ---------- Carga de modelo ----------
_model: mlflow.pyfunc.PyFuncModel | None = None
_label_names: list[str] | None = None
_expected_features: list[str] | None = None
_model_version: str | None = None
_label_encoders: Dict[str, LabelEncoder] | None = None
_scaler: StandardScaler | None = None
_run_id: str | None = None


def _load_model():
    """Load the diabetes readmission model from MLflow Model Registry"""
    global _model, _label_names, _expected_features, _model_version, _label_encoders, _scaler, _run_id
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()


    # Load model from Model Registry
    # - Stage: models:/<name>/<Stage>  (e.g. Production)
    # - Version: models:/<name>/<version>
    uri = f"models:/{REGISTERED_MODEL}/{MODEL_STAGE_OR_VERSION}"
    try:
        _model = mlflow.pyfunc.load_model(uri)
        print(f"✓ Model loaded from {uri}")
    except Exception as e:
        raise ValueError(f"Model not found: {str(e)}. Ensure model is registered in MLflow Model Registry.")


    # Extract label names from sklearn classifier
    try:
        # Try to get the underlying sklearn model
        sk_model = _model._model_impl
        if hasattr(sk_model, 'python_model'):
            sk_model = sk_model.python_model
        
        # If it's a sklearn model, get classes
        if hasattr(sk_model, 'classes_'):
            _label_names = [str(c) for c in sk_model.classes_]
        # If it's wrapped, try to get the model attribute
        elif hasattr(sk_model, 'model') and hasattr(sk_model.model, 'classes_'):
            _label_names = [str(c) for c in sk_model.model.classes_]
        else:
            # Default labels for diabetes readmission (NO, <30, >30)
            _label_names = ["NO", "<30", ">30"]
        print(f"✓ Label names: {_label_names}")
    except Exception as e:
        print(f"⚠ Could not extract label names: {e}")
        # Default labels for diabetes readmission
        _label_names = ["NO", "<30", ">30"]


    # Try to load expected feature names and preprocessing artifacts from MLflow
    _run_id = None
    try:
        # Get the model version info to access artifacts
        model_versions = client.search_model_versions(f"name='{REGISTERED_MODEL}'")
        if model_versions:
            # Get the production version or specified version
            for mv in model_versions:
                if MODEL_STAGE_OR_VERSION in ["Production", "Staging"]:
                    if mv.current_stage == MODEL_STAGE_OR_VERSION:
                        _run_id = mv.run_id
                        _model_version = mv.version
                        break
                else:
                    # It's a version number
                    if mv.version == MODEL_STAGE_OR_VERSION:
                        _run_id = mv.run_id
                        _model_version = mv.version
                        break
            
            if _run_id:
                try:
                    # Try to download feature_names.json artifact
                    local_path = client.download_artifacts(_run_id, 'feature_names.json')
                    with open(local_path, 'r') as f:
                        feature_data = json.load(f)
                        _expected_features = feature_data.get('feature_names', [])
                    print(f"✓ Loaded {len(_expected_features)} expected features from MLflow")
                except Exception as e:
                    print(f"⚠ Could not load feature names: {e}")
                    _expected_features = None
                
                # Try to load label encoders (if saved)
                try:
                    encoders_path = client.download_artifacts(_run_id, 'label_encoders.pkl')
                    with open(encoders_path, 'rb') as f:
                        _label_encoders = pickle.load(f)
                    print(f"✓ Loaded {len(_label_encoders)} label encoders from MLflow")
                except Exception as e:
                    print(f"⚠ Could not load label encoders: {e}. Will create new encoders (predictions may be inaccurate).")
                    _label_encoders = None
                
                # Try to load scaler (if saved)
                try:
                    scaler_path = client.download_artifacts(_run_id, 'scaler.pkl')
                    with open(scaler_path, 'rb') as f:
                        _scaler = pickle.load(f)
                    print(f"✓ Loaded scaler from MLflow")
                except Exception as e:
                    print(f"⚠ Could not load scaler: {e}. Will create new scaler (predictions may be inaccurate).")
                    _scaler = None
    except Exception as e:
        print(f"⚠ Could not access model version info: {e}")
        _expected_features = None
        _label_encoders = None
        _scaler = None


# ---------- Preprocessing ----------
def _clean_input_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean input data similar to training pipeline.
    Replicates cleaning logic from dag_04_clean_training_data.py and dag_05b_train_cumulative_batches.py
    """
    # Replace '?' with NaN
    df = df.replace('?', np.nan)
    
    # Remove ID columns if they exist (not needed for prediction)
    id_columns = ['encounter_id', 'patient_nbr', 'batch_id']
    existing_id_cols = [col for col in id_columns if col in df.columns]
    if existing_id_cols:
        df = df.drop(columns=existing_id_cols)
    
    # Fill missing values based on data type
    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype == 'object':
                # For categorical: fill with 'Unknown' or most common value
                if df[col].notna().any():
                    mode_val = df[col].mode()
                    if len(mode_val) > 0:
                        df[col] = df[col].fillna(mode_val[0])
                    else:
                        df[col] = df[col].fillna('Unknown')
                else:
                    df[col] = df[col].fillna('Unknown')
            else:
                # For numerical: fill with 0 (median would require training data)
                df[col] = df[col].fillna(0)
    
    return df



def _preprocess_features(features: Dict[str, Any]) -> np.ndarray:
    """
    Preprocess features for model prediction.
    This function replicates the preprocessing pipeline from training:
    1. Clean data (handle '?', missing values)
    2. Align features with expected feature order
    3. Encode categorical features using LabelEncoder
    4. Scale features using StandardScaler
    
    Returns:
        numpy array of preprocessed features ready for model prediction
    """
    global _expected_features, _label_encoders, _scaler
    
    # Convert to DataFrame
    df = pd.DataFrame([features])
    
    # Clean the data
    df = _clean_input_data(df)
    
    # Align features with expected feature order if available
    if _expected_features is not None:
        # Add missing features with appropriate default values
        for feat in _expected_features:
            if feat not in df.columns:
                # Use 'Unknown' for categorical, 0 for numerical
                # We'll determine type based on expected features
                df[feat] = 'Unknown'  # Default to string, will be handled in encoding
        
        # Remove extra features (features not in expected list)
        extra_features = [col for col in df.columns if col not in _expected_features]
        if extra_features:
            df = df.drop(columns=extra_features)
        
        # Reorder columns to match expected feature order
        df = df[_expected_features]
    else:
        print("⚠ Warning: Expected features not loaded. Using provided features as-is.")
    
    # Encode categorical variables
    # Create a copy for encoding
    df_encoded = df.copy()
    
    # First, try to convert numeric columns that might be strings
    for col in df_encoded.columns:
        if df_encoded[col].dtype == 'object':
            # Try to convert to numeric first
            try:
                numeric_val = pd.to_numeric(df_encoded[col], errors='raise')
                df_encoded[col] = numeric_val
            except (ValueError, TypeError):
                # If conversion fails, it's likely categorical
                pass
    
    # Now encode remaining categorical (object/string) columns
    for col in df_encoded.columns:
        if df_encoded[col].dtype == 'object':
            # This is a categorical feature (string type)
            # Get the value as string
            col_value = str(df_encoded[col].iloc[0])
            
            if _label_encoders is not None and col in _label_encoders:
                # Use the fitted encoder from training
                le = _label_encoders[col]
                # Check if the value exists in the encoder's classes
                if col_value in le.classes_:
                    encoded_value = le.transform([col_value])[0]
                    df_encoded[col] = encoded_value
                else:
                    # Unseen label - use the first class from training as fallback
                    fallback_value = le.classes_[0]
                    encoded_value = le.transform([fallback_value])[0]
                    df_encoded[col] = encoded_value
                    print(f"⚠ Warning: Unseen value '{col_value}' in '{col}'. Replaced with '{fallback_value}' (encoded as {encoded_value}).")
            else:
                # No fitted encoder available
                # CRITICAL: Without the fitted encoder, we can't match training encoding exactly
                # This is a limitation - encoders should be saved during training
                # For now, use a simple numeric mapping (0, 1, 2, ...) based on sorted unique values
                # This won't match training but will allow the API to work
                # In production, encoders MUST be saved to MLflow artifacts
                print(f"⚠ ERROR: No encoder found for '{col}' with value '{col_value}'. "
                      f"Encoders must be saved during training. Using default encoding (0). "
                      f"Predictions will be inaccurate!")
                df_encoded[col] = 0
        
        # Final check: ensure all values are numeric
        if df_encoded[col].dtype == 'object':
            try:
                df_encoded[col] = pd.to_numeric(df_encoded[col], errors='coerce')
                df_encoded[col] = df_encoded[col].fillna(0)
            except Exception as e:
                print(f"⚠ Warning: Could not convert '{col}' to numeric: {e}. Using 0.")
                df_encoded[col] = 0
    
    # Convert to numpy array
    X = df_encoded.values.astype(float)
    
    # Scale features if scaler is available
    # IMPORTANT: The model was trained on scaled features, so we MUST scale for accurate predictions
    if _scaler is not None:
        try:
            X = _scaler.transform(X)
        except Exception as e:
            print(f"⚠ ERROR: Could not scale features with saved scaler: {e}")
            print("⚠ WARNING: Proceeding without scaling. Predictions will be INACCURATE!")
            # Don't raise - allow prediction to proceed but it will be wrong
    else:
        # No scaler available - critical warning
        print("⚠ ERROR: No scaler loaded from MLflow!")
        print("⚠ WARNING: The model was trained on scaled features. ")
        print("⚠ WARNING: Predictions without scaling will be INACCURATE!")
        print("⚠ WARNING: Please ensure the scaler is saved as 'scaler.pkl' in MLflow artifacts during training.")
        # Don't scale - this will cause inaccurate predictions but allows the API to work
        # In production, you MUST save the scaler during training
    
    return X


# ---------- Endpoints ----------
@app.get("/health")
async def health():
    """Health check endpoint"""
    REQUEST_COUNT.labels(method='GET', endpoint='/health', status='200').inc()
    return {"status": "ok", "model": REGISTERED_MODEL}


@app.get("/model-info")
async def model_info():
    """Get information about the loaded model"""
    global _model, _label_names, _expected_features, _model_version, _label_encoders, _scaler
    
    if _model is None:
        try:
            # Load model in thread pool to avoid blocking
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(_executor, _load_model)
        except ValueError as e:
            ERROR_COUNTER.labels(error_type='model_not_found', endpoint='/model-info').inc()
            REQUEST_COUNT.labels(method='GET', endpoint='/model-info', status='404').inc()
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            ERROR_COUNTER.labels(error_type='model_load_error', endpoint='/model-info').inc()
            REQUEST_COUNT.labels(method='GET', endpoint='/model-info', status='500').inc()
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
    
    REQUEST_COUNT.labels(method='GET', endpoint='/model-info', status='200').inc()
    return {
        "model_name": REGISTERED_MODEL,
        "model_stage_or_version": MODEL_STAGE_OR_VERSION,
        "model_version": _model_version,
        "label_names": _label_names,
        "expected_features_count": len(_expected_features) if _expected_features else None,
        "expected_features": _expected_features[:10] if _expected_features else None,  # Show first 10
        "preprocessing_status": {
            "label_encoders_loaded": _label_encoders is not None,
            "num_label_encoders": len(_label_encoders) if _label_encoders else 0,
            "scaler_loaded": _scaler is not None,
            "warning": "Predictions may be inaccurate if encoders/scaler are not loaded. Ensure they are saved during training."
        }
    }


class ModelNotFoundError(Exception):
    """Exception raised when model is not found"""
    pass


class ModelLoadError(Exception):
    """Exception raised when model fails to load"""
    pass


def _run_prediction(item_features: Dict[str, Any]) -> PredictionOutput:
    """
    Synchronous prediction function that runs in a thread pool.
    This function performs all CPU-bound operations (preprocessing, prediction).
    Raises regular exceptions (not HTTPException) which will be handled in the async endpoint.
    """
    global _model, _label_names
    
    # Load model if not already loaded (thread-safe check)
    if _model is None:
        try:
            _load_model()
        except ValueError as e:
            ERROR_COUNTER.labels(error_type='model_not_found', endpoint='/predict').inc()
            raise ModelNotFoundError(str(e))
        except Exception as e:
            ERROR_COUNTER.labels(error_type='model_load_error', endpoint='/predict').inc()
            raise ModelLoadError(f"Failed to load model: {str(e)}")


    if _model is None:
        ERROR_COUNTER.labels(error_type='model_not_loaded', endpoint='/predict').inc()
        raise ModelLoadError("Model not loaded")


    # Preprocess features
    X = _preprocess_features(item_features)
    
    # Reshape to 2D array if needed (model expects 2D input)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    
    # Make prediction
    y_pred = _model.predict(X)
    prediction = str(y_pred[0])
    
    # Get probabilities if available
    probs = {}
    try:
        proba = _model.predict_proba(X)[0]
        if _label_names is not None and len(_label_names) == len(proba):
            probs = {str(lbl): float(p) for lbl, p in zip(_label_names, proba)}
        else:
            # Use indices if label names don't match
            probs = {str(i): float(p) for i, p in enumerate(proba)}
    except Exception as e:
        print(f"⚠ Could not get probabilities: {e}")
        # If probabilities not available, set prediction probability to 1.0
        probs = {prediction: 1.0}
    
    # Map prediction to label names if needed
    if _label_names is not None and prediction.isdigit():
        pred_idx = int(prediction)
        if 0 <= pred_idx < len(_label_names):
            prediction = _label_names[pred_idx]
    
    # Increment prediction counter by class (successful prediction)
    PREDICTION_COUNTER.labels(prediction_class=prediction, status='success').inc()
    
    return PredictionOutput(
        readmission_prediction=prediction,
        probabilities=probs
    )


@app.post("/predict", response_model=PredictionOutput)
async def predict(item: DiabetesInput):
    """
    Predict diabetes patient readmission.
    This async endpoint runs CPU-bound operations in a thread pool for better concurrency.
    """
    # Track in-progress requests
    IN_PROGRESS.labels(endpoint='/predict').inc()
    
    try:
        # Measure prediction latency
        with REQUEST_LATENCY.labels(endpoint='/predict').time():
            # Run CPU-bound prediction in thread pool to avoid blocking the event loop
            # This allows the server to handle other requests while processing this one
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(_executor, _run_prediction, item.features)
        
        # Count successful request after completion
        REQUEST_COUNT.labels(method='POST', endpoint='/predict', status='200').inc()
        return result
        
    except ModelNotFoundError as e:
        # Handle model not found error
        REQUEST_COUNT.labels(method='POST', endpoint='/predict', status='404').inc()
        PREDICTION_COUNTER.labels(prediction_class='unknown', status='failed').inc()
        raise HTTPException(status_code=404, detail=str(e))
    except ModelLoadError as e:
        # Handle model load error
        REQUEST_COUNT.labels(method='POST', endpoint='/predict', status='500').inc()
        PREDICTION_COUNTER.labels(prediction_class='unknown', status='failed').inc()
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        # Handle other prediction errors
        ERROR_COUNTER.labels(error_type='prediction_error', endpoint='/predict').inc()
        REQUEST_COUNT.labels(method='POST', endpoint='/predict', status='500').inc()
        PREDICTION_COUNTER.labels(prediction_class='unknown', status='failed').inc()
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}. Ensure input features match model expectations."
        )
    finally:
        # Decrement in-progress counter
        IN_PROGRESS.labels(endpoint='/predict').dec()


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    REQUEST_COUNT.labels(method='GET', endpoint='/metrics', status='200').inc()
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
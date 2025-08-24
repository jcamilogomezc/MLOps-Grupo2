from fastapi import FastAPI, HTTPException
from data_models import PredictRequest, PredictResponse
import pandas as pd
import numpy as np
import joblib
import json
import os

app = FastAPI(title="Penguin Species Prediction API", version="2.0", description="API for predicting penguin species using trained models.")


# Read the files in the models folder and create a dictionary with the model name as the key and the file path as the value 
# Select only the files that end with .joblib
def read_models_paths():
    return {file.split(".")[0]: f"../models/{file}" for file in os.listdir("../models") if file.endswith(".joblib")}

@app.get("/")
def read_root():
    return {"message": "Welcome to the Penguin Species Prediction API, UPDATED!"}

# Server status
@app.get("/status")
def get_status():
    model_paths = read_models_paths()
    return {"status": "running", 
            "version": "2.0", 
            "Available models": list(model_paths.keys())}

# Predict endpoint
@app.post("/predict", response_model=PredictResponse)
def predict_species(request: PredictRequest) -> PredictResponse:
    model_paths = read_models_paths()
    try:
        # Check if the requested model exists
        if request.model not in model_paths:
            raise HTTPException(status_code=404, detail=f'Model {request.model} not found.')
        # Load the model based on the request
        model = joblib.load(f'../models/{model_paths[request.model]}')

        # Prepare the input data for prediction
        df = pd.DataFrame([features.model_dump() for features in request.penguins])

        # Make predictions
        predictions = model.predict(df)

        # Aplanar la lista si es necesario. Esto convierte [[...], [...]] en [...]. Catboost lo arroja así
        if isinstance(predictions, np.ndarray) and predictions.ndim > 1:
            flat_predictions = [item[0] for item in predictions]
        else:
            flat_predictions = predictions.tolist()

        # Preparar respuesta
        response = PredictResponse(
            model=request.model,
            species=flat_predictions,
            num_predictions=len(flat_predictions)
        )

        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

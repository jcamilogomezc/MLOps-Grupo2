"""
API de Predicción de Especies de Pingüinos

Este módulo proporciona una aplicación FastAPI para predecir especies de pingüinos 
utilizando modelos de machine learning entrenados. La API soporta múltiples modelos 
y puede manejar predicciones en lote para múltiples pingüinos.

Versión: 2.0
"""

from fastapi import FastAPI, HTTPException
from data_models import PredictRequest, PredictResponse
import pandas as pd
import numpy as np
import joblib
import os
from typing import Dict, List, Union

# Inicializar aplicación FastAPI con metadatos
app = FastAPI(
    title="API de Predicción de Especies de Pingüinos", 
    version="2.0", 
    description="API para predecir especies de pingüinos usando modelos entrenados."
)


def read_models_paths() -> Dict[str, str]:
    """
    Lee los archivos de modelos disponibles del directorio de modelos.
    
    Escanea el directorio '../models' en busca de archivos que terminen con 
    la extensión '.joblib' y crea un mapeo de nombres de modelos a sus rutas de archivo.
    
    Returns:
        Dict[str, str]: Diccionario donde las claves son nombres de modelos (sin extensión) 
                       y los valores son rutas de archivo relativas.
    
    Ejemplo:
        >>> read_models_paths()
        {'random_forest': '../models/random_forest.joblib', 
         'gradient_boosting': '../models/gradient_boosting.joblib'}
    """
    return {file.split(".")[0]: f"../models/{file}" 
            for file in os.listdir("../models") 
            if file.endswith(".joblib")}


@app.get("/")
def read_root() -> Dict[str, str]:
    """
    Endpoint raíz que devuelve un mensaje de bienvenida.
    
    Returns:
        Dict[str, str]: Mensaje de bienvenida para la API.
    """
    return {"message": "¡Bienvenido a la API de Predicción de Especies de Pingüinos, ACTUALIZADA!"}


@app.get("/status")
def get_status() -> Dict[str, Union[str, List[str]]]:
    """
    Obtiene el estado actual del servidor API.
    
    Devuelve información sobre el estado de la API, versión y modelos disponibles.
    
    Returns:
        Dict[str, Union[str, List[str]]]: Diccionario que contiene:
            - status: Estado actual del servidor
            - version: Versión de la API
            - Available models: Lista de nombres de modelos disponibles
    """
    model_paths = read_models_paths()
    return {
        "status": "running", 
        "version": "2.0", 
        "Available models": list(model_paths.keys())
    }


@app.post("/predict", response_model=PredictResponse)
def predict_species(request: PredictRequest) -> PredictResponse:
    """
    Predice la especie de pingüino usando el modelo especificado.
    
    Este endpoint acepta una solicitud que contiene características de pingüinos 
    y un nombre de modelo, luego devuelve predicciones para la especie de cada pingüino.
    
    Args:
        request (PredictRequest): Objeto de solicitud que contiene:
            - model: Nombre del modelo a usar para la predicción
            - penguins: Lista de objetos con características de pingüinos
    
    Returns:
        PredictResponse: Objeto de respuesta que contiene:
            - model: Nombre del modelo utilizado
            - species: Lista de especies predichas
            - num_predictions: Número de predicciones realizadas
    
    Raises:
        HTTPException: 
            - 404: Si el modelo solicitado no se encuentra
            - 500: Si hay un error durante la predicción
    
    Ejemplo:
        POST /predict
        {
            "model": "random_forest",
            "penguins": [
                {"bill_length_mm": 39.1, "bill_depth_mm": 18.7, ...}
            ]
        }
    """
    model_paths = read_models_paths()
    
    try:
        # Verificar si el modelo solicitado existe
        if request.model not in model_paths:
            raise HTTPException(
                status_code=404, 
                detail=f'Modelo {request.model} no encontrado. Modelos disponibles: {list(model_paths.keys())}'
            )
        
        # Cargar el modelo desde el archivo
        model_path = model_paths[request.model]
        model = joblib.load(model_path)

        # Preparar los datos de entrada para la predicción
        # Convertir características de pingüinos a formato DataFrame
        df = pd.DataFrame([features.model_dump() for features in request.penguins])

        # Realizar predicciones usando el modelo cargado
        predictions = model.predict(df)

        # Manejar diferentes formatos de salida de predicción
        # Algunos modelos (como CatBoost) devuelven arrays anidados que necesitan ser aplanados
        if isinstance(predictions, np.ndarray) and predictions.ndim > 1:
            flat_predictions = [item[0] for item in predictions]
        else:
            flat_predictions = predictions.tolist()

        # Crear y devolver la respuesta
        response = PredictResponse(
            model=request.model,
            species=flat_predictions,
            num_predictions=len(flat_predictions)
        )

        return response
    
    except HTTPException:
        # Re-lanzar excepciones HTTP (como 404) sin modificación
        raise
    except Exception as e:
        # Capturar cualquier otra excepción y devolver un error 500
        raise HTTPException(status_code=500, detail=f"Error de predicción: {str(e)}")

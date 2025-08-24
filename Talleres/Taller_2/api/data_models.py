from pydantic import BaseModel, Field
from typing import Literal, List, Optional

# Los modelos pueden recibir datos nulos o faltantes ya que se incluyó imputadores en el pipeline del modelo

class PenguinFeatures(BaseModel):
    """
    Modelo de datos para las características de un pingüino.
    
    Este modelo define la estructura de datos para las características
    físicas y demográficas de un pingüino que serán utilizadas para
    la predicción de especies.
    
    Todos los campos son opcionales ya que el pipeline del modelo
    incluye imputadores para manejar valores faltantes.
    """
    
    island: Optional[Literal['Biscoe', 'Dream', 'Torgersen']] = Field(
        None, 
        description="Isla donde fue observado el pingüino"
    )
    bill_length_mm: Optional[float] = Field(
        None, 
        ge=0, 
        description="Longitud del pico en milímetros (debe ser >= 0)"
    )
    bill_depth_mm: Optional[float] = Field(
        None, 
        ge=0, 
        description="Profundidad del pico en milímetros (debe ser >= 0)"
    )
    flipper_length_mm: Optional[float] = Field(
        None, 
        ge=0, 
        description="Longitud de la aleta en milímetros (debe ser >= 0)"
    )
    body_mass_g: Optional[float] = Field(
        None, 
        ge=0, 
        description="Masa corporal en gramos (debe ser >= 0)"
    )
    sex: Optional[Literal['male', 'female']] = Field(
        None, 
        description="Sexo del pingüino"
    )

class PredictRequest(BaseModel):
    """
    Modelo de datos para las solicitudes de predicción.
    
    Define la estructura de datos requerida para realizar
    predicciones de especies de pingüinos.
    """
    
    model: str = Field(
        description="Nombre del modelo a utilizar para la predicción"
    )
    penguins: List[PenguinFeatures] = Field(
        description="Lista de pingüinos con sus características para predecir"
    )

class PredictResponse(BaseModel):
    """
    Modelo de datos para las respuestas de predicción.
    
    Define la estructura de datos que se devuelve después
    de realizar predicciones de especies de pingüinos.
    """
    
    model: str = Field(
        description="Nombre del modelo utilizado para la predicción"
    )
    species: List[str] = Field(
        description="Lista de especies predichas para cada pingüino"
    )
    num_predictions: int = Field(
        description="Número total de predicciones realizadas"
    )
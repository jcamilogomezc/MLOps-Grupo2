from pydantic import BaseModel, Field
from typing import Literal, List, Optional

# Los modelos pueden recibir datos nulos o faltantes ya que se incluyó imputadores en el pipeline del modelo
class PenguinFeatures(BaseModel):
    island: Optional[Literal['Biscoe', 'Dream', 'Torgersen']] = None
    bill_length_mm: Optional[float] = Field(None, ge=0)
    bill_depth_mm: Optional[float] = Field(None, ge=0)
    flipper_length_mm: Optional[float] = Field(None, ge=0)
    body_mass_g: Optional[float] = Field(None, ge=0)
    sex: Optional[Literal['male', 'female']] = None

class PredictRequest(BaseModel):
    model: str
    penguins: List[PenguinFeatures]

class PredictResponse(BaseModel):
    model: str
    species: List[str]
    num_predictions: int
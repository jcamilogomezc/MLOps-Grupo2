from pydantic import BaseModel, Field
from typing import Literal, List

class PenguinFeatures(BaseModel):
    bill_length_mm: float
    bill_depth_mm: float
    flipper_length_mm: float
    body_mass_g: float
    island: str
    sex: str

class PredictRequest(BaseModel):
    model_name: Literal["MLP", "HGB", "LogReg"] = Field(default="LogReg")
    items: List[PenguinFeatures]

class PredictResponse(BaseModel):
    model_name: str
    predictions: List[str]
    count: int
from pydantic import BaseModel, Field
from typing import Literal


class PredictionResult(BaseModel):
    filename: str
    predicted_class: Literal["benign", "malignant"]
    malignant_probability: float = Field(ge=0.0, le=1.0)
    threshold: float
    threshold_decision: str
    model_id: str


class BatchPredictionResponse(BaseModel):
    count: int
    results: list[PredictionResult]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_id: str

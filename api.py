"""
FastAPI REST Service for Student Performance Prediction.
Exposes endpoints to score individual students and batch records.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import pandas as pd
from src.predict import StudentPerformancePredictor

app = FastAPI(
    title="Education Analytics - Student Performance Prediction API",
    description="Predict student Pass/Fail probability scores based on attendance, study hours, and past academic records.",
    version="1.0.0",
)

predictor = StudentPerformancePredictor()


class StudentInput(BaseModel):
    attendance_rate: float = Field(..., ge=0.0, le=100.0, description="Attendance percentage (0-100%)", json_schema_extra={"example": 82.5})
    study_hours: float = Field(..., ge=0.0, le=80.0, description="Weekly study hours", json_schema_extra={"example": 14.0})
    past_academic_score: float = Field(..., ge=0.0, le=100.0, description="Prior academic mark / exam score (0-100)", json_schema_extra={"example": 70.0})
    model_type: Optional[str] = Field("best", description="Model: 'best', 'random_forest', or 'decision_tree'")
    threshold: Optional[float] = Field(0.50, ge=0.0, le=1.0, description="Decision threshold for Pass outcome")


class BatchStudentInput(BaseModel):
    students: List[StudentInput]


@app.get("/")
def root():
    return {
        "service": "Student Performance Prediction API",
        "status": "online",
        "supported_models": ["best", "random_forest", "decision_tree"],
        "core_features": ["attendance_rate", "study_hours", "past_academic_score"],
    }


@app.post("/predict")
def predict_student(payload: StudentInput):
    """Predict pass/fail probability for a single student."""
    try:
        result = predictor.predict_single(
            attendance_rate=payload.attendance_rate,
            study_hours=payload.study_hours,
            past_academic_score=payload.past_academic_score,
            model_type=payload.model_type or "best",
            threshold=payload.threshold or 0.50,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/compare")
def compare_models_endpoint(payload: StudentInput):
    """Compare Decision Tree vs Random Forest outputs and probabilities side-by-side."""
    try:
        comparison = predictor.compare_models(
            attendance_rate=payload.attendance_rate,
            study_hours=payload.study_hours,
            past_academic_score=payload.past_academic_score,
        )
        return comparison
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/metrics")
def get_model_metrics():
    """Retrieve training and test evaluation metrics."""
    return predictor.metadata

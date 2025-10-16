"""
API Request/Response Schemas for Crohn Flare Predictor

Pydantic models for validating API requests and formatting responses.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict
from datetime import datetime


class SymptomFeatures(BaseModel):
    """
    Daily symptom tracking features for a single user/date.

    All features must be provided in the exact order and names
    expected by the trained model.
    """
    # Core symptom aggregates
    severity_mean: float = Field(..., ge=0, le=4, description="Mean symptom severity (0-4)")
    severity_max: float = Field(..., ge=0, le=4, description="Maximum symptom severity (0-4)")
    severity_sum: float = Field(..., ge=0, description="Sum of all symptom severities")
    symptom_count: int = Field(..., ge=1, description="Number of symptoms tracked")

    # Individual symptom severities
    abdominal_pain: float = Field(..., ge=0, le=4, description="Abdominal pain severity (0-4)")
    blood_in_stool: float = Field(..., ge=0, le=4, description="Blood in stool severity (0-4)")
    diarrhea: float = Field(..., ge=0, le=4, description="Diarrhea severity (0-4)")
    fatigue: float = Field(..., ge=0, le=4, description="Fatigue severity (0-4)")
    fever: float = Field(..., ge=0, le=4, description="Fever severity (0-4)")
    joint_pain: float = Field(..., ge=0, le=4, description="Joint pain severity (0-4)")
    nausea: float = Field(..., ge=0, le=4, description="Nausea severity (0-4)")
    other: float = Field(..., ge=0, le=4, description="Other symptoms severity (0-4)")
    weight_loss: float = Field(..., ge=0, le=4, description="Weight loss severity (0-4)")

    # Baseline and change features
    severity_baseline_7d: float = Field(..., ge=0, le=4, description="Baseline severity (7-day avg)")
    severity_change_pct: float = Field(..., description="Percentage change from baseline")

    # Rolling window features (3-day)
    severity_mean_3d: float = Field(..., ge=0, le=4, description="3-day rolling mean severity")
    severity_max_3d: float = Field(..., ge=0, le=4, description="3-day rolling max severity")
    symptom_count_3d: float = Field(..., ge=0, description="3-day rolling symptom count")

    # Rolling window features (7-day)
    severity_mean_7d: float = Field(..., ge=0, le=4, description="7-day rolling mean severity")
    severity_max_7d: float = Field(..., ge=0, le=4, description="7-day rolling max severity")
    symptom_count_7d: float = Field(..., ge=0, description="7-day rolling symptom count")

    # Rolling window features (14-day)
    severity_mean_14d: float = Field(..., ge=0, le=4, description="14-day rolling mean severity")
    severity_max_14d: float = Field(..., ge=0, le=4, description="14-day rolling max severity")
    symptom_count_14d: float = Field(..., ge=0, description="14-day rolling symptom count")

    # Trend features
    severity_trend_3d: float = Field(..., description="3-day severity trend (slope)")
    severity_std_3d: float = Field(..., ge=0, description="3-day severity std deviation")
    severity_trend_7d: float = Field(..., description="7-day severity trend (slope)")
    severity_std_7d: float = Field(..., ge=0, description="7-day severity std deviation")

    # Symptom-specific rolling features
    diarrhea_7d: float = Field(..., ge=0, le=4, description="7-day avg diarrhea severity")
    abdominal_pain_7d: float = Field(..., ge=0, le=4, description="7-day avg abdominal pain severity")
    blood_in_stool_7d: float = Field(..., ge=0, le=4, description="7-day avg blood in stool severity")
    fatigue_7d: float = Field(..., ge=0, le=4, description="7-day avg fatigue severity")

    # Temporal features
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Monday, 6=Sunday)")
    day_of_month: int = Field(..., ge=1, le=31, description="Day of month (1-31)")
    month: int = Field(..., ge=1, le=12, description="Month (1-12)")
    is_weekend: int = Field(..., ge=0, le=1, description="Is weekend (0=No, 1=Yes)")
    days_since_start: int = Field(..., ge=1, description="Days since patient started tracking")

    # Lag features
    severity_mean_lag1: float = Field(..., ge=0, le=4, description="Previous day mean severity")
    is_flare_lag1: float = Field(..., ge=0, le=1, description="Previous day flare status (0 or 1)")
    severity_mean_lag2: float = Field(..., ge=0, le=4, description="2 days ago mean severity")
    is_flare_lag2: float = Field(..., ge=0, le=1, description="2 days ago flare status (0 or 1)")
    severity_mean_lag3: float = Field(..., ge=0, le=4, description="3 days ago mean severity")
    is_flare_lag3: float = Field(..., ge=0, le=1, description="3 days ago flare status (0 or 1)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "severity_mean": 1.5,
                "severity_max": 3.0,
                "severity_sum": 18.0,
                "symptom_count": 12,
                "abdominal_pain": 2.0,
                "blood_in_stool": 0.0,
                "diarrhea": 3.0,
                "fatigue": 2.0,
                "fever": 0.0,
                "joint_pain": 1.0,
                "nausea": 1.0,
                "other": 0.5,
                "weight_loss": 0.0,
                "severity_baseline_7d": 1.2,
                "severity_change_pct": 0.25,
                "severity_mean_3d": 1.4,
                "severity_max_3d": 3.0,
                "symptom_count_3d": 11.5,
                "severity_mean_7d": 1.3,
                "severity_max_7d": 3.0,
                "symptom_count_7d": 11.0,
                "severity_mean_14d": 1.2,
                "severity_max_14d": 3.0,
                "symptom_count_14d": 10.5,
                "severity_trend_3d": 0.15,
                "severity_std_3d": 0.5,
                "severity_trend_7d": 0.1,
                "severity_std_7d": 0.6,
                "diarrhea_7d": 2.5,
                "abdominal_pain_7d": 1.8,
                "blood_in_stool_7d": 0.1,
                "fatigue_7d": 1.9,
                "day_of_week": 2,
                "day_of_month": 15,
                "month": 6,
                "is_weekend": 0,
                "days_since_start": 120,
                "severity_mean_lag1": 1.3,
                "is_flare_lag1": 0.0,
                "severity_mean_lag2": 1.1,
                "is_flare_lag2": 0.0,
                "severity_mean_lag3": 1.0,
                "is_flare_lag3": 0.0
            }
        }
    }


class PredictionRequest(BaseModel):
    """Request for single prediction"""
    features: SymptomFeatures
    user_id: Optional[str] = Field(None, description="Optional user identifier for logging")
    threshold: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="Custom prediction threshold (default: model's optimal threshold)"
    )


class BatchPredictionRequest(BaseModel):
    """Request for batch predictions"""
    predictions: List[SymptomFeatures] = Field(..., min_length=1, max_length=1000)
    threshold: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="Custom prediction threshold (default: model's optimal threshold)"
    )


class PredictionResponse(BaseModel):
    """Response for single prediction"""
    is_flare: bool = Field(..., description="Predicted flare status (True/False)")
    flare_probability: float = Field(..., ge=0, le=1, description="Probability of flare (0-1)")
    confidence: str = Field(..., description="Confidence level (Low/Medium/High)")
    threshold_used: float = Field(..., description="Threshold used for classification")
    risk_level: str = Field(..., description="Risk level (Low/Medium/High/Critical)")
    recommendation: str = Field(..., description="Clinical recommendation")
    timestamp: datetime = Field(default_factory=datetime.now, description="Prediction timestamp")


class BatchPredictionResponse(BaseModel):
    """Response for batch predictions"""
    predictions: List[PredictionResponse]
    total_predictions: int
    flare_count: int
    average_flare_probability: float


class ModelInfo(BaseModel):
    """Model metadata and performance metrics"""
    model_type: str
    model_name: str
    n_features: int
    optimal_threshold_f2: float
    optimal_threshold_recall85: float
    test_recall: float
    test_precision: float
    test_f2: float
    test_pr_auc: float
    date_created: str
    features: List[str]


class HealthResponse(BaseModel):
    """API health check response"""
    status: str
    message: str
    timestamp: datetime = Field(default_factory=datetime.now)
    model_loaded: bool
    model_type: Optional[str] = None

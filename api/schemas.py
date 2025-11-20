"""
Pydantic schemas for API request/response validation.
"""
from datetime import date
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


# Health Check
class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    version: str = "1.0.0"


# Symptom tracking
class Symptoms(BaseModel):
    """Patient symptoms data."""
    abdominal_pain: int = Field(..., ge=0, le=10, description="Pain scale 0-10")
    diarrhea: int = Field(..., ge=0, le=10, description="Severity scale 0-10")
    fatigue: int = Field(..., ge=0, le=10, description="Fatigue scale 0-10")
    fever: bool = Field(..., description="Presence of fever")
    weight_change: Optional[float] = Field(default=0.0, description="Weight change in kg (negative = loss)")
    blood_in_stool: Optional[bool] = Field(default=False, description="Blood in stool")
    nausea: Optional[int] = Field(default=0, ge=0, le=10, description="Nausea scale 0-10")


class Demographics(BaseModel):
    """Patient demographic information."""
    age: int = Field(..., ge=0, le=120, description="Patient age")
    gender: str = Field(..., pattern="^[MFO]$", description="M/F/O")
    disease_duration_years: float = Field(..., ge=0, description="Years since diagnosis")
    bmi: Optional[float] = Field(default=None, ge=10, le=60, description="Body Mass Index")

    # IBD Type and Classification
    ibd_type: Optional[str] = Field(
        default="crohn",
        description="Type of IBD: 'crohn' or 'ulcerative_colitis'",
        pattern="^(crohn|ulcerative_colitis)$"
    )
    montreal_location: Optional[str] = Field(
        default=None,
        description="Montreal location for Crohn (L1/L2/L3/L4) or extent for UC (E1/E2/E3)",
        pattern="^(L[1-4]|E[1-3])$"
    )

    @field_validator("montreal_location")
    @classmethod
    def validate_montreal_for_ibd_type(cls, v: Optional[str], info) -> Optional[str]:
        """Validate Montreal classification matches IBD type."""
        if v is None:
            return v

        ibd_type = info.data.get('ibd_type', 'crohn')

        # Crohn should have L classification
        if ibd_type == 'crohn' and v and not v.startswith('L'):
            raise ValueError("Crohn disease should use L classification (L1/L2/L3/L4)")

        # UC should have E classification
        if ibd_type == 'ulcerative_colitis' and v and not v.startswith('E'):
            raise ValueError("Ulcerative colitis should use E classification (E1/E2/E3)")

        return v


class MedicalHistory(BaseModel):
    """Patient medical history."""
    previous_flares: int = Field(..., ge=0, description="Number of previous flares")
    medications: Optional[List[str]] = Field(default=[], description="Current medications")
    last_flare_days_ago: int = Field(..., ge=0, description="Days since last flare")
    surgery_history: Optional[bool] = Field(default=False, description="Previous IBD surgery")
    smoking_status: Optional[str] = Field(default="never", description="never/former/current")


# Prediction requests
class PredictionRequest(BaseModel):
    """Single prediction request."""
    symptoms: Symptoms
    demographics: Demographics
    history: MedicalHistory


class ClusterInfo(BaseModel):
    """Cluster assignment information."""
    cluster_id: int = Field(..., ge=0, le=2, description="Patient phenotype cluster (0-2)")
    cluster_confidence: float = Field(..., ge=0, le=1, description="Confidence in cluster assignment")
    model_source: str = Field(..., description="cluster_specific / global_fallback / rule_based")
    cluster_description: Optional[str] = Field(default=None, description="Human-readable cluster description")

    @field_validator("model_source")
    @classmethod
    def validate_model_source(cls, v: str) -> str:
        if v not in ["cluster_specific", "global_fallback", "rule_based"]:
            raise ValueError("model_source must be cluster_specific, global_fallback, or rule_based")
        return v


class IBDInfo(BaseModel):
    """IBD type and classification information."""
    ibd_type: str = Field(..., description="crohn / ulcerative_colitis")
    montreal_classification: Optional[str] = Field(default=None, description="Montreal code (L1-L4 for Crohn, E1-E3 for UC)")


class PredictionMetadata(BaseModel):
    """Metadata about the prediction."""
    prediction_timestamp: str = Field(..., description="ISO timestamp of prediction")
    model_version: str = Field(default="2.0.0", description="Model version")
    api_version: str = Field(default="1.0.0", description="API version")


class PredictionResponse(BaseModel):
    """Prediction result."""
    prediction: "FlareRiskPrediction"
    factors: "ContributingFactors"
    recommendation: str
    metadata: PredictionMetadata


class FlareRiskPrediction(BaseModel):
    """Flare risk prediction details."""
    flare_risk: str = Field(..., description="low/medium/high")
    probability: float = Field(..., ge=0, le=1, description="Probability of the predicted class")
    confidence: float = Field(..., ge=0, le=1, description="Confidence gap (difference between top 2 classes)")
    probabilities: Optional[dict] = Field(default=None, description="Probability distribution for all classes")
    cluster_info: Optional[ClusterInfo] = Field(default=None, description="Cluster assignment details (if using cluster-stratified model)")
    ibd_info: Optional[IBDInfo] = Field(default=None, description="IBD type and classification")

    # Legacy fields (deprecated but kept for backwards compatibility)
    cluster_id: Optional[int] = Field(default=None, description="DEPRECATED: Use cluster_info.cluster_id")
    cluster_confidence: Optional[float] = Field(default=None, description="DEPRECATED: Use cluster_info.cluster_confidence")

    @field_validator("flare_risk")
    @classmethod
    def validate_risk_level(cls, v: str) -> str:
        if v not in ["low", "medium", "high"]:
            raise ValueError("flare_risk must be low, medium, or high")
        return v


class ContributingFactors(BaseModel):
    """Factors contributing to the prediction."""
    top_contributors: List[str] = Field(..., description="Top contributing features")
    symptom_severity_score: Optional[float] = Field(default=None, ge=0, le=1)
    trend_indicator: Optional[str] = Field(default=None)


# Batch prediction
class PatientData(BaseModel):
    """Patient data for batch prediction."""
    patient_id: str
    symptoms: Symptoms
    demographics: Demographics
    history: Optional[MedicalHistory] = None


class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""
    patients: List[PatientData] = Field(..., min_length=1, max_length=100)


class PatientPredictionResult(BaseModel):
    """Individual patient prediction result."""
    patient_id: str
    prediction: FlareRiskPrediction
    factors: ContributingFactors


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""
    results: List[PatientPredictionResult]
    processed_count: int
    failed_count: int = 0
    errors: Optional[List[str]] = None


# Trend analysis
class DailySymptomRecord(BaseModel):
    """Daily symptom record."""
    date: date
    symptoms: Symptoms


class TrendAnalysisRequest(BaseModel):
    """Trend analysis request."""
    patient_id: str
    daily_records: List[DailySymptomRecord] = Field(..., min_length=7)
    window_days: int = Field(default=14, ge=7, le=90)


class TrendAnalysisResponse(BaseModel):
    """Trend analysis response."""
    patient_id: str
    analysis_period: "AnalysisPeriod"
    trends: "SymptomTrends"
    risk_assessment: FlareRiskPrediction
    recommendations: List[str]


class AnalysisPeriod(BaseModel):
    """Analysis time period."""
    start_date: date
    end_date: date
    days_analyzed: int


class SymptomTrends(BaseModel):
    """Symptom trend indicators."""
    overall_trend: str = Field(..., description="improving/stable/worsening")
    severity_change: float = Field(..., description="Change in severity score")
    concerning_patterns: List[str]
    symptom_correlations: Optional[dict] = None


# Model information
class ModelMetrics(BaseModel):
    """Model performance metrics."""
    accuracy: float = Field(..., ge=0, le=1)
    precision: float = Field(..., ge=0, le=1)
    recall: float = Field(..., ge=0, le=1)
    f1_score: float = Field(..., ge=0, le=1)
    roc_auc: Optional[float] = Field(default=None, ge=0, le=1)


class ModelInfoResponse(BaseModel):
    """Model information response."""
    model_version: str
    trained_date: date
    metrics: ModelMetrics
    features_count: int
    training_samples: Optional[int] = None
    model_type: str = "RandomForest"

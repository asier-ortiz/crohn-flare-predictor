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
    """
    Patient symptoms data for daily tracking.

    All severity fields use a 0-10 scale where:
    - 0 = No symptom
    - 1-3 = Mild
    - 4-6 = Moderate
    - 7-10 = Severe
    """
    abdominal_pain: int = Field(..., ge=0, le=10, description="Abdominal pain severity (0-10)", examples=[7])
    diarrhea: int = Field(..., ge=0, le=10, description="Diarrhea frequency/severity (0-10)", examples=[6])
    fatigue: int = Field(..., ge=0, le=10, description="Fatigue level (0-10)", examples=[5])
    fever: bool = Field(..., description="Presence of fever (>38°C)", examples=[False])
    weight_change: Optional[float] = Field(default=0.0, description="Weight change in kg since last measurement (negative = loss)", examples=[-1.5])
    blood_in_stool: Optional[bool] = Field(default=False, description="Visible blood in stool", examples=[False])
    nausea: Optional[int] = Field(default=0, ge=0, le=10, description="Nausea severity (0-10)", examples=[3])

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "abdominal_pain": 7,
                    "diarrhea": 6,
                    "fatigue": 5,
                    "fever": False,
                    "weight_change": -1.5,
                    "blood_in_stool": False,
                    "nausea": 3
                }
            ]
        }
    }


class Demographics(BaseModel):
    """
    Patient demographic and disease classification information.

    Montreal Classification:
    - Crohn's Disease: L1 (ileal), L2 (colonic), L3 (ileocolonic), L4 (upper GI)
    - Ulcerative Colitis: E1 (proctitis), E2 (left-sided), E3 (extensive)
    """
    age: int = Field(..., ge=0, le=120, description="Patient age in years", examples=[32])
    gender: str = Field(..., pattern="^[MFO]$", description="Gender: M (Male), F (Female), O (Other)", examples=["F"])
    disease_duration_years: float = Field(..., ge=0, description="Years since IBD diagnosis", examples=[5.0])
    bmi: Optional[float] = Field(default=None, ge=10, le=60, description="Body Mass Index (kg/m²)", examples=[22.5])

    # IBD Type and Classification
    ibd_type: Optional[str] = Field(
        default="crohn",
        description="Type of inflammatory bowel disease: 'crohn' or 'ulcerative_colitis'",
        pattern="^(crohn|ulcerative_colitis)$",
        examples=["crohn"]
    )
    montreal_location: Optional[str] = Field(
        default=None,
        description="Montreal classification - Location for Crohn (L1/L2/L3/L4) or extent for UC (E1/E2/E3)",
        pattern="^(L[1-4]|E[1-3])$",
        examples=["L3"]
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "age": 32,
                    "gender": "F",
                    "disease_duration_years": 5.0,
                    "bmi": 22.5,
                    "ibd_type": "crohn",
                    "montreal_location": "L3"
                }
            ]
        }
    }

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
    """
    Patient medical history relevant to IBD management.

    This information is critical for accurate flare prediction as past
    disease activity strongly predicts future risk.
    """
    previous_flares: int = Field(..., ge=0, description="Total number of documented previous flares", examples=[3])
    medications: Optional[List[str]] = Field(default=[], description="Current medications (e.g., mesalamine, prednisone)", examples=[["mesalamine", "azathioprine"]])
    last_flare_days_ago: int = Field(..., ge=0, description="Days elapsed since most recent flare", examples=[120])
    surgery_history: Optional[bool] = Field(default=False, description="Has had previous IBD-related surgery", examples=[False])
    smoking_status: Optional[str] = Field(default="never", description="Smoking status: never, former, or current", examples=["never"])

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "previous_flares": 3,
                    "medications": ["mesalamine", "azathioprine"],
                    "last_flare_days_ago": 120,
                    "surgery_history": False,
                    "smoking_status": "never"
                }
            ]
        }
    }


class TemporalFeatures(BaseModel):
    """
    Temporal features calculated from user's symptom history.

    **Optional but recommended** - These features significantly improve prediction accuracy
    when the web app has stored historical symptom data (7+ days).

    If not provided, the API will use fallback values based on current symptoms only,
    which may result in less accurate predictions.
    """
    pain_trend_7d: Optional[float] = Field(default=None, ge=0, le=1, description="7-day normalized pain trend (0=improving, 1=worsening)", examples=[0.15])
    diarrhea_trend_7d: Optional[float] = Field(default=None, ge=0, le=1, description="7-day normalized diarrhea trend (0=improving, 1=worsening)", examples=[0.10])
    fatigue_trend_7d: Optional[float] = Field(default=None, ge=0, le=1, description="7-day normalized fatigue trend (0=improving, 1=worsening)", examples=[0.05])
    symptom_volatility_7d: Optional[float] = Field(default=None, ge=0, description="7-day symptom volatility (standard deviation)", examples=[1.2])
    symptom_change_rate: Optional[float] = Field(default=None, description="Rate of symptom change vs 7 days ago", examples=[0.08])
    days_since_low_symptoms: Optional[int] = Field(default=None, ge=0, description="Consecutive days with elevated symptoms", examples=[5])

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "pain_trend_7d": 0.15,
                    "diarrhea_trend_7d": 0.10,
                    "fatigue_trend_7d": 0.05,
                    "symptom_volatility_7d": 1.2,
                    "symptom_change_rate": 0.08,
                    "days_since_low_symptoms": 5
                }
            ]
        }
    }


# Prediction requests
class PredictionRequest(BaseModel):
    """
    Complete prediction request for a single patient.

    This is the main request format for the /predict endpoint. Include all
    current symptoms, patient demographics, medical history, and optionally
    temporal features if you have 7+ days of historical symptom data.
    """
    symptoms: Symptoms
    demographics: Demographics
    history: MedicalHistory
    temporal_features: Optional[TemporalFeatures] = Field(
        default=None,
        description="Optional temporal features calculated from historical data. If omitted, API uses fallback values."
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "symptoms": {
                        "abdominal_pain": 7,
                        "diarrhea": 6,
                        "fatigue": 5,
                        "fever": False,
                        "weight_change": -1.5,
                        "blood_in_stool": False,
                        "nausea": 3
                    },
                    "demographics": {
                        "age": 32,
                        "gender": "F",
                        "disease_duration_years": 5.0,
                        "bmi": 22.5,
                        "ibd_type": "crohn",
                        "montreal_location": "L3"
                    },
                    "history": {
                        "previous_flares": 3,
                        "medications": ["mesalamine", "azathioprine"],
                        "last_flare_days_ago": 120,
                        "surgery_history": False,
                        "smoking_status": "never"
                    },
                    "temporal_features": {
                        "pain_trend_7d": 0.15,
                        "diarrhea_trend_7d": 0.10,
                        "fatigue_trend_7d": 0.05,
                        "symptom_volatility_7d": 1.2,
                        "symptom_change_rate": 0.08,
                        "days_since_low_symptoms": 5
                    }
                }
            ]
        }
    }


class ClusterInfo(BaseModel):
    """
    Cluster assignment information for cluster-stratified models.

    Patients are grouped into phenotype clusters based on disease location,
    behavior, and symptom patterns. Each cluster uses a specialized ML model
    trained on similar patients for more accurate predictions.
    """
    cluster_id: int = Field(..., ge=0, le=2, description="Patient phenotype cluster (0-2)", examples=[1])
    cluster_confidence: float = Field(..., ge=0, le=1, description="Confidence in cluster assignment (0-1)", examples=[0.92])
    model_source: str = Field(..., description="Model type used: cluster_specific, global_fallback, or rule_based", examples=["cluster_specific"])
    cluster_description: Optional[str] = Field(default=None, description="Human-readable cluster description", examples=["Ileocolonic disease with moderate symptoms"])

    @field_validator("model_source")
    @classmethod
    def validate_model_source(cls, v: str) -> str:
        if v not in ["cluster_specific", "global_fallback", "rule_based"]:
            raise ValueError("model_source must be cluster_specific, global_fallback, or rule_based")
        return v


class IBDInfo(BaseModel):
    """
    IBD type and classification information.

    Includes the patient's disease type and Montreal classification,
    which are key factors in phenotype clustering and risk prediction.
    """
    ibd_type: str = Field(..., description="Type of IBD: crohn or ulcerative_colitis", examples=["crohn"])
    montreal_classification: Optional[str] = Field(default=None, description="Montreal code (L1-L4 for Crohn, E1-E3 for UC)", examples=["L3"])


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
    """
    Flare risk prediction details.

    This is the core prediction output, including:
    - Risk level (low/medium/high)
    - Probability of the predicted risk level
    - Confidence score (how certain the model is)
    - Full probability distribution across all risk levels
    - Cluster and IBD classification information
    """
    flare_risk: str = Field(..., description="Predicted risk level: low, medium, or high", examples=["medium"])
    probability: float = Field(..., ge=0, le=1, description="Probability of the predicted risk class (0-1)", examples=[0.65])
    confidence: float = Field(..., ge=0, le=1, description="Model confidence - gap between top 2 classes (0-1)", examples=[0.82])
    probabilities: Optional[dict] = Field(default=None, description="Full probability distribution for all risk levels", examples=[{"low": 0.15, "medium": 0.65, "high": 0.20}])
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

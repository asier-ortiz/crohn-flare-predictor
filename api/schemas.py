"""
Pydantic schemas for API request/response validation.
"""
from datetime import date
from typing import List, Optional
from enum import Enum
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
    medications: Optional[List[str]] = Field(default=[], description="Current medications (accepts Spanish or English names)", examples=[["mesalamine", "azathioprine"]])
    last_flare_days_ago: int = Field(..., ge=0, description="Days elapsed since most recent flare", examples=[120])
    surgery_history: Optional[bool] = Field(default=False, description="Has had previous IBD-related surgery", examples=[False])
    smoking_status: Optional[str] = Field(default="never", description="Smoking status: never, former, or current", examples=["never"])
    cumulative_flare_days: Optional[int] = Field(default=0, ge=0, description="Total days in flare state over disease history", examples=[45])

    @field_validator("medications", mode="before")
    @classmethod
    def normalize_medication_names(cls, v):
        """Normalize medication names from Spanish to English."""
        if not v:
            return []

        # Spanish to English medication mapping
        # Includes generic names (Spanish), trade names, and common variants
        medication_map = {
            # ========== AMINOSALICYLATES ==========
            # Generic names (Spanish)
            "mesalazina": "mesalamine",
            "mesalamina": "mesalamine",
            "sulfasalazina": "sulfasalazine",
            "balsalazida": "balsalazide",
            # Trade names
            "asacol": "mesalamine",
            "pentasa": "mesalamine",
            "lialda": "mesalamine",
            "apriso": "mesalamine",
            "delzicol": "mesalamine",
            "salofalk": "mesalamine",
            "claversal": "mesalamine",
            "azulfidine": "sulfasalazine",
            "azulfidina": "sulfasalazine",
            "colazal": "balsalazide",

            # ========== CORTICOSTEROIDS ==========
            # Generic names (Spanish)
            "prednisona": "prednisone",
            "prednisolona": "prednisolone",
            "budesonida": "budesonide",
            "hidrocortisona": "hydrocortisone",
            "metilprednisolona": "methylprednisolone",
            # Trade names
            "entocort": "budesonide",
            "uceris": "budesonide",
            "cortiment": "budesonide",
            "solu-medrol": "methylprednisolone",

            # ========== IMMUNOSUPPRESSANTS ==========
            # Generic names (Spanish)
            "azatioprina": "azathioprine",
            "mercaptopurina": "mercaptopurine",
            "6-mercaptopurina": "mercaptopurine",
            "metotrexato": "methotrexate",
            "metotrexate": "methotrexate",
            "ciclosporina": "cyclosporine",
            "tacrolimus": "tacrolimus",
            "tacrolimús": "tacrolimus",
            # Trade names
            "imuran": "azathioprine",
            "azasan": "azathioprine",
            "purinethol": "mercaptopurine",
            "purinetol": "mercaptopurine",
            "rheumatrex": "methotrexate",
            "trexall": "methotrexate",
            "sandimmune": "cyclosporine",
            "neoral": "cyclosporine",
            "prograf": "tacrolimus",

            # ========== BIOLOGICS (anti-TNF) ==========
            # Generic names
            "infliximab": "infliximab",
            "adalimumab": "adalimumab",
            "golimumab": "golimumab",
            "certolizumab": "certolizumab",
            # Trade names
            "remicade": "infliximab",
            "remsima": "infliximab",
            "inflectra": "infliximab",
            "humira": "adalimumab",
            "simponi": "golimumab",
            "cimzia": "certolizumab",

            # ========== BIOLOGICS (anti-integrin) ==========
            # Generic names
            "vedolizumab": "vedolizumab",
            "natalizumab": "natalizumab",
            # Trade names
            "entyvio": "vedolizumab",
            "tysabri": "natalizumab",

            # ========== BIOLOGICS (anti-IL) ==========
            # Generic names
            "ustekinumab": "ustekinumab",
            "risankizumab": "risankizumab",
            # Trade names
            "stelara": "ustekinumab",
            "skyrizi": "risankizumab",

            # ========== JAK INHIBITORS ==========
            # Generic names
            "tofacitinib": "tofacitinib",
            "upadacitinib": "upadacitinib",
            # Trade names
            "xeljanz": "tofacitinib",
            "rinvoq": "upadacitinib",
        }

        normalized = []
        for med in v:
            # Convert to lowercase for matching
            med_lower = med.lower().strip()
            # Use mapped name if available, otherwise keep original
            normalized_name = medication_map.get(med_lower, med_lower)
            normalized.append(normalized_name)

        return normalized

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "previous_flares": 3,
                    "medications": ["mesalazina", "azatioprina"],
                    "last_flare_days_ago": 120,
                    "surgery_history": False,
                    "smoking_status": "never",
                    "cumulative_flare_days": 45
                }
            ]
        }
    }


class TemporalFeatures(BaseModel):
    """
    **DEPRECATED** - These features are now calculated automatically by the API.

    Temporal features calculated from user's symptom history.
    This class is kept for backwards compatibility but is no longer used.
    Use daily_records in PredictionRequest instead.
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


# Exercise level enum
class ExerciseLevel(str, Enum):
    """Exercise intensity levels."""
    NONE = "none"
    MODERATE = "moderate"
    HIGH = "high"


# Daily symptom record (must be defined before PredictionRequest)
class DailySymptomRecord(BaseModel):
    """
    Daily symptom record with optional food and exercise tracking.

    Foods are tracked as free text (e.g., "pizza con queso", "ensalada césar")
    and automatically categorized by the API into food groups for correlation analysis.

    Exercise is tracked as a simple level (none, moderate, high) to analyze
    its impact on symptom severity.
    """
    date: date
    symptoms: Symptoms

    # Food tracking (free text, categorized by API)
    foods: Optional[List[str]] = Field(
        default=None,
        description="List of foods consumed (free text, will be categorized automatically)",
        examples=[["pizza con queso", "café con leche", "ensalada"]]
    )

    # Exercise tracking (simple enum)
    exercise: Optional[ExerciseLevel] = Field(
        default=ExerciseLevel.NONE,
        description="Exercise level for the day: none, moderate, or high",
        examples=["moderate"]
    )


# Prediction requests
class PredictionRequest(BaseModel):
    """
    Complete prediction request for a single patient.

    This is the main request format for the /predict endpoint. Provide daily symptom
    records (minimum 1 day for current symptoms, recommended 7-14 days for improved
    accuracy through temporal feature calculation).

    ## How It Works

    - **Day 1 (no history)**: Send 1 daily record → API uses fallback → basic prediction
    - **Day 7+ (with history)**: Send 7-14 daily records → API calculates temporal features → improved prediction

    The API automatically calculates temporal features (trends, volatility, change rates)
    from the daily records when 7+ days are provided, eliminating the need for external
    feature calculation.
    """
    daily_records: List[DailySymptomRecord] = Field(
        ...,
        min_length=1,
        description="Daily symptom records. Minimum 1 (today), recommended 7-14 for temporal analysis"
    )
    demographics: Demographics
    history: MedicalHistory

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "daily_records": [
                        {
                            "date": "2025-11-15",
                            "symptoms": {
                                "abdominal_pain": 3,
                                "diarrhea": 2,
                                "fatigue": 2,
                                "fever": False,
                                "weight_change": 0.0,
                                "blood_in_stool": False,
                                "nausea": 1
                            },
                            "foods": ["ensalada", "pollo a la plancha", "arroz blanco"],
                            "exercise": "moderate"
                        },
                        {
                            "date": "2025-11-16",
                            "symptoms": {
                                "abdominal_pain": 4,
                                "diarrhea": 3,
                                "fatigue": 3,
                                "fever": False,
                                "weight_change": -0.2,
                                "blood_in_stool": False,
                                "nausea": 2
                            },
                            "foods": ["café con leche", "tostadas", "verduras al vapor"],
                            "exercise": "moderate"
                        },
                        {
                            "date": "2025-11-17",
                            "symptoms": {
                                "abdominal_pain": 4,
                                "diarrhea": 4,
                                "fatigue": 3,
                                "fever": False,
                                "weight_change": -0.3,
                                "blood_in_stool": False,
                                "nausea": 2
                            },
                            "foods": ["pizza con queso", "café con leche"],
                            "exercise": "none"
                        },
                        {
                            "date": "2025-11-18",
                            "symptoms": {
                                "abdominal_pain": 5,
                                "diarrhea": 5,
                                "fatigue": 4,
                                "fever": False,
                                "weight_change": -0.5,
                                "blood_in_stool": False,
                                "nausea": 3
                            },
                            "foods": ["pasta carbonara", "helado"],
                            "exercise": "none"
                        },
                        {
                            "date": "2025-11-19",
                            "symptoms": {
                                "abdominal_pain": 6,
                                "diarrhea": 5,
                                "fatigue": 5,
                                "fever": False,
                                "weight_change": -0.4,
                                "blood_in_stool": False,
                                "nausea": 3
                            },
                            "foods": ["hamburguesa con queso", "patatas fritas", "café"],
                            "exercise": "none"
                        },
                        {
                            "date": "2025-11-20",
                            "symptoms": {
                                "abdominal_pain": 6,
                                "diarrhea": 6,
                                "fatigue": 5,
                                "fever": False,
                                "weight_change": -0.6,
                                "blood_in_stool": True,
                                "nausea": 4
                            },
                            "foods": ["pizza cuatro quesos", "comida picante"],
                            "exercise": "none"
                        },
                        {
                            "date": "2025-11-21",
                            "symptoms": {
                                "abdominal_pain": 7,
                                "diarrhea": 6,
                                "fatigue": 6,
                                "fever": False,
                                "weight_change": -0.8,
                                "blood_in_stool": True,
                                "nausea": 4
                            },
                            "foods": ["pan con mantequilla", "yogur", "café"],
                            "exercise": "none"
                        }
                    ],
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
                        "medications": ["mesalazina", "azatioprina"],
                        "last_flare_days_ago": 120,
                        "surgery_history": False,
                        "smoking_status": "never",
                        "cumulative_flare_days": 45
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
    correlation_insights: Optional[List[str]] = Field(default=None, description="Human-readable interpretation of symptom correlations")


class FoodInsight(BaseModel):
    """Insight about a specific food category."""
    correlation: float = Field(..., description="Correlation coefficient with symptom severity (-1 to 1)")
    occurrences: int = Field(..., description="Number of days this food was consumed")
    insight: str = Field(..., description="Human-readable interpretation")


class ExerciseInsight(BaseModel):
    """Insight about exercise impact."""
    correlation: float = Field(..., description="Correlation coefficient with symptom severity (-1 to 1)")
    days_with_exercise: int = Field(..., description="Number of days with exercise")
    average_severity_with: float = Field(..., description="Average symptom severity on exercise days")
    average_severity_without: float = Field(..., description="Average symptom severity on non-exercise days")
    insight: str = Field(..., description="Human-readable interpretation")


class LifestyleInsights(BaseModel):
    """
    Lifestyle insights from food and exercise tracking.

    Analyzes correlations between food categories, exercise levels, and symptom severity
    to provide personalized recommendations. Only available when 7+ days of data with
    food/exercise tracking are provided.
    """
    trigger_foods: Optional[dict[str, FoodInsight]] = Field(
        default=None,
        description="Food categories that correlate with increased symptoms (correlation > 0.5)"
    )
    beneficial_foods: Optional[dict[str, FoodInsight]] = Field(
        default=None,
        description="Food categories that correlate with decreased symptoms (correlation < -0.5)"
    )
    exercise_impact: Optional[ExerciseInsight] = Field(
        default=None,
        description="Impact of exercise on symptom severity"
    )
    recommendations: Optional[List[str]] = Field(
        default=None,
        description="Actionable recommendations based on food and exercise patterns"
    )


class PredictionResponse(BaseModel):
    """
    Prediction result with optional trend analysis.

    When 7+ days of data are provided, includes temporal trend analysis.
    With fewer days, only basic prediction is returned.
    """
    prediction: "FlareRiskPrediction"
    factors: "ContributingFactors"
    recommendation: str
    metadata: PredictionMetadata

    # Optional fields (only present when 7+ days of data)
    trends: Optional["SymptomTrends"] = Field(
        default=None,
        description="Temporal trend analysis (only when 7+ days of data)"
    )
    analysis_period: Optional["AnalysisPeriod"] = Field(
        default=None,
        description="Time period analyzed for trends (only when 7+ days of data)"
    )
    lifestyle_insights: Optional["LifestyleInsights"] = Field(
        default=None,
        description="Food and exercise insights (only when 7+ days with food/exercise tracking)"
    )


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


# Trend analysis
class TrendAnalysisRequest(BaseModel):
    """
    Trend analysis request for temporal symptom patterns.

    Requires minimum 7 days of symptom history to analyze trends and patterns.
    Recommended 14-30 days for more reliable trend detection.
    """
    patient_id: str
    daily_records: List[DailySymptomRecord] = Field(..., min_length=7)
    demographics: Demographics
    history: MedicalHistory
    window_days: int = Field(default=14, ge=7, le=90, description="Analysis window in days")


class TrendAnalysisResponse(BaseModel):
    """
    DEPRECATED: Use /predict endpoint instead.

    Trend analysis response. This schema is kept for backward compatibility
    but the /predict/trends endpoint has been merged into /predict.
    """
    patient_id: str
    analysis_period: AnalysisPeriod
    trends: SymptomTrends
    risk_assessment: FlareRiskPrediction
    recommendations: List[str]


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

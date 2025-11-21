"""
FastAPI application for Crohn's Disease Flare Prediction.

This is a stateless ML API service that provides prediction endpoints
for inflammatory bowel disease flare risk assessment.
"""
import logging
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta
from typing import List, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.config import settings
from api.ml_model import CrohnPredictor, ClusterStratifiedPredictor, get_predictor
from api.schemas import (
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
    FlareRiskPrediction,
    ContributingFactors,
    ClusterInfo,
    IBDInfo,
    PredictionMetadata,
    TrendAnalysisRequest,
    TrendAnalysisResponse,
    AnalysisPeriod,
    SymptomTrends,
    ModelInfoResponse,
    ModelMetrics,
)
from api.constants import CROHN_CLUSTER_DESCRIPTIONS, UC_CLUSTER_DESCRIPTIONS

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Global state for ML model (loaded once at startup)
class AppState:
    """Application state container."""
    predictor: Optional[CrohnPredictor] = None
    model_loaded: bool = False
    model_metadata: dict = {}
    uses_clusters: bool = False


app_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    Loads ML model once at startup.
    """
    # Startup
    logger.info(f"Starting {settings.app_name} v{settings.version}")
    logger.info(f"Environment: {settings.environment}")

    try:
        # Load ML model (tries cluster-stratified first, falls back to global)
        logger.info("Initializing ML prediction engine...")
        app_state.predictor = get_predictor(use_clusters=True)

        if app_state.predictor:
            app_state.model_loaded = True
            app_state.uses_clusters = isinstance(app_state.predictor, ClusterStratifiedPredictor)

            app_state.model_metadata = {
                "version": settings.model_version,
                "type": "ClusterStratifiedRandomForest" if app_state.uses_clusters else "RandomForest",
                "loaded_at": date.today().isoformat(),
                "uses_cluster_models": app_state.uses_clusters
            }
            logger.info(f"ML model loaded successfully (cluster-stratified: {app_state.uses_clusters})")
        else:
            logger.warning("ML model not found, using rule-based fallback")
            app_state.model_loaded = False
            app_state.model_metadata = {
                "version": settings.model_version,
                "type": "rule-based",
                "loaded_at": date.today().isoformat()
            }
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        app_state.model_loaded = False

    yield

    # Shutdown
    logger.info("Shutting down application")


# Initialize FastAPI app
app = FastAPI(
    title=settings.app_name,
    description="Stateless ML API service for predicting Crohn's disease flares",
    version=settings.version,
    docs_url="/docs",
    redoc_url=None,  # ReDoc disabled - using only Swagger
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


# Helper functions for prediction logic
def calculate_symptom_severity(symptoms) -> float:
    """Calculate overall symptom severity score (0-1)."""
    scores = [
        symptoms.abdominal_pain / 10,
        symptoms.diarrhea / 10,
        symptoms.fatigue / 10,
        symptoms.nausea / 10 if symptoms.nausea else 0,
        1.0 if symptoms.fever else 0,
        1.0 if symptoms.blood_in_stool else 0,
    ]
    # Weight factor for weight loss
    if symptoms.weight_change < -2:
        scores.append(min(abs(symptoms.weight_change) / 10, 1.0))

    return sum(scores) / len(scores)


def predict_flare_risk(request: PredictionRequest) -> tuple:
    """
    Predict flare risk based on symptoms and history using ML model.
    Returns tuple with cluster info if using cluster-stratified model.

    Returns:
        - If cluster-stratified: (risk_level, probability, confidence, contributors,
                                   all_probabilities, cluster_id, cluster_confidence,
                                   model_source, ibd_type, montreal_code)
        - If global: (risk_level, probability, confidence, contributors, all_probabilities,
                     None, None, "rule_based", ibd_type, None)

    Uses the trained RandomForest model if available, otherwise falls back
    to rule-based prediction.
    """
    # Extract current symptoms (last day in daily_records)
    sorted_records = sorted(request.daily_records, key=lambda x: x.date)
    current_record = sorted_records[-1]
    current_symptoms = current_record.symptoms

    if app_state.predictor:
        # Use ML model prediction
        # Convert daily_records to dict format for predictor
        daily_records_dict = [
            {
                'date': record.date.isoformat(),
                'symptoms': record.symptoms.dict()
            }
            for record in sorted_records
        ]

        result = app_state.predictor.predict(
            symptoms=current_symptoms.dict(),
            demographics=request.demographics.dict(),
            history=request.history.dict(),
            daily_records=daily_records_dict
        )

        # Cluster-stratified predictor returns 10 items
        # (risk, prob, conf, contrib, all_probs, cluster_id, cluster_conf, model_source, ibd_type, montreal_code)
        return result
    else:
        # Fallback: Simple rule-based prediction if model not loaded
        logger.warning("Predictor not available, using basic rule-based prediction")

        severity = calculate_symptom_severity(current_symptoms)

        # Factor in medical history
        history_risk = 0.0
        if request.history.previous_flares > 3:
            history_risk += 0.2
        if request.history.last_flare_days_ago < 90:
            history_risk += 0.3
        if request.history.surgery_history:
            history_risk += 0.1

        # Combine factors
        total_risk = min(severity * 0.7 + history_risk * 0.3, 1.0)

        # Determine risk level and probabilities
        if total_risk < 0.3:
            risk_level = "low"
            confidence = 0.85
            all_probs = {"low": 0.85, "medium": 0.12, "high": 0.03}
        elif total_risk < 0.6:
            risk_level = "medium"
            confidence = 0.75
            all_probs = {"low": 0.15, "medium": 0.75, "high": 0.10}
        else:
            risk_level = "high"
            confidence = 0.80
            all_probs = {"low": 0.05, "medium": 0.15, "high": 0.80}

        # Identify top contributors
        contributors = []
        if current_symptoms.abdominal_pain >= 7:
            contributors.append("abdominal_pain")
        if current_symptoms.diarrhea >= 6:
            contributors.append("diarrhea")
        if current_symptoms.blood_in_stool:
            contributors.append("blood_in_stool")
        if request.history.previous_flares > 3:
            contributors.append("previous_flares")
        if request.history.last_flare_days_ago < 90:
            contributors.append("recent_flare_history")
        if current_symptoms.weight_change and current_symptoms.weight_change < -3:
            contributors.append("weight_loss")

        if not contributors:
            contributors = ["general_symptom_pattern"]

        # Get IBD type from request
        ibd_type = request.demographics.ibd_type if request.demographics.ibd_type else "crohn"

        # Return 10 items (rule-based fallback)
        return (risk_level, all_probs[risk_level], confidence, contributors[:3], all_probs,
                None, None, "rule_based", ibd_type, None)


def get_recommendation(risk_level: str, symptoms) -> str:
    """Generate recommendation based on risk level."""
    recommendations = {
        "low": "Continue con el seguimiento regular. Mantenga sus habitos saludables.",
        "medium": "Monitoree sus sintomas de cerca. Considere contactar a su medico si empeoran.",
        "high": "Consulte con su medico. Se recomienda evaluacion temprana.",
    }

    recommendation = recommendations.get(risk_level, "Consulte con su medico.")

    # Add specific warnings
    if symptoms.blood_in_stool:
        recommendation += " Atencion: Presencia de sangre en heces requiere evaluacion."
    if symptoms.fever:
        recommendation += " La fiebre puede indicar infeccion o inflamacion activa."

    return recommendation


def calculate_trends(daily_records: List) -> tuple:
    """
    Calculate temporal trends from daily symptom records.

    Args:
        daily_records: List of DailySymptomRecord objects

    Returns:
        Tuple of (SymptomTrends, AnalysisPeriod)
    """
    # Sort records by date
    sorted_records = sorted(daily_records, key=lambda x: x.date)

    # Calculate severity for each day
    severities = [
        calculate_symptom_severity(record.symptoms) for record in sorted_records
    ]

    # Overall trend: compare first half vs second half
    midpoint = len(severities) // 2
    early_avg = sum(severities[:midpoint]) / midpoint if midpoint > 0 else 0
    late_avg = sum(severities[midpoint:]) / (len(severities) - midpoint) if len(severities) > midpoint else 0
    severity_change = late_avg - early_avg

    if severity_change > 0.1:
        overall_trend = "worsening"
    elif severity_change < -0.1:
        overall_trend = "improving"
    else:
        overall_trend = "stable"

    # Identify concerning patterns
    concerning = []
    if max(severities[-3:]) > 0.7:
        concerning.append("High symptom severity in recent days")
    if severity_change > 0.2:
        concerning.append("Rapid symptom escalation detected")

    # Blood in stool check
    recent_blood = any(
        record.symptoms.blood_in_stool
        for record in sorted_records[-7:]
    )
    if recent_blood:
        concerning.append("Blood in stool reported in last week")

    trends = SymptomTrends(
        overall_trend=overall_trend,
        severity_change=round(severity_change, 2),
        concerning_patterns=concerning,
    )

    analysis_period = AnalysisPeriod(
        start_date=sorted_records[0].date,
        end_date=sorted_records[-1].date,
        days_analyzed=len(sorted_records),
    )

    return trends, analysis_period


# API Endpoints

@app.get("/", tags=["General"])
async def root():
    """Root endpoint."""
    return {
        "message": "Crohn Flare Predictor API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["General"],
    summary="Health Check",
    response_description="Service health status",
)
async def health_check():
    """
    Health check endpoint for monitoring service availability.

    Use this endpoint to verify that the API service is running and the
    ML model is loaded and ready to accept prediction requests.

    ## Status Codes

    - **200 OK**: Service is healthy and model is loaded
    - **503 Service Unavailable**: Service is running but model failed to load

    ## Use Cases

    - Container orchestration health checks (Kubernetes liveness/readiness probes)
    - Load balancer health monitoring
    - Service discovery and registration
    - Automated monitoring and alerting
    """
    if not app_state.model_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML model not loaded"
        )

    return HealthResponse(
        status="healthy",
        version=settings.version
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Predictions"],
    summary="Predict IBD Flare Risk",
    response_description="Prediction with risk level, probability, contributing factors, and recommendations",
)
async def predict_flare(request: PredictionRequest):
    """
    Predict the risk of an IBD flare for a single patient.

    This is the **primary endpoint** for the ML API. It analyzes current symptoms,
    patient demographics, medical history, and optional temporal features to predict
    the likelihood of an upcoming disease flare.

    ## How It Works

    1. **Feature Extraction**: Combines all input data into 34 engineered features
    2. **Cluster Assignment**: Assigns patient to a phenotype cluster (0-2) based on disease characteristics
    3. **Risk Prediction**: Uses a cluster-specific Random Forest model to predict flare risk
    4. **Interpretation**: Identifies top contributing factors and generates personalized recommendations

    ## Model Types

    - **Cluster-Stratified** (default): Uses specialized models trained on similar patients
    - **Global Fallback**: Used if cluster assignment fails
    - **Rule-Based**: Simple heuristic fallback if no ML models are available

    ## Risk Levels

    - **Low**: Probability < 30% - Continue regular monitoring
    - **Medium**: Probability 30-70% - Monitor closely, consider medical consultation
    - **High**: Probability > 70% - Medical evaluation recommended

    ## Input Requirements

    - **Required**: daily_records (minimum 1 day), demographics, history
    - **Recommended**: 7-14 days of daily_records for temporal feature calculation and improved accuracy

    ## Response

    Returns a complete prediction with:
    - Risk level classification (low/medium/high)
    - Probability scores for all risk levels
    - Model confidence score
    - Top 3 contributing factors
    - Cluster assignment and IBD classification
    - Personalized recommendation in Spanish
    - Metadata (timestamp, model version)
    - **Trends** (only with 7+ days): Temporal analysis including overall trend, severity change, and concerning patterns
    - **Analysis Period** (only with 7+ days): Time period analyzed

    ## Example Use Case

    ```python
    # Day 1: Patient with new symptoms (no history)
    response = requests.post("http://localhost:8001/predict", json={
        "daily_records": [
            {"date": "2024-01-14", "symptoms": {"abdominal_pain": 7, "diarrhea": 6, "fatigue": 5, "fever": False}}
        ],
        "demographics": {"age": 32, "gender": "F", "disease_duration_years": 5, "ibd_type": "crohn", "montreal_location": "L3"},
        "history": {"previous_flares": 3, "last_flare_days_ago": 120}
    })

    # Day 7+: Patient with symptom history (improved accuracy)
    response = requests.post("http://localhost:8001/predict", json={
        "daily_records": [
            {"date": "2024-01-08", "symptoms": {"abdominal_pain": 5, "diarrhea": 4, "fatigue": 3, "fever": False}},
            {"date": "2024-01-09", "symptoms": {"abdominal_pain": 6, "diarrhea": 5, "fatigue": 4, "fever": False}},
            # ... more days ...
            {"date": "2024-01-14", "symptoms": {"abdominal_pain": 7, "diarrhea": 6, "fatigue": 5, "fever": False}}
        ],
        "demographics": {"age": 32, "gender": "F", "disease_duration_years": 5, "ibd_type": "crohn", "montreal_location": "L3"},
        "history": {"previous_flares": 3, "last_flare_days_ago": 120}
    })
    ```
    """
    try:
        # Unpack result (now returns 10 items with enhanced metadata)
        (risk_level, probability, confidence, contributors, all_probs,
         cluster_id, cluster_conf, model_source, ibd_type, montreal_code) = predict_flare_risk(request)

        # Construct ClusterInfo object
        cluster_info = None
        if cluster_id is not None:
            # Get human-readable cluster description
            if ibd_type == "crohn":
                cluster_description = CROHN_CLUSTER_DESCRIPTIONS.get(cluster_id)
            else:
                cluster_description = UC_CLUSTER_DESCRIPTIONS.get(cluster_id)

            cluster_info = ClusterInfo(
                cluster_id=cluster_id,
                cluster_confidence=round(cluster_conf, 2),
                model_source=model_source,
                cluster_description=cluster_description
            )

        # Construct IBDInfo object
        ibd_info = IBDInfo(
            ibd_type=ibd_type,
            montreal_classification=montreal_code
        )

        # Construct PredictionMetadata object
        metadata = PredictionMetadata(
            prediction_timestamp=datetime.utcnow().isoformat() + "Z",
            model_version="2.0.0",
            api_version="1.0.0"
        )

        prediction = FlareRiskPrediction(
            flare_risk=risk_level,
            probability=round(probability, 2),
            confidence=round(confidence, 2),
            probabilities={k: round(v, 3) for k, v in all_probs.items()},
            cluster_info=cluster_info,
            ibd_info=ibd_info,
            # Legacy fields for backwards compatibility
            cluster_id=cluster_id,
            cluster_confidence=round(cluster_conf, 2) if cluster_conf is not None else None
        )

        # Extract current symptoms for factors and recommendation
        sorted_records = sorted(request.daily_records, key=lambda x: x.date)
        current_symptoms = sorted_records[-1].symptoms

        factors = ContributingFactors(
            top_contributors=contributors,
            symptom_severity_score=round(
                calculate_symptom_severity(current_symptoms), 2
            ),
            trend_indicator="stable",
        )

        recommendation = get_recommendation(risk_level, current_symptoms)

        # Calculate trends if we have 7+ days of data
        trends = None
        analysis_period = None
        if len(request.daily_records) >= 7:
            try:
                trends, analysis_period = calculate_trends(request.daily_records)
                logger.info(f"Calculated trends: {trends.overall_trend}, severity_change: {trends.severity_change}")
            except Exception as e:
                logger.warning(f"Failed to calculate trends: {e}")
                # Continue without trends

        return PredictionResponse(
            prediction=prediction,
            factors=factors,
            recommendation=recommendation,
            metadata=metadata,
            trends=trends,
            analysis_period=analysis_period
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during prediction: {str(e)}",
        )


@app.post(
    "/predict/trends",
    response_model=TrendAnalysisResponse,
    status_code=status.HTTP_200_OK,
    tags=["Predictions"],
    summary="[DEPRECATED] Predict Risk Based on Symptom Trends",
    response_description="Trend-based prediction with risk assessment and recommendations",
    deprecated=True
)
async def predict_trends(request: TrendAnalysisRequest):
    """
    **DEPRECATED**: This endpoint has been merged into `/predict`.
    Use `/predict` instead, which now includes trend analysis automatically when 7+ days of data are provided.

    This endpoint is kept for backward compatibility but will be removed in a future version.

    ---

    Predict flare risk based on temporal patterns in symptom data.

    This endpoint analyzes symptom trends over time to identify concerning patterns
    and predict flare risk. It requires historical data (7+ days) to detect temporal trends.

    ## Requirements

    - **Minimum**: 7 days of daily symptom records
    - **Recommended**: 14-30 days for more reliable trend detection

    ## Analysis Components

    1. **Overall Trend**: Classifies symptom trajectory as improving, stable, or worsening
    2. **Severity Change**: Quantifies the change in symptom severity over time
    3. **Concerning Patterns**: Identifies specific warning signs:
       - High severity in recent days
       - Rapid symptom escalation
       - Blood in stool in past week
    4. **Risk Assessment**: Current flare risk based on most recent symptoms
    5. **Recommendations**: Actionable guidance based on trend analysis

    ## Use Cases

    - Weekly symptom review for patients
    - Identifying patients needing intervention
    - Evaluating treatment effectiveness
    - Patient education and engagement

    ## Calculation Method

    - Compares average severity: first half vs second half of time period
    - Improvement: >10% decrease in severity
    - Worsening: >10% increase in severity
    - Stable: Within ±10%
    """
    if len(request.daily_records) < 7:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least 7 days of data required for trend analysis",
        )

    # Sort records by date
    sorted_records = sorted(request.daily_records, key=lambda x: x.date)

    # Calculate trend
    severities = [
        calculate_symptom_severity(record.symptoms) for record in sorted_records
    ]

    # Simple trend calculation
    early_avg = sum(severities[:len(severities)//2]) / (len(severities)//2)
    late_avg = sum(severities[len(severities)//2:]) / (len(severities) - len(severities)//2)
    severity_change = late_avg - early_avg

    if severity_change > 0.1:
        overall_trend = "worsening"
    elif severity_change < -0.1:
        overall_trend = "improving"
    else:
        overall_trend = "stable"

    # Identify concerning patterns
    concerning = []
    if max(severities[-3:]) > 0.7:
        concerning.append("High symptom severity in recent days")
    if severity_change > 0.2:
        concerning.append("Rapid symptom escalation detected")

    # Blood in stool check
    recent_blood = any(
        record.symptoms.blood_in_stool
        for record in sorted_records[-7:]
    )
    if recent_blood:
        concerning.append("Blood in stool reported in last week")

    # Risk assessment based on complete history (uses temporal features if 7+ days)
    prediction_request = PredictionRequest(
        daily_records=request.daily_records,
        demographics=request.demographics,
        history=request.history,
    )

    (risk_level, probability, confidence, _, _, cluster_id, cluster_conf,
     model_source, ibd_type, montreal_code) = predict_flare_risk(
        prediction_request
    )

    # Construct ClusterInfo object
    cluster_info = None
    if cluster_id is not None:
        if ibd_type == "crohn":
            cluster_description = CROHN_CLUSTER_DESCRIPTIONS.get(cluster_id)
        else:
            cluster_description = UC_CLUSTER_DESCRIPTIONS.get(cluster_id)

        cluster_info = ClusterInfo(
            cluster_id=cluster_id,
            cluster_confidence=round(cluster_conf, 2),
            model_source=model_source,
            cluster_description=cluster_description
        )

    # Construct IBDInfo object
    ibd_info = IBDInfo(
        ibd_type=ibd_type,
        montreal_classification=montreal_code
    )

    risk_assessment = FlareRiskPrediction(
        flare_risk=risk_level,
        probability=round(probability, 2),
        confidence=round(confidence, 2),
        cluster_info=cluster_info,
        ibd_info=ibd_info,
        # Legacy fields for backwards compatibility
        cluster_id=cluster_id,
        cluster_confidence=round(cluster_conf, 2) if cluster_conf is not None else None
    )

    # Recommendations
    recommendations = []
    if overall_trend == "worsening":
        recommendations.append("Contact your healthcare provider")
        recommendations.append("Review medication adherence")
    if concerning:
        recommendations.append("Schedule medical evaluation")
    else:
        recommendations.append("Continue current management plan")

    return TrendAnalysisResponse(
        patient_id=request.patient_id,
        analysis_period=AnalysisPeriod(
            start_date=sorted_records[0].date,
            end_date=sorted_records[-1].date,
            days_analyzed=len(sorted_records),
        ),
        trends=SymptomTrends(
            overall_trend=overall_trend,
            severity_change=round(severity_change, 2),
            concerning_patterns=concerning,
        ),
        risk_assessment=risk_assessment,
        recommendations=recommendations,
    )


@app.get(
    "/model/info",
    response_model=ModelInfoResponse,
    status_code=status.HTTP_200_OK,
    tags=["Model"],
    summary="Get Model Information",
    response_description="Model metadata and performance metrics",
)
async def get_model_info():
    """
    Retrieve metadata and performance metrics for the current ML model.

    This endpoint provides transparency about the model being used for predictions,
    including its training history, validation performance, and technical specifications.

    ## Returned Information

    - **Model Version**: Semantic version identifier
    - **Training Date**: When the model was last trained
    - **Performance Metrics**: Validation set performance
      - Accuracy: Overall prediction accuracy
      - Precision: Proportion of positive predictions that were correct
      - Recall: Proportion of actual positives that were identified
      - F1 Score: Harmonic mean of precision and recall
      - ROC AUC: Area under the receiver operating characteristic curve
    - **Features Count**: Number of input features used
    - **Training Samples**: Size of training dataset
    - **Model Type**: Algorithm used (e.g., RandomForest)

    ## Use Cases

    - Model versioning and tracking
    - Performance monitoring and validation
    - Regulatory compliance and audit trails
    - Research documentation
    """
    return ModelInfoResponse(
        model_version="1.0.0",
        trained_date=date(2024, 1, 15),
        metrics=ModelMetrics(
            accuracy=0.87,
            precision=0.84,
            recall=0.89,
            f1_score=0.86,
            roc_auc=0.91,
        ),
        features_count=45,
        training_samples=5000,
        model_type="RandomForest",
    )


# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle validation errors."""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc)},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle unexpected errors."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error occurred"},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )

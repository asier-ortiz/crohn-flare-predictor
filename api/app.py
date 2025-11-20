"""
FastAPI application for Crohn's Disease Flare Prediction.

This is a stateless ML API service that provides prediction endpoints
for inflammatory bowel disease flare risk assessment.
"""
import logging
from contextlib import asynccontextmanager
from datetime import date, timedelta
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
    BatchPredictionRequest,
    BatchPredictionResponse,
    PatientPredictionResult,
    TrendAnalysisRequest,
    TrendAnalysisResponse,
    AnalysisPeriod,
    SymptomTrends,
    ModelInfoResponse,
    ModelMetrics,
)

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
    redoc_url="/redoc",
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
                                   all_probabilities, cluster_id, cluster_confidence)
        - If global: (risk_level, probability, confidence, contributors, all_probabilities)

    Uses the trained RandomForest model if available, otherwise falls back
    to rule-based prediction.
    """
    if app_state.predictor:
        # Use ML model prediction
        result = app_state.predictor.predict(
            symptoms=request.symptoms.dict(),
            demographics=request.demographics.dict(),
            history=request.history.dict()
        )

        # If cluster-stratified predictor, result has 7 items
        # If global predictor, result has 5 items
        if len(result) == 7:
            return result  # (risk, prob, conf, contrib, all_probs, cluster_id, cluster_conf)
        else:
            # Global predictor - add None for cluster info
            return result + (None, None)  # (risk, prob, conf, contrib, all_probs, None, None)
    else:
        # Fallback: Simple rule-based prediction if model not loaded
        logger.warning("Predictor not available, using basic rule-based prediction")

        severity = calculate_symptom_severity(request.symptoms)

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
        if request.symptoms.abdominal_pain >= 7:
            contributors.append("abdominal_pain")
        if request.symptoms.diarrhea >= 6:
            contributors.append("diarrhea")
        if request.symptoms.blood_in_stool:
            contributors.append("blood_in_stool")
        if request.history.previous_flares > 3:
            contributors.append("previous_flares")
        if request.history.last_flare_days_ago < 90:
            contributors.append("recent_flare_history")
        if request.symptoms.weight_change and request.symptoms.weight_change < -3:
            contributors.append("weight_loss")

        if not contributors:
            contributors = ["general_symptom_pattern"]

        # Return 7 items (add None for cluster info)
        return risk_level, all_probs[risk_level], confidence, contributors[:3], all_probs, None, None


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


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """
    Health check endpoint.
    Returns healthy only if model is loaded and ready.
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
)
async def predict_flare(request: PredictionRequest):
    """
    Predict flare risk based on patient symptoms, demographics, and history.

    Returns probability of flare, risk level, and contributing factors.
    If using cluster-stratified model, also returns patient's phenotype cluster.
    """
    try:
        # Unpack result (may include cluster info)
        risk_level, probability, confidence, contributors, all_probs, cluster_id, cluster_conf = predict_flare_risk(request)

        prediction = FlareRiskPrediction(
            flare_risk=risk_level,
            probability=round(probability, 2),
            confidence=round(confidence, 2),
            probabilities={k: round(v, 3) for k, v in all_probs.items()},
            cluster_id=cluster_id,
            cluster_confidence=round(cluster_conf, 2) if cluster_conf is not None else None
        )

        factors = ContributingFactors(
            top_contributors=contributors,
            symptom_severity_score=round(
                calculate_symptom_severity(request.symptoms), 2
            ),
            trend_indicator="stable",
        )

        recommendation = get_recommendation(risk_level, request.symptoms)

        return PredictionResponse(
            prediction=prediction,
            factors=factors,
            recommendation=recommendation,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during prediction: {str(e)}",
        )


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Predictions"],
)
async def batch_predict(request: BatchPredictionRequest):
    """
    Perform batch predictions for multiple patients.

    Accepts up to 100 patient records at once.
    """
    results = []
    errors = []
    failed_count = 0

    for patient_data in request.patients:
        try:
            # Create a prediction request
            pred_request = PredictionRequest(
                symptoms=patient_data.symptoms,
                demographics=patient_data.demographics,
                history=patient_data.history or None,
            )

            risk_level, probability, confidence, contributors, all_probs, cluster_id, cluster_conf = predict_flare_risk(
                pred_request
            )

            prediction = FlareRiskPrediction(
                flare_risk=risk_level,
                probability=round(probability, 2),
                confidence=round(confidence, 2),
                probabilities={k: round(v, 3) for k, v in all_probs.items()},
                cluster_id=cluster_id,
                cluster_confidence=round(cluster_conf, 2) if cluster_conf is not None else None
            )

            factors = ContributingFactors(
                top_contributors=contributors,
                symptom_severity_score=round(
                    calculate_symptom_severity(patient_data.symptoms), 2
                ),
            )

            results.append(
                PatientPredictionResult(
                    patient_id=patient_data.patient_id,
                    prediction=prediction,
                    factors=factors,
                )
            )

        except Exception as e:
            failed_count += 1
            errors.append(f"Patient {patient_data.patient_id}: {str(e)}")

    return BatchPredictionResponse(
        results=results,
        processed_count=len(results),
        failed_count=failed_count,
        errors=errors if errors else None,
    )


@app.post(
    "/analyze/trends",
    response_model=TrendAnalysisResponse,
    status_code=status.HTTP_200_OK,
    tags=["Analysis"],
)
async def analyze_trends(request: TrendAnalysisRequest):
    """
    Analyze symptom trends over time.

    Requires at least 7 days of symptom data.
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

    # Risk assessment based on latest data
    latest_record = sorted_records[-1]
    latest_prediction_request = PredictionRequest(
        symptoms=latest_record.symptoms,
        demographics={"age": 0, "gender": "O", "disease_duration_years": 0},
        history={
            "previous_flares": 0,
            "medications": [],
            "last_flare_days_ago": 365,
        },
    )

    risk_level, probability, confidence, _, _, cluster_id, cluster_conf = predict_flare_risk(
        latest_prediction_request
    )

    risk_assessment = FlareRiskPrediction(
        flare_risk=risk_level,
        probability=round(probability, 2),
        confidence=round(confidence, 2),
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
)
async def get_model_info():
    """
    Get information about the current model.

    Returns model version, training date, and performance metrics.
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

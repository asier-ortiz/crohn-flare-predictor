"""
Crohn Flare Predictor API

FastAPI application for predicting inflammatory bowel disease flares
based on daily symptom tracking data.

Usage:
    uvicorn api.app:app --reload --port 8000

Endpoints:
    GET  /              - API information
    GET  /health        - Health check
    GET  /model/info    - Model metadata
    POST /predict       - Single prediction
    POST /predict/batch - Batch predictions
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging
from datetime import datetime
from typing import Optional

from api.schemas import (
    PredictionRequest,
    BatchPredictionRequest,
    PredictionResponse,
    BatchPredictionResponse,
    ModelInfo,
    HealthResponse,
    SymptomFeatures
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Crohn Flare Predictor API",
    description="Machine learning API for predicting IBD flares based on symptom tracking",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware (allow all origins for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and metadata
MODEL = None
SCALER = None
METADATA = None
MODEL_DIR = Path(__file__).parent.parent / "models"


def load_model():
    """Load trained model, scaler, and metadata on startup"""
    global MODEL, SCALER, METADATA

    try:
        # Load XGBoost model
        model_path = MODEL_DIR / "xgboost_flare_predictor.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        MODEL = joblib.load(model_path)
        logger.info(f" Model loaded from {model_path}")

        # Load scaler
        scaler_path = MODEL_DIR / "scaler.pkl"
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found at {scaler_path}")
        SCALER = joblib.load(scaler_path)
        logger.info(f" Scaler loaded from {scaler_path}")

        # Load metadata
        metadata_path = MODEL_DIR / "model_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found at {metadata_path}")
        with open(metadata_path, 'r') as f:
            METADATA = json.load(f)
        logger.info(f" Metadata loaded from {metadata_path}")

        logger.info(f" Model ready: {METADATA['model_name']}")
        logger.info(f"  - Optimal threshold (F2): {METADATA['optimal_threshold_f2']:.3f}")
        logger.info(f"  - Test Recall: {METADATA['test_recall']:.3f}")
        logger.info(f"  - Test Precision: {METADATA['test_precision']:.3f}")

    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise


@app.on_event("startup")
async def startup_event():
    """Load model when API starts"""
    logger.info("=" * 80)
    logger.info("Starting Crohn Flare Predictor API")
    logger.info("=" * 80)
    load_model()
    logger.info("=" * 80)
    logger.info("API ready to accept requests")
    logger.info("=" * 80)


def features_to_array(features: SymptomFeatures) -> np.ndarray:
    """
    Convert SymptomFeatures Pydantic model to numpy array in correct order.

    Args:
        features: SymptomFeatures object

    Returns:
        numpy array of shape (1, 43) with features in correct order
    """
    # Convert to dict and extract values in the order defined in metadata
    feature_dict = features.model_dump()

    # Ensure features are in the exact order expected by the model
    feature_values = [feature_dict[feat] for feat in METADATA['features']]

    return np.array(feature_values).reshape(1, -1)


def classify_risk_level(probability: float) -> str:
    """
    Classify risk level based on flare probability.

    Args:
        probability: Flare probability (0-1)

    Returns:
        Risk level: "Low", "Medium", "High", or "Critical"
    """
    if probability < 0.2:
        return "Low"
    elif probability < 0.5:
        return "Medium"
    elif probability < 0.8:
        return "High"
    else:
        return "Critical"


def get_confidence_level(probability: float, threshold: float) -> str:
    """
    Determine confidence level based on distance from threshold.

    Args:
        probability: Predicted probability
        threshold: Classification threshold

    Returns:
        Confidence level: "Low", "Medium", or "High"
    """
    distance = abs(probability - threshold)

    if distance < 0.1:
        return "Low"
    elif distance < 0.3:
        return "Medium"
    else:
        return "High"


def get_recommendation(is_flare: bool, probability: float, risk_level: str) -> str:
    """
    Generate clinical recommendation based on prediction.

    Args:
        is_flare: Predicted flare status
        probability: Flare probability
        risk_level: Risk level classification

    Returns:
        Recommendation string
    """
    if risk_level == "Critical":
        return "URGENT: High risk of flare detected. Contact your healthcare provider immediately. Monitor symptoms closely."
    elif risk_level == "High":
        return "High risk of flare. Consider contacting your healthcare provider. Increase symptom monitoring."
    elif risk_level == "Medium":
        return "Moderate risk of flare. Continue monitoring symptoms and maintain treatment plan."
    else:
        return "Low risk of flare. Continue current treatment plan and routine monitoring."


@app.get("/", tags=["Info"])
async def root():
    """API welcome message and basic information"""
    return {
        "name": "Crohn Flare Predictor API",
        "version": "1.0.0",
        "description": "ML-powered API for predicting IBD flares from symptom tracking data",
        "endpoints": {
            "GET /health": "Health check and model status",
            "GET /model/info": "Model metadata and performance metrics",
            "POST /predict": "Single prediction from symptom features",
            "POST /predict/batch": "Batch predictions (up to 1000 at once)",
            "GET /docs": "Interactive API documentation (Swagger UI)",
            "GET /redoc": "Alternative API documentation (ReDoc)"
        },
        "model": METADATA['model_name'] if METADATA else "Not loaded",
        "status": "ready" if MODEL else "not ready"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint to verify API and model status.

    Returns:
        HealthResponse with API status and model information
    """
    if MODEL is None or SCALER is None or METADATA is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. API not ready."
        )

    return HealthResponse(
        status="healthy",
        message="API is running and model is loaded",
        model_loaded=True,
        model_type=METADATA['model_type']
    )


@app.get("/model/info", response_model=ModelInfo, tags=["Model"])
async def get_model_info():
    """
    Get model metadata and performance metrics.

    Returns:
        ModelInfo with model details, metrics, and feature list
    """
    if METADATA is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model metadata not loaded"
        )

    return ModelInfo(**METADATA)


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """
    Predict flare risk from daily symptom features.

    Args:
        request: PredictionRequest containing SymptomFeatures and optional threshold

    Returns:
        PredictionResponse with flare prediction, probability, and recommendations

    Raises:
        HTTPException: If model is not loaded or prediction fails
    """
    if MODEL is None or SCALER is None or METADATA is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )

    try:
        # Convert features to numpy array
        X = features_to_array(request.features)

        # Note: XGBoost doesn't need scaling, but we keep scaler for consistency
        # with other models that might be added later

        # Get prediction probability
        y_proba = MODEL.predict_proba(X)[0, 1]  # Probability of flare (class 1)

        # Determine threshold
        threshold = request.threshold if request.threshold is not None else METADATA['optimal_threshold_f2']

        # Classify
        is_flare = bool(y_proba >= threshold)

        # Get risk level and confidence
        risk_level = classify_risk_level(y_proba)
        confidence = get_confidence_level(y_proba, threshold)
        recommendation = get_recommendation(is_flare, y_proba, risk_level)

        # Log prediction (optional: save to database)
        logger.info(
            f"Prediction: is_flare={is_flare}, probability={y_proba:.3f}, "
            f"risk={risk_level}, user_id={request.user_id}"
        )

        return PredictionResponse(
            is_flare=is_flare,
            flare_probability=float(y_proba),
            confidence=confidence,
            threshold_used=threshold,
            risk_level=risk_level,
            recommendation=recommendation
        )

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest):
    """
    Batch prediction for multiple symptom feature sets.

    Args:
        request: BatchPredictionRequest with list of SymptomFeatures

    Returns:
        BatchPredictionResponse with list of predictions and summary statistics

    Raises:
        HTTPException: If model is not loaded or predictions fail
    """
    if MODEL is None or SCALER is None or METADATA is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )

    try:
        predictions = []
        threshold = request.threshold if request.threshold is not None else METADATA['optimal_threshold_f2']

        for features in request.predictions:
            # Convert features to array
            X = features_to_array(features)

            # Predict
            y_proba = MODEL.predict_proba(X)[0, 1]
            is_flare = bool(y_proba >= threshold)

            # Calculate additional info
            risk_level = classify_risk_level(y_proba)
            confidence = get_confidence_level(y_proba, threshold)
            recommendation = get_recommendation(is_flare, y_proba, risk_level)

            predictions.append(
                PredictionResponse(
                    is_flare=is_flare,
                    flare_probability=float(y_proba),
                    confidence=confidence,
                    threshold_used=threshold,
                    risk_level=risk_level,
                    recommendation=recommendation
                )
            )

        # Calculate summary statistics
        total_predictions = len(predictions)
        flare_count = sum(1 for p in predictions if p.is_flare)
        avg_probability = sum(p.flare_probability for p in predictions) / total_predictions

        logger.info(
            f"Batch prediction: {total_predictions} predictions, "
            f"{flare_count} flares, avg_prob={avg_probability:.3f}"
        )

        return BatchPredictionResponse(
            predictions=predictions,
            total_predictions=total_predictions,
            flare_count=flare_count,
            average_flare_probability=avg_probability
        )

    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unexpected errors"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Internal server error occurred",
            "message": str(exc)
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

"""
ML Model loader and predictor.
Loads and manages the trained RandomForest model.
"""
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class CrohnPredictor:
    """
    Wrapper for the trained ML model.
    Handles feature extraction and prediction.
    """

    def __init__(self, model_path: str = "models/rf_severity_classifier.pkl"):
        """
        Initialize predictor and load model.

        Args:
            model_path: Path to the trained model file
        """
        self.model = None
        self.model_path = Path(model_path)
        self.is_loaded = False

    def load_model(self):
        """Load the trained model from disk."""
        try:
            if not self.model_path.exists():
                logger.warning(f"Model file not found: {self.model_path}")
                logger.warning("Using rule-based predictions as fallback")
                return False

            logger.info(f"Loading model from {self.model_path}")
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)

            logger.info(f"Model loaded successfully: {type(self.model).__name__}")
            self.is_loaded = True
            return True

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.is_loaded = False
            return False

    def extract_features(
        self,
        symptoms: Dict,
        demographics: Dict,
        history: Dict
    ) -> np.ndarray:
        """
        Extract features from input data.

        IMPORTANT: This must match EXACTLY the 13 features the model was trained with.
        Feature order from notebooks/03_model_training.ipynb:
        ['abdominal_pain', 'blood_in_stool', 'diarrhea', 'fatigue', 'fever', 'nausea',
         'age', 'gender', 'disease_duration_years', 'previous_flares',
         'last_flare_days_ago', 'month', 'day_of_week']

        Args:
            symptoms: Symptom data
            demographics: Demographic data
            history: Medical history data

        Returns:
            Feature array for model prediction (1, 13)
        """
        # Get current date for temporal features
        now = datetime.now()

        # Feature vector - MUST BE EXACTLY 13 features in this order
        features = [
            # Symptoms (6 features) - normalized to 0-1 scale
            symptoms.get("abdominal_pain", 0) / 10.0,  # Scale 0-10 → 0-1
            int(symptoms.get("blood_in_stool", False)),  # Boolean → 0/1
            symptoms.get("diarrhea", 0) / 10.0,  # Scale 0-10 → 0-1
            symptoms.get("fatigue", 0) / 10.0,  # Scale 0-10 → 0-1
            int(symptoms.get("fever", False)),  # Boolean → 0/1
            symptoms.get("nausea", 0) / 10.0,  # Scale 0-10 → 0-1

            # Demographics (2 features)
            demographics.get("age", 30.0),  # Default age 30
            1 if demographics.get("gender") == "M" else (2 if demographics.get("gender") == "F" else 0),

            # History (3 features)
            demographics.get("disease_duration_years", 0.0),
            history.get("previous_flares", 0),
            history.get("last_flare_days_ago", 365),

            # Temporal features (2 features)
            now.month,  # 1-12
            now.weekday()  # 0-6 (Monday=0)
        ]

        return np.array(features).reshape(1, -1)

    def predict(
        self,
        symptoms: Dict,
        demographics: Dict,
        history: Dict
    ) -> Tuple[str, float, float, List[str], Dict[str, float]]:
        """
        Make a prediction using the ML model.

        Args:
            symptoms: Symptom data
            demographics: Demographic data
            history: Medical history data

        Returns:
            Tuple of (risk_level, probability, confidence, contributors, all_probabilities)
        """
        if not self.is_loaded or self.model is None:
            # Fallback to rule-based prediction
            return self._rule_based_prediction(symptoms, demographics, history)

        try:
            # Extract features
            features = self.extract_features(symptoms, demographics, history)

            # Get prediction (model predicts 'low', 'medium', or 'high' directly)
            prediction = self.model.predict(features)[0]
            probabilities = self.model.predict_proba(features)[0]

            # Get class labels from model
            class_labels = self.model.classes_  # ['high', 'low', 'medium'] or similar
            risk_level = str(prediction).lower()  # Ensure lowercase string

            # Probability for predicted class
            pred_idx = list(class_labels).index(prediction)
            probability = float(probabilities[pred_idx])

            # Confidence (diferencia entre top 2 probabilidades)
            sorted_probs = sorted(probabilities, reverse=True)
            confidence = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else 1.0

            # All probabilities as dict
            all_probs = {str(cls).lower(): float(prob) for cls, prob in zip(class_labels, probabilities)}

            # Identify top contributors (features con más peso)
            contributors = self._identify_contributors(symptoms, history)

            logger.info(f"ML Prediction: {risk_level} (prob={probability:.2f}, conf={confidence:.2f})")

            return risk_level, probability, confidence, contributors, all_probs

        except Exception as e:
            logger.error(f"Error in ML prediction: {e}", exc_info=True)
            # Fallback to rule-based
            return self._rule_based_prediction(symptoms, demographics, history)

    def _identify_contributors(self, symptoms: Dict, history: Dict) -> List[str]:
        """
        Identify top contributing factors based on symptoms and history.
        Uses feature importance from trained model (top 3 features: diarrhea, abdominal_pain, fatigue).

        Args:
            symptoms: Symptom data
            history: Medical history data

        Returns:
            List of top contributing factor names
        """
        contributors = []

        # Check symptom severity (most important features according to model)
        if symptoms.get("diarrhea", 0) >= 6:
            contributors.append("diarrhea")
        if symptoms.get("abdominal_pain", 0) >= 7:
            contributors.append("abdominal_pain")
        if symptoms.get("fatigue", 0) >= 6:
            contributors.append("fatigue")
        if symptoms.get("blood_in_stool", False):
            contributors.append("blood_in_stool")
        if symptoms.get("fever", False):
            contributors.append("fever")
        if symptoms.get("nausea", 0) >= 6:
            contributors.append("nausea")

        # Check history
        if history.get("previous_flares", 0) > 3:
            contributors.append("previous_flares")
        if history.get("last_flare_days_ago", 365) < 90:
            contributors.append("recent_flare_history")

        # If no specific contributors, use general
        if not contributors:
            contributors.append("general_symptom_pattern")

        return contributors[:3]  # Top 3

    def _rule_based_prediction(
        self,
        symptoms: Dict,
        demographics: Dict,
        history: Dict
    ) -> Tuple[str, float, float, List[str], Dict[str, float]]:
        """
        Fallback rule-based prediction when ML model is not available.

        Returns:
            Tuple of (risk_level, probability, confidence, contributors, all_probabilities)
        """
        logger.warning("Using rule-based prediction (model not loaded)")

        # Calculate symptom severity
        severity_score = (
            symptoms.get("abdominal_pain", 0) / 10 +
            symptoms.get("diarrhea", 0) / 10 +
            symptoms.get("fatigue", 0) / 10 +
            symptoms.get("nausea", 0) / 10 +
            (1.0 if symptoms.get("fever", False) else 0) +
            (1.0 if symptoms.get("blood_in_stool", False) else 0)
        ) / 6.0

        # History risk
        history_risk = 0.0
        if history.get("previous_flares", 0) > 3:
            history_risk += 0.2
        if history.get("last_flare_days_ago", 365) < 90:
            history_risk += 0.3
        if history.get("surgery_history", False):
            history_risk += 0.1

        # Weight change
        if symptoms.get("weight_change", 0) < -2:
            history_risk += min(abs(symptoms.get("weight_change", 0)) / 20, 0.2)

        # Total risk
        total_risk = min(severity_score * 0.7 + history_risk * 0.3, 1.0)

        # Classify and create probability distribution
        if total_risk < 0.3:
            risk_level = "low"
            confidence = 0.85
            all_probs = {
                "low": 0.85,
                "medium": 0.12,
                "high": 0.03
            }
        elif total_risk < 0.6:
            risk_level = "medium"
            confidence = 0.75
            all_probs = {
                "low": 0.15,
                "medium": 0.75,
                "high": 0.10
            }
        else:
            risk_level = "high"
            confidence = 0.80
            all_probs = {
                "low": 0.05,
                "medium": 0.15,
                "high": 0.80
            }

        contributors = self._identify_contributors(symptoms, history)

        return risk_level, all_probs[risk_level], confidence, contributors, all_probs

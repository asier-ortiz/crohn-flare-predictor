"""
ML Model loader and predictor.
Loads and manages the trained RandomForest model.
Supports both global model and cluster-stratified models.
Supports separate models for Crohn's Disease and Ulcerative Colitis.
"""
import pickle
import numpy as np
import json
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import logging
from datetime import datetime

from .constants import (
    get_cluster_from_montreal,
    get_ibd_type_from_montreal,
    IBD_TYPES
)

logger = logging.getLogger(__name__)


def calculate_temporal_features_from_records(
    daily_records: List[Dict],
    ibd_type: str = 'crohn'
) -> Optional[Dict]:
    """
    Calculate temporal features from an array of daily symptom records.

    This function replicates the logic from notebooks/03_advanced_feature_engineering.ipynb
    to ensure consistency between training and prediction.

    Args:
        daily_records: List of dicts with 'date' and 'symptoms' keys
        ibd_type: Type of IBD ('crohn' or 'ulcerative_colitis')

    Returns:
        Dict with temporal features, or None if insufficient data

    Required features (6):
        - pain_trend_7d: 7-day mean of pain
        - diarrhea_trend_7d: 7-day mean of diarrhea
        - fatigue_trend_7d: 7-day mean of fatigue
        - symptom_volatility_7d: 7-day std of total symptom score
        - symptom_change_rate: Change from 7 days ago
        - days_since_low_symptoms: Consecutive days with high symptoms
    """
    if len(daily_records) < 7:
        logger.debug(f"Insufficient records for temporal features: {len(daily_records)} < 7")
        return None

    # Sort by date to ensure chronological order
    sorted_records = sorted(daily_records, key=lambda x: x['date'])

    # Extract symptom values (normalized to 0-1)
    pain_values = []
    diarrhea_values = []
    fatigue_values = []
    symptom_scores = []

    for record in sorted_records:
        symptoms = record['symptoms']

        # Normalize to 0-1 scale
        pain_norm = symptoms.get('abdominal_pain', 0) / 10.0
        diarrhea_norm = symptoms.get('diarrhea', 0) / 10.0
        fatigue_norm = symptoms.get('fatigue', 0) / 10.0
        nausea_norm = symptoms.get('nausea', 0) / 10.0
        blood_int = int(symptoms.get('blood_in_stool', False))
        fever_int = int(symptoms.get('fever', False))

        pain_values.append(pain_norm)
        diarrhea_values.append(diarrhea_norm)
        fatigue_values.append(fatigue_norm)

        # Calculate total_symptom_score (same weights as notebook)
        if ibd_type == 'crohn':
            score = (
                pain_norm * 1.2 +
                diarrhea_norm * 1.3 +
                fatigue_norm * 1.0 +
                nausea_norm * 0.8 +
                blood_int * 2.0 +
                fever_int * 1.5
            )
        else:  # UC - blood and diarrhea weigh more
            score = (
                pain_norm * 1.0 +
                diarrhea_norm * 1.5 +
                fatigue_norm * 1.0 +
                nausea_norm * 0.8 +
                blood_int * 2.5 +
                fever_int * 1.5
            )
        symptom_scores.append(score)

    # 1. Rolling means (trends) - last 7 days
    pain_trend_7d = float(np.mean(pain_values[-7:]))
    diarrhea_trend_7d = float(np.mean(diarrhea_values[-7:]))
    fatigue_trend_7d = float(np.mean(fatigue_values[-7:]))

    # 2. Volatility (standard deviation of symptom scores)
    symptom_volatility_7d = float(np.std(symptom_scores[-7:]))

    # 3. Change rate (current vs 7 days ago)
    if len(symptom_scores) >= 7:
        symptom_change_rate = float(symptom_scores[-1] - symptom_scores[-7])
    else:
        symptom_change_rate = 0.0

    # 4. Days since low symptoms (consecutive days with high symptoms)
    # High symptoms = total_symptom_score > 3.0
    days_since_low = 0
    for score in reversed(symptom_scores):
        if score > 3.0:
            days_since_low += 1
        else:
            break

    temporal_features = {
        'pain_trend_7d': pain_trend_7d,
        'diarrhea_trend_7d': diarrhea_trend_7d,
        'fatigue_trend_7d': fatigue_trend_7d,
        'symptom_volatility_7d': symptom_volatility_7d,
        'symptom_change_rate': symptom_change_rate,
        'days_since_low_symptoms': days_since_low
    }

    logger.info(f"Calculated temporal features from {len(sorted_records)} days of data")
    logger.debug(f"Temporal features: {temporal_features}")

    return temporal_features


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


class ClusterStratifiedPredictor:
    """
    Cluster-stratified predictor that:
    1. Infers patient's phenotype cluster based on symptoms
    2. Uses cluster-specific model for prediction
    3. Supports separate models for Crohn and Ulcerative Colitis

    This predictor automatically determines which cluster a patient belongs to
    and uses the specialized model trained for that cluster and IBD type.
    """

    def __init__(self, models_dir: str = "models"):
        """
        Initialize cluster-stratified predictor.

        Args:
            models_dir: Directory containing cluster models
        """
        self.models_dir = Path(models_dir)

        # Separate storage for each IBD type
        self.cluster_models = {
            'crohn': {},
            'ulcerative_colitis': {}
        }
        self.kmeans = {
            'crohn': None,
            'ulcerative_colitis': None
        }
        self.scaler = {
            'crohn': None,
            'ulcerative_colitis': None
        }
        self.metadata = {
            'crohn': None,
            'ulcerative_colitis': None
        }
        self.cluster_meta = {
            'crohn': None,
            'ulcerative_colitis': None
        }
        self.global_models = {
            'crohn': None,
            'ulcerative_colitis': None
        }

        self.is_loaded = {
            'crohn': False,
            'ulcerative_colitis': False
        }

    def load_models(self):
        """
        Load cluster models for both Crohn and UC.
        Tries to load from ibd_type-specific subdirectories (crohn/, cu/)
        Falls back to root models/ directory for backwards compatibility.
        """
        any_loaded = False

        for ibd_type in ['crohn', 'ulcerative_colitis']:
            # Try both directory structures
            # 1. New structure: models/crohn/, models/cu/
            # 2. Old structure: models/ (backwards compatibility for 'crohn' only)
            if ibd_type == 'ulcerative_colitis':
                ibd_dirs = [self.models_dir / 'cu']
            else:
                ibd_dirs = [self.models_dir / 'crohn', self.models_dir]

            loaded = False
            for ibd_dir in ibd_dirs:
                try:
                    if self._load_ibd_models(ibd_type, ibd_dir):
                        loaded = True
                        any_loaded = True
                        break
                except Exception as e:
                    logger.debug(f"Could not load {ibd_type} models from {ibd_dir}: {e}")
                    continue

            if not loaded:
                logger.warning(f"No cluster models loaded for {ibd_type}")
            else:
                logger.info(f"✓ Loaded cluster models for {ibd_type}")

        return any_loaded

    def _load_ibd_models(self, ibd_type: str, models_dir: Path) -> bool:
        """
        Load models for a specific IBD type from a directory.

        Args:
            ibd_type: 'crohn' or 'ulcerative_colitis'
            models_dir: Directory containing the models

        Returns:
            True if successfully loaded
        """
        # Load cluster metadata
        metadata_path = models_dir / "cluster_models_metadata.json"
        if not metadata_path.exists():
            return False

        with open(metadata_path, 'r') as f:
            self.metadata[ibd_type] = json.load(f)

        logger.debug(f"Loaded {ibd_type} metadata: {self.metadata[ibd_type]['n_clusters']} clusters")

        # Load cluster models
        for cluster_id in self.metadata[ibd_type]['clusters'].keys():
            model_file = self.metadata[ibd_type]['clusters'][cluster_id]['model_file']
            model_path = models_dir / model_file

            if not model_path.exists():
                logger.warning(f"Model not found: {model_path}")
                continue

            with open(model_path, 'rb') as f:
                self.cluster_models[ibd_type][int(cluster_id)] = pickle.load(f)

            logger.debug(f"Loaded {ibd_type} model for cluster {cluster_id}")

        if not self.cluster_models[ibd_type]:
            return False

        # Load KMeans
        kmeans_path = models_dir / "cluster_kmeans.pkl"
        if not kmeans_path.exists():
            return False

        with open(kmeans_path, 'rb') as f:
            self.kmeans[ibd_type] = pickle.load(f)

        # Load scaler
        scaler_path = models_dir / "cluster_scaler.pkl"
        if not scaler_path.exists():
            return False

        with open(scaler_path, 'rb') as f:
            self.scaler[ibd_type] = pickle.load(f)

        # Load cluster metadata for feature names
        cluster_meta_path = models_dir / "cluster_metadata.json"
        if cluster_meta_path.exists():
            with open(cluster_meta_path, 'r') as f:
                self.cluster_meta[ibd_type] = json.load(f)
        else:
            # Default features
            self.cluster_meta[ibd_type] = {
                'features': ['abdominal_pain', 'blood_in_stool', 'diarrhea',
                            'fatigue', 'fever', 'nausea']
            }

        # Load global model (fallback for clusters without specific models)
        global_model_path = models_dir / "rf_severity_classifier_global.pkl"
        if global_model_path.exists():
            with open(global_model_path, 'rb') as f:
                self.global_models[ibd_type] = pickle.load(f)
            logger.info(f"Loaded global model for {ibd_type} from {global_model_path}")
        else:
            logger.warning(f"Global model not found for {ibd_type} at {global_model_path}")

        self.is_loaded[ibd_type] = True
        return True

    def get_cluster_from_montreal(
        self,
        montreal_code: str,
        ibd_type: str
    ) -> Tuple[int, float]:
        """
        Get cluster ID from Montreal classification.
        Uses high confidence since this is user-provided medical classification.

        Args:
            montreal_code: Montreal code (L1-L4 for Crohn, E1-E3 for UC)
            ibd_type: IBD type ('crohn' or 'ulcerative_colitis')

        Returns:
            Tuple of (cluster_id, confidence)
        """
        try:
            cluster_id = get_cluster_from_montreal(montreal_code)
            # High confidence for user-provided classification
            confidence = 0.95
            logger.info(f"Using Montreal classification {montreal_code} → cluster {cluster_id}")
            return cluster_id, confidence
        except ValueError as e:
            logger.warning(f"Invalid Montreal code: {e}")
            # Fall back to inference
            return None, 0.0

    def infer_cluster(
        self,
        symptoms: Dict,
        ibd_type: str,
        montreal_code: Optional[str] = None
    ) -> Tuple[int, float]:
        """
        Infer patient's cluster based on symptoms or Montreal classification.
        Prioritizes user-provided Montreal classification if available.

        Args:
            symptoms: Symptom data
            ibd_type: IBD type ('crohn' or 'ulcerative_colitis')
            montreal_code: Optional Montreal classification code

        Returns:
            Tuple of (cluster_id, confidence)
        """
        if not self.is_loaded.get(ibd_type, False):
            raise RuntimeError(f"Models for {ibd_type} not loaded. Call load_models() first.")

        # Priority 1: Use Montreal classification if provided
        if montreal_code:
            cluster_id, confidence = self.get_cluster_from_montreal(montreal_code, ibd_type)
            if cluster_id is not None:
                return cluster_id, confidence

        # Priority 2: Infer from symptoms using KMeans
        # Extract clustering features from symptoms as dict
        clustering_features = {}
        for feature in self.cluster_meta[ibd_type]['features']:
            if feature == 'abdominal_pain':
                clustering_features[feature] = symptoms.get('abdominal_pain', 0) / 10.0
            elif feature == 'blood_in_stool':
                clustering_features[feature] = float(symptoms.get('blood_in_stool', False))
            elif feature == 'diarrhea':
                clustering_features[feature] = symptoms.get('diarrhea', 0) / 10.0
            elif feature == 'fatigue':
                clustering_features[feature] = symptoms.get('fatigue', 0) / 10.0
            elif feature == 'fever':
                clustering_features[feature] = float(symptoms.get('fever', False))
            elif feature == 'nausea':
                clustering_features[feature] = symptoms.get('nausea', 0) / 10.0

        # Create DataFrame with feature names to avoid sklearn warning
        import pandas as pd
        features_df = pd.DataFrame([clustering_features])

        # Scale features
        features_scaled = self.scaler[ibd_type].transform(features_df)

        # Predict cluster
        cluster_id = self.kmeans[ibd_type].predict(features_scaled)[0]

        # Calculate confidence (distance to nearest centroid)
        distances = self.kmeans[ibd_type].transform(features_scaled)[0]
        min_distance = distances[cluster_id]
        second_min = sorted(distances)[1]

        # Confidence: larger gap = higher confidence
        confidence = (second_min - min_distance) / (second_min + 0.001)
        confidence = min(max(confidence, 0.0), 1.0)

        logger.info(f"Inferred {ibd_type} cluster {cluster_id} with confidence {confidence:.3f}")

        return int(cluster_id), float(confidence)

    def extract_features(
        self,
        symptoms: Dict,
        demographics: Dict,
        history: Dict,
        temporal_features: Dict = None
    ):
        """
        Extract features for model prediction including derived features.

        Args:
            symptoms: Symptom data
            demographics: Demographic data
            history: Medical history data
            temporal_features: Optional temporal features from user history

        Returns:
            DataFrame with feature names (1 row, N columns)
        """
        import pandas as pd
        now = datetime.now()

        # Normalize symptom values (0-10 scale to 0-1)
        pain_norm = symptoms.get("abdominal_pain", 0) / 10.0
        diarrhea_norm = symptoms.get("diarrhea", 0) / 10.0
        fatigue_norm = symptoms.get("fatigue", 0) / 10.0
        nausea_norm = symptoms.get("nausea", 0) / 10.0
        fever_int = int(symptoms.get("fever", False))
        blood_int = int(symptoms.get("blood_in_stool", False))

        # Base features (original 13)
        feature_dict = {
            'abdominal_pain': pain_norm,
            'blood_in_stool': blood_int,
            'diarrhea': diarrhea_norm,
            'fatigue': fatigue_norm,
            'fever': fever_int,
            'nausea': nausea_norm,
            'age': demographics.get("age", 30.0),
            'gender': 1 if demographics.get("gender") == "M" else (2 if demographics.get("gender") == "F" else 0),
            'disease_duration_years': demographics.get("disease_duration_years", 0.0),
            'previous_flares': history.get("previous_flares", 0),
            'last_flare_days_ago': history.get("last_flare_days_ago", 365),
            'month': now.month,
            'day_of_week': now.weekday()
        }

        # ===== DERIVED FEATURES =====
        # These are calculated automatically from base features

        # 1. Symptom Aggregations
        ibd_type = demographics.get('ibd_type', 'crohn')
        if ibd_type == 'crohn':
            feature_dict['total_symptom_score'] = (
                pain_norm * 1.2 + diarrhea_norm * 1.3 + fatigue_norm * 1.0 +
                nausea_norm * 0.8 + blood_int * 2.0 + fever_int * 1.5
            )
        else:  # UC - blood and diarrhea weigh more
            feature_dict['total_symptom_score'] = (
                pain_norm * 1.0 + diarrhea_norm * 1.5 + fatigue_norm * 1.0 +
                nausea_norm * 0.8 + blood_int * 2.5 + fever_int * 1.5
            )

        feature_dict['gi_score'] = pain_norm + diarrhea_norm + nausea_norm + (blood_int * 2)
        feature_dict['systemic_score'] = fatigue_norm + (fever_int * 2)
        feature_dict['red_flag_score'] = (blood_int * 3) + (fever_int * 2) + (1 if pain_norm >= 0.7 else 0)
        feature_dict['symptom_count'] = sum([
            pain_norm > 0.2, diarrhea_norm > 0.2, fatigue_norm > 0.2,
            nausea_norm > 0.2, blood_int > 0, fever_int > 0
        ])

        # 2. History-derived features
        disease_duration = demographics.get("disease_duration_years", 1.0)
        previous_flares = history.get("previous_flares", 0)
        last_flare_days = history.get("last_flare_days_ago", 365)

        feature_dict['flare_frequency'] = previous_flares / max(disease_duration, 1)
        feature_dict['recency_score'] = 1 / (1 + last_flare_days / 30)
        feature_dict['disease_burden'] = disease_duration * 0.3 + previous_flares * 0.7
        feature_dict['young_longduration'] = int(
            demographics.get("age", 30) < 30 and disease_duration > 5
        )

        # 3. Interaction features
        feature_dict['pain_diarrhea_combo'] = pain_norm * diarrhea_norm
        feature_dict['blood_and_pain'] = int(blood_int == 1 and pain_norm >= 0.6)
        feature_dict['vulnerable_state'] = int(
            last_flare_days < 180 and feature_dict['total_symptom_score'] > 4.0
        )

        # Symptom severity category
        if feature_dict['total_symptom_score'] < 3.0:
            feature_dict['symptom_severity_category'] = 0  # Mild
        elif feature_dict['total_symptom_score'] < 6.0:
            feature_dict['symptom_severity_category'] = 1  # Moderate
        else:
            feature_dict['symptom_severity_category'] = 2  # Severe

        feature_dict['gi_dominant'] = int(feature_dict['gi_score'] > feature_dict['systemic_score'] * 1.5)

        # 4. Temporal features (use provided values or fallback to current)
        if temporal_features:
            feature_dict['pain_trend_7d'] = temporal_features.get('pain_trend_7d', pain_norm)
            feature_dict['diarrhea_trend_7d'] = temporal_features.get('diarrhea_trend_7d', diarrhea_norm)
            feature_dict['fatigue_trend_7d'] = temporal_features.get('fatigue_trend_7d', fatigue_norm)
            feature_dict['symptom_volatility_7d'] = temporal_features.get('symptom_volatility_7d', 0.0)
            feature_dict['symptom_change_rate'] = temporal_features.get('symptom_change_rate', 0.0)
            feature_dict['days_since_low_symptoms'] = temporal_features.get('days_since_low_symptoms', 0)
        else:
            # Fallback: use current symptom values as trends
            feature_dict['pain_trend_7d'] = pain_norm
            feature_dict['diarrhea_trend_7d'] = diarrhea_norm
            feature_dict['fatigue_trend_7d'] = fatigue_norm
            feature_dict['symptom_volatility_7d'] = 0.0
            feature_dict['symptom_change_rate'] = 0.0
            feature_dict['days_since_low_symptoms'] = 0

        # Add is_bad_day for consistency
        feature_dict['is_bad_day'] = int(feature_dict['total_symptom_score'] > 3.0)

        # CRITICAL: Ensure features are in the exact order as training CSV
        # Order: 13 base features + 21 derived features = 34 total
        feature_order = [
            # Base features (13) - from ml_dataset.csv
            'abdominal_pain', 'blood_in_stool', 'diarrhea', 'fatigue', 'fever', 'nausea',  # symptoms (6)
            'age', 'gender',  # demographics (2)
            'disease_duration_years', 'previous_flares', 'last_flare_days_ago',  # history (3)
            'month', 'day_of_week',  # temporal (2)

            # Derived features (21) - from notebook 03
            # Symptom aggregations (5)
            'total_symptom_score', 'gi_score', 'systemic_score', 'red_flag_score', 'symptom_count',

            # Temporal features (7)
            'pain_trend_7d', 'diarrhea_trend_7d', 'fatigue_trend_7d',
            'symptom_volatility_7d', 'symptom_change_rate', 'is_bad_day', 'days_since_low_symptoms',

            # History features (4)
            'flare_frequency', 'recency_score', 'disease_burden', 'young_longduration',

            # Interaction features (5)
            'pain_diarrhea_combo', 'blood_and_pain', 'vulnerable_state',
            'symptom_severity_category', 'gi_dominant'
        ]

        # Create DataFrame with columns in correct order
        df = pd.DataFrame([feature_dict])

        # Reorder columns to match training data (only include columns that exist)
        available_features = [col for col in feature_order if col in df.columns]
        df = df[available_features]

        return df

    def predict(
        self,
        symptoms: Dict,
        demographics: Dict,
        history: Dict,
        temporal_features: Dict = None,
        daily_records: List[Dict] = None
    ) -> Tuple[str, float, float, List[str], Dict[str, float], int, float]:
        """
        Make prediction using cluster-specific model for the patient's IBD type.

        Args:
            symptoms: Symptom data
            demographics: Demographic data (must include ibd_type)
            history: Medical history data
            temporal_features: DEPRECATED - Optional temporal features (for backward compatibility)
            daily_records: Optional list of daily symptom records for temporal feature calculation

        Returns:
            Tuple of (risk_level, probability, confidence, contributors,
                     all_probabilities, cluster_id, cluster_confidence, model_source, ibd_type, montreal_code)
        """
        # Extract IBD type and Montreal classification
        ibd_type = demographics.get('ibd_type', 'crohn')
        montreal_code = demographics.get('montreal_location', None)

        # Validate IBD type
        if ibd_type not in ['crohn', 'ulcerative_colitis']:
            logger.warning(f"Invalid ibd_type '{ibd_type}', defaulting to 'crohn'")
            ibd_type = 'crohn'

        # Check if models are loaded for this IBD type
        if not self.is_loaded.get(ibd_type, False):
            raise RuntimeError(f"Models for {ibd_type} not loaded. Call load_models() first.")

        try:
            # Calculate temporal features from daily_records if provided
            if daily_records and len(daily_records) >= 7:
                temporal_features = calculate_temporal_features_from_records(
                    daily_records=daily_records,
                    ibd_type=ibd_type
                )
                logger.info(f"Calculated temporal features from {len(daily_records)} daily records")
            elif temporal_features:
                logger.info("Using provided temporal features (backward compatibility mode)")
            else:
                logger.info("No temporal features available, using fallback values")

            # 1. Infer cluster (prioritizes Montreal classification if provided)
            cluster_id, cluster_confidence = self.infer_cluster(
                symptoms=symptoms,
                ibd_type=ibd_type,
                montreal_code=montreal_code
            )

            # 2. Get cluster-specific model for this IBD type
            model_source = "cluster_specific"  # Track which model is used

            if cluster_id not in self.cluster_models[ibd_type]:
                logger.warning(f"Cluster {cluster_id} not found for {ibd_type}, falling back to global model")
                # Use global model as fallback (trained on all clusters)
                if self.global_models[ibd_type] is not None:
                    model = self.global_models[ibd_type]
                    cluster_confidence = 0.5  # Lower confidence when using fallback
                    model_source = "global_fallback"
                else:
                    # Last resort: use any available cluster model
                    logger.warning(f"Global model not available, using first available cluster model")
                    cluster_id = list(self.cluster_models[ibd_type].keys())[0]
                    model = self.cluster_models[ibd_type][cluster_id]
                    model_source = "cluster_specific"
            else:
                model = self.cluster_models[ibd_type][cluster_id]
                model_source = "cluster_specific"

            # 3. Extract features for prediction (including derived features)
            features = self.extract_features(symptoms, demographics, history, temporal_features)

            # 4. Predict
            prediction = model.predict(features)[0]
            probabilities = model.predict_proba(features)[0]

            # Get class labels
            class_labels = model.classes_
            risk_level = str(prediction).lower()

            # Probability for predicted class
            pred_idx = list(class_labels).index(prediction)
            probability = float(probabilities[pred_idx])

            # Confidence (gap between top 2)
            sorted_probs = sorted(probabilities, reverse=True)
            confidence = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else 1.0

            # All probabilities
            all_probs = {str(cls).lower(): float(prob) for cls, prob in zip(class_labels, probabilities)}

            # Contributors
            contributors = self._identify_contributors(symptoms, history)

            logger.info(f"Cluster-stratified prediction: ibd_type={ibd_type}, cluster={cluster_id}, "
                       f"risk={risk_level}, prob={probability:.2f}, model_source={model_source}")

            return (risk_level, probability, confidence, contributors,
                   all_probs, cluster_id, cluster_confidence, model_source, ibd_type, montreal_code)

        except Exception as e:
            logger.error(f"Error in cluster-stratified prediction: {e}", exc_info=True)
            raise

    def _identify_contributors(self, symptoms: Dict, history: Dict) -> List[str]:
        """
        Identify top contributing factors.

        Args:
            symptoms: Symptom data
            history: Medical history data

        Returns:
            List of top contributing factors
        """
        contributors = []

        # Check symptoms
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

        # Default if empty
        if not contributors:
            contributors.append("general_symptom_pattern")

        return contributors[:3]


class TemporalPredictor:
    """
    Temporal predictor for IBD flare risk prediction.

    Uses temporal models trained on 7-day symptom trends and medication data.
    Predicts flare risk at T+7 days based on current symptoms and history.

    Features (27 total):
    - Medications (5): biologics, immunosuppressants, corticosteroids, aminosalicylates, total_meds
    - Symptom trends (10): 7-day trends and volatility for pain, diarrhea, fatigue, blood
    - History (6): disease_duration, cumulative_flare_days, previous_flares, days_since_last_flare, flare_frequency, recency_score
    - Demographics (6): age, month, day_of_week, is_weekend, gender_M, gender_O
    """

    def __init__(self, models_dir: str = "models"):
        """
        Initialize temporal predictor.

        Args:
            models_dir: Directory containing temporal models
        """
        self.models_dir = Path(models_dir)
        self.models = {
            'crohn': None,
            'ulcerative_colitis': None
        }
        self.metadata = {
            'crohn': None,
            'ulcerative_colitis': None
        }
        self.is_loaded = {
            'crohn': False,
            'ulcerative_colitis': False
        }

    def load_models(self):
        """Load temporal models for both Crohn and UC."""
        any_loaded = False

        for ibd_type in ['crohn', 'ulcerative_colitis']:
            # Determine directory (crohn/ or cu/)
            ibd_dir = self.models_dir / ('crohn' if ibd_type == 'crohn' else 'cu')

            try:
                # Load model
                model_path = ibd_dir / 'rf_temporal_classifier.pkl'
                if not model_path.exists():
                    logger.warning(f"Temporal model not found: {model_path}")
                    continue

                with open(model_path, 'rb') as f:
                    self.models[ibd_type] = pickle.load(f)

                # Load metadata
                metadata_path = ibd_dir / 'model_metadata_temporal.json'
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        self.metadata[ibd_type] = json.load(f)

                self.is_loaded[ibd_type] = True
                any_loaded = True
                logger.info(f"✓ Loaded temporal model for {ibd_type} (Test Acc: {self.metadata[ibd_type].get('test_accuracy', 0):.1%})")

            except Exception as e:
                logger.error(f"Failed to load temporal model for {ibd_type}: {e}")
                continue

        return any_loaded

    def extract_medication_features(self, medications: List[str]) -> dict:
        """
        Extract medication features from medication list.

        Args:
            medications: List of medication names

        Returns:
            Dict with medication features
        """
        # Normalize to lowercase for matching
        meds_lower = [med.lower() for med in medications]

        # Define medication categories (from notebook 02b)
        biologics = ['infliximab', 'remicade', 'humira', 'adalimumab', 'stelara',
                    'ustekinumab', 'entyvio', 'vedolizumab', 'simponi', 'golimumab',
                    'cimzia', 'certolizumab']

        immunosuppressants = ['azathioprine', 'imuran', '6-mp', 'mercaptopurine',
                             'methotrexate', 'purinethol', 'cyclosporine']

        corticosteroids = ['prednisone', 'prednisolone', 'budesonide', 'entocort',
                          'hydrocortisone', 'methylprednisolone']

        aminosalicylates = ['mesalamine', 'mesalazine', 'asacol', 'pentasa', 'lialda',
                           'sulfasalazine', 'azulfidine', 'balsalazide', 'colazal']

        # Count active medications in each category
        biologics_active = sum(1 for med in meds_lower if any(b in med for b in biologics))
        immunosuppressants_active = sum(1 for med in meds_lower if any(i in med for i in immunosuppressants))
        corticosteroids_active = sum(1 for med in meds_lower if any(c in med for c in corticosteroids))
        aminosalicylates_active = sum(1 for med in meds_lower if any(a in med for a in aminosalicylates))

        total_ibd_meds = biologics_active + immunosuppressants_active + corticosteroids_active + aminosalicylates_active

        return {
            'biologics_active': biologics_active,
            'immunosuppressants_active': immunosuppressants_active,
            'corticosteroids_active': corticosteroids_active,
            'aminosalicylates_active': aminosalicylates_active,
            'total_ibd_meds': total_ibd_meds
        }

    def calculate_symptom_trends(self, daily_records: List[Dict]) -> dict:
        """
        Calculate 7-day symptom trends and volatility from daily records.

        Args:
            daily_records: List of daily symptom records (dicts with 'date' and 'symptoms')

        Returns:
            Dict with trend features
        """
        if len(daily_records) < 7:
            logger.warning(f"Insufficient daily records for trends: {len(daily_records)} < 7. Using fallback values.")
            # Return fallback values (zeros/current symptoms)
            if daily_records:
                current = daily_records[-1]['symptoms']
                return {
                    'abdominal_pain_trend_7d': current.get('abdominal_pain', 0) / 10.0,
                    'abdominal_pain_volatility_7d': 0.0,
                    'diarrhea_trend_7d': current.get('diarrhea', 0) / 10.0,
                    'diarrhea_volatility_7d': 0.0,
                    'fatigue_trend_7d': current.get('fatigue', 0) / 10.0,
                    'fatigue_volatility_7d': 0.0,
                    'blood_in_stool_trend_7d': float(current.get('blood_in_stool', False)),
                    'blood_in_stool_volatility_7d': 0.0,
                    'symptom_count_avg_7d': 0.0,
                    'red_flag_trend_7d': 0.0
                }
            else:
                return {
                    'abdominal_pain_trend_7d': 0.0,
                    'abdominal_pain_volatility_7d': 0.0,
                    'diarrhea_trend_7d': 0.0,
                    'diarrhea_volatility_7d': 0.0,
                    'fatigue_trend_7d': 0.0,
                    'fatigue_volatility_7d': 0.0,
                    'blood_in_stool_trend_7d': 0.0,
                    'blood_in_stool_volatility_7d': 0.0,
                    'symptom_count_avg_7d': 0.0,
                    'red_flag_trend_7d': 0.0
                }

        # Sort by date
        sorted_records = sorted(daily_records, key=lambda x: x['date'])
        last_7 = sorted_records[-7:]

        # Extract symptom values
        pain_values = [r['symptoms'].get('abdominal_pain', 0) / 10.0 for r in last_7]
        diarrhea_values = [r['symptoms'].get('diarrhea', 0) / 10.0 for r in last_7]
        fatigue_values = [r['symptoms'].get('fatigue', 0) / 10.0 for r in last_7]
        blood_values = [float(r['symptoms'].get('blood_in_stool', False)) for r in last_7]

        # Calculate symptom counts and red flags
        symptom_counts = []
        red_flags = []

        for record in last_7:
            symptoms = record['symptoms']
            count = sum([
                symptoms.get('abdominal_pain', 0) > 0,
                symptoms.get('diarrhea', 0) > 0,
                symptoms.get('fatigue', 0) > 0,
                symptoms.get('nausea', 0) > 0,
                symptoms.get('blood_in_stool', False),
                symptoms.get('fever', False)
            ])
            symptom_counts.append(count)

            # Red flags: blood, fever, severe pain
            red_flag = (
                int(symptoms.get('blood_in_stool', False)) +
                int(symptoms.get('fever', False)) +
                int(symptoms.get('abdominal_pain', 0) >= 7)
            )
            red_flags.append(red_flag)

        return {
            'abdominal_pain_trend_7d': float(np.mean(pain_values)),
            'abdominal_pain_volatility_7d': float(np.std(pain_values)),
            'diarrhea_trend_7d': float(np.mean(diarrhea_values)),
            'diarrhea_volatility_7d': float(np.std(diarrhea_values)),
            'fatigue_trend_7d': float(np.mean(fatigue_values)),
            'fatigue_volatility_7d': float(np.std(fatigue_values)),
            'blood_in_stool_trend_7d': float(np.mean(blood_values)),
            'blood_in_stool_volatility_7d': float(np.std(blood_values)),
            'symptom_count_avg_7d': float(np.mean(symptom_counts)),
            'red_flag_trend_7d': float(np.mean(red_flags))
        }

    def extract_features(
        self,
        symptoms: Dict,
        demographics: Dict,
        history: Dict,
        daily_records: List[Dict] = None
    ) -> 'pd.DataFrame':
        """
        Extract all 27 features required by temporal models.

        Args:
            symptoms: Current symptom data
            demographics: Demographic data
            history: Medical history data
            daily_records: Optional list of daily records for trend calculation

        Returns:
            DataFrame with 27 features in correct order
        """
        import pandas as pd
        from datetime import datetime

        now = datetime.now()

        # 1. Medication features (5)
        medications = history.get('medications', [])
        med_features = self.extract_medication_features(medications)

        # 2. Symptom trends (10) - from daily_records if available
        if daily_records and len(daily_records) > 0:
            trend_features = self.calculate_symptom_trends(daily_records)
        else:
            # Fallback: use current symptoms as trends
            trend_features = {
                'abdominal_pain_trend_7d': symptoms.get('abdominal_pain', 0) / 10.0,
                'abdominal_pain_volatility_7d': 0.0,
                'diarrhea_trend_7d': symptoms.get('diarrhea', 0) / 10.0,
                'diarrhea_volatility_7d': 0.0,
                'fatigue_trend_7d': symptoms.get('fatigue', 0) / 10.0,
                'fatigue_volatility_7d': 0.0,
                'blood_in_stool_trend_7d': float(symptoms.get('blood_in_stool', False)),
                'blood_in_stool_volatility_7d': 0.0,
                'symptom_count_avg_7d': 0.0,
                'red_flag_trend_7d': 0.0
            }

        # 3. History features (6)
        disease_duration = demographics.get('disease_duration_years', 0.0)
        previous_flares = history.get('previous_flares', 0)
        days_since_last_flare = history.get('last_flare_days_ago', 365)
        cumulative_flare_days = history.get('cumulative_flare_days', 0)

        flare_frequency = previous_flares / max(disease_duration, 1.0)
        recency_score = 1 / (1 + days_since_last_flare / 30)

        # 4. Demographics and temporal (6)
        age = demographics.get('age', 30)
        gender = demographics.get('gender', 'O')

        # Combine all features
        features = {
            # Medications (5)
            'biologics_active': med_features['biologics_active'],
            'immunosuppressants_active': med_features['immunosuppressants_active'],
            'corticosteroids_active': med_features['corticosteroids_active'],
            'aminosalicylates_active': med_features['aminosalicylates_active'],
            'total_ibd_meds': med_features['total_ibd_meds'],

            # Symptom trends (10)
            'abdominal_pain_trend_7d': trend_features['abdominal_pain_trend_7d'],
            'abdominal_pain_volatility_7d': trend_features['abdominal_pain_volatility_7d'],
            'diarrhea_trend_7d': trend_features['diarrhea_trend_7d'],
            'diarrhea_volatility_7d': trend_features['diarrhea_volatility_7d'],
            'fatigue_trend_7d': trend_features['fatigue_trend_7d'],
            'fatigue_volatility_7d': trend_features['fatigue_volatility_7d'],
            'blood_in_stool_trend_7d': trend_features['blood_in_stool_trend_7d'],
            'blood_in_stool_volatility_7d': trend_features['blood_in_stool_volatility_7d'],
            'symptom_count_avg_7d': trend_features['symptom_count_avg_7d'],
            'red_flag_trend_7d': trend_features['red_flag_trend_7d'],

            # History (6)
            'disease_duration_years': disease_duration,
            'cumulative_flare_days': cumulative_flare_days,
            'previous_flares': previous_flares,
            'days_since_last_flare': days_since_last_flare,
            'flare_frequency': flare_frequency,
            'recency_score': recency_score,

            # Demographics + temporal (6)
            'age': age,
            'month': now.month,
            'day_of_week': now.weekday(),
            'is_weekend': int(now.weekday() >= 5),
            'gender_M': int(gender == 'M'),
            'gender_O': int(gender == 'O')
        }

        # Return as DataFrame with features in correct order (from metadata)
        feature_order = [
            'biologics_active', 'immunosuppressants_active', 'corticosteroids_active',
            'aminosalicylates_active', 'total_ibd_meds',
            'abdominal_pain_trend_7d', 'abdominal_pain_volatility_7d',
            'diarrhea_trend_7d', 'diarrhea_volatility_7d',
            'fatigue_trend_7d', 'fatigue_volatility_7d',
            'blood_in_stool_trend_7d', 'blood_in_stool_volatility_7d',
            'symptom_count_avg_7d', 'red_flag_trend_7d',
            'disease_duration_years', 'cumulative_flare_days', 'previous_flares',
            'days_since_last_flare', 'flare_frequency', 'recency_score',
            'age', 'month', 'day_of_week', 'is_weekend', 'gender_M', 'gender_O'
        ]

        df = pd.DataFrame([features])
        return df[feature_order]

    def predict(
        self,
        symptoms: Dict,
        demographics: Dict,
        history: Dict,
        daily_records: List[Dict] = None
    ) -> Tuple[str, float, float, List[str], Dict[str, float], None, None, str, str, None]:
        """
        Make temporal prediction (T+7 days).

        Args:
            symptoms: Current symptom data
            demographics: Demographic data (must include ibd_type)
            history: Medical history data (must include medications)
            daily_records: Optional list of daily records for trend calculation

        Returns:
            Tuple of (risk_level, probability, confidence, contributors,
                     all_probabilities, None, None, model_source, ibd_type, None)
            Note: Returns None for cluster fields since temporal models don't use clusters
        """
        # Get IBD type
        ibd_type = demographics.get('ibd_type', 'crohn')
        if ibd_type not in ['crohn', 'ulcerative_colitis']:
            logger.warning(f"Invalid ibd_type '{ibd_type}', defaulting to 'crohn'")
            ibd_type = 'crohn'

        # Check if model is loaded
        if not self.is_loaded.get(ibd_type, False):
            raise RuntimeError(f"Temporal model for {ibd_type} not loaded. Call load_models() first.")

        try:
            # Extract features
            features = self.extract_features(symptoms, demographics, history, daily_records)

            # Get model
            model = self.models[ibd_type]

            # Predict
            prediction = model.predict(features)[0]
            probabilities = model.predict_proba(features)[0]

            # Get class labels
            class_labels = model.classes_
            risk_level = str(prediction).lower()

            # Probability for predicted class
            pred_idx = list(class_labels).index(prediction)
            probability = float(probabilities[pred_idx])

            # Confidence (gap between top 2)
            sorted_probs = sorted(probabilities, reverse=True)
            confidence = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else 1.0

            # All probabilities
            all_probs = {str(cls).lower(): float(prob) for cls, prob in zip(class_labels, probabilities)}

            # Identify contributors
            contributors = self._identify_contributors(symptoms, history, features)

            logger.info(f"Temporal prediction (T+7): ibd_type={ibd_type}, risk={risk_level}, "
                       f"prob={probability:.2f}, confidence={confidence:.2f}")

            # Return 10-tuple (cluster fields are None for temporal model)
            return (risk_level, probability, confidence, contributors, all_probs,
                   None, None, "temporal", ibd_type, None)

        except Exception as e:
            logger.error(f"Error in temporal prediction: {e}", exc_info=True)
            raise

    def _identify_contributors(self, symptoms: Dict, history: Dict, features: 'pd.DataFrame') -> List[str]:
        """
        Identify top contributing factors based on feature importance.

        Top features from temporal models:
        1. Medication features (biologics, corticosteroids)
        2. Symptom trends (abdominal_pain_trend_7d, diarrhea_trend_7d)
        3. History (days_since_last_flare, recency_score)

        Args:
            symptoms: Symptom data
            history: Medical history data
            features: Extracted features DataFrame

        Returns:
            List of top 3 contributors
        """
        contributors = []

        # Check medications (very important in temporal models)
        if features['biologics_active'].iloc[0] > 0:
            contributors.append("biologics_treatment")
        if features['corticosteroids_active'].iloc[0] > 0:
            contributors.append("corticosteroids_active")

        # Check symptom trends
        if features['abdominal_pain_trend_7d'].iloc[0] >= 0.6:
            contributors.append("high_pain_trend")
        if features['diarrhea_trend_7d'].iloc[0] >= 0.6:
            contributors.append("high_diarrhea_trend")
        if features['blood_in_stool_trend_7d'].iloc[0] > 0.5:
            contributors.append("blood_in_stool")
        if features['red_flag_trend_7d'].iloc[0] >= 1.0:
            contributors.append("red_flag_symptoms")

        # Check history
        if history.get('days_since_last_flare', 365) < 90:
            contributors.append("recent_flare_history")
        if history.get('previous_flares', 0) > 3:
            contributors.append("high_flare_frequency")

        # Default
        if not contributors:
            contributors.append("general_symptom_pattern")

        return contributors[:3]


def get_predictor(use_clusters: bool = True, use_temporal: bool = True) -> Optional[CrohnPredictor]:
    """
    Factory function to get the appropriate predictor.

    Priority order:
    1. Temporal predictor (T+7 days, medications, symptom trends) - DEFAULT
    2. Cluster-stratified predictor (if use_clusters=True and temporal not available)
    3. Global model (fallback)

    Args:
        use_clusters: If True and temporal not available, use cluster-stratified predictor.
        use_temporal: If True, prioritize temporal predictor. Default: True.

    Returns:
        Initialized predictor (loaded) or None if loading fails
    """
    # Priority 1: Try temporal predictor (NEW - default)
    if use_temporal:
        predictor = TemporalPredictor()
        if predictor.load_models():
            logger.info("Using temporal predictor (T+7 prediction with medications)")
            return predictor
        else:
            logger.warning("Temporal models not available, falling back...")

    # Priority 2: Try cluster-stratified predictor
    if use_clusters:
        predictor = ClusterStratifiedPredictor()
        if predictor.load_models():
            logger.info("Using cluster-stratified predictor")
            return predictor
        else:
            logger.warning("Cluster models not available, falling back to global model")

    # Priority 3: Use global model
    predictor = CrohnPredictor()
    if predictor.load_model():
        logger.info("Using global predictor")
        return predictor

    logger.error("Failed to load any predictor")
    return None

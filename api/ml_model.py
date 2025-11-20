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
        history: Dict
    ):
        """
        Extract features for model prediction (same as CrohnPredictor).

        Args:
            symptoms: Symptom data
            demographics: Demographic data
            history: Medical history data

        Returns:
            DataFrame with feature names (1 row, 13 columns)
        """
        import pandas as pd
        now = datetime.now()

        # Feature names must match training data
        feature_dict = {
            'abdominal_pain': symptoms.get("abdominal_pain", 0) / 10.0,
            'blood_in_stool': int(symptoms.get("blood_in_stool", False)),
            'diarrhea': symptoms.get("diarrhea", 0) / 10.0,
            'fatigue': symptoms.get("fatigue", 0) / 10.0,
            'fever': int(symptoms.get("fever", False)),
            'nausea': symptoms.get("nausea", 0) / 10.0,
            'age': demographics.get("age", 30.0),
            'gender': 1 if demographics.get("gender") == "M" else (2 if demographics.get("gender") == "F" else 0),
            'disease_duration_years': demographics.get("disease_duration_years", 0.0),
            'previous_flares': history.get("previous_flares", 0),
            'last_flare_days_ago': history.get("last_flare_days_ago", 365),
            'month': now.month,
            'day_of_week': now.weekday()
        }

        return pd.DataFrame([feature_dict])

    def predict(
        self,
        symptoms: Dict,
        demographics: Dict,
        history: Dict
    ) -> Tuple[str, float, float, List[str], Dict[str, float], int, float]:
        """
        Make prediction using cluster-specific model for the patient's IBD type.

        Args:
            symptoms: Symptom data
            demographics: Demographic data (must include ibd_type)
            history: Medical history data

        Returns:
            Tuple of (risk_level, probability, confidence, contributors,
                     all_probabilities, cluster_id, cluster_confidence)
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
            # 1. Infer cluster (prioritizes Montreal classification if provided)
            cluster_id, cluster_confidence = self.infer_cluster(
                symptoms=symptoms,
                ibd_type=ibd_type,
                montreal_code=montreal_code
            )

            # 2. Get cluster-specific model for this IBD type
            if cluster_id not in self.cluster_models[ibd_type]:
                logger.warning(f"Cluster {cluster_id} not found for {ibd_type}, falling back to global model")
                # Use global model as fallback (trained on all clusters)
                if self.global_models[ibd_type] is not None:
                    model = self.global_models[ibd_type]
                    cluster_confidence = 0.5  # Lower confidence when using fallback
                else:
                    # Last resort: use any available cluster model
                    logger.warning(f"Global model not available, using first available cluster model")
                    cluster_id = list(self.cluster_models[ibd_type].keys())[0]
                    model = self.cluster_models[ibd_type][cluster_id]
            else:
                model = self.cluster_models[ibd_type][cluster_id]

            # 3. Extract features for prediction
            features = self.extract_features(symptoms, demographics, history)

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
                       f"risk={risk_level}, prob={probability:.2f}")

            return (risk_level, probability, confidence, contributors,
                   all_probs, cluster_id, cluster_confidence)

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


def get_predictor(use_clusters: bool = True) -> Optional[CrohnPredictor]:
    """
    Factory function to get the appropriate predictor.

    Args:
        use_clusters: If True, use cluster-stratified predictor if available.
                     Falls back to global model if cluster models not found.

    Returns:
        Initialized predictor (loaded) or None if loading fails
    """
    if use_clusters:
        # Try cluster-stratified predictor first
        predictor = ClusterStratifiedPredictor()
        if predictor.load_models():
            logger.info("Using cluster-stratified predictor")
            return predictor
        else:
            logger.warning("Cluster models not available, falling back to global model")

    # Use global model
    predictor = CrohnPredictor()
    if predictor.load_model():
        logger.info("Using global predictor")
        return predictor

    logger.error("Failed to load any predictor")
    return None

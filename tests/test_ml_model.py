"""Tests unitarios para el módulo ML (sin cargar modelos)."""
import pytest
from api.ml_model import ClusterStratifiedPredictor


class TestFeatureExtraction:
    """Tests para la extracción de features (sin cargar modelo)."""

    @pytest.fixture
    def predictor(self):
        """Fixture que crea un predictor SIN cargar modelos."""
        # No llamamos a _load_models, solo queremos testear extract_features
        predictor = ClusterStratifiedPredictor.__new__(ClusterStratifiedPredictor)
        predictor.ibd_type = "crohn"
        predictor.models_dir = "models/crohn"
        predictor.models = {}  # Vacío, no cargamos modelos
        predictor.cluster_model = None
        predictor.cluster_scaler = None
        predictor.global_model = None
        predictor.metadata = {}
        return predictor

    def test_extract_features_returns_dataframe(self, predictor):
        """Test que extract_features devuelve un DataFrame válido."""
        symptoms = {
            "abdominal_pain": 7,
            "blood_in_stool": False,
            "diarrhea": 6,
            "fatigue": 5,
            "fever": False,
            "nausea": 3
        }
        demographics = {
            "age": 32,
            "gender": "F",
            "disease_duration_years": 5,
            "bmi": 22.5
        }
        history = {
            "previous_flares": 3,
            "last_flare_days_ago": 120
        }

        df = predictor.extract_features(symptoms, demographics, history)

        # Verificar que devuelve un DataFrame
        import pandas as pd
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1  # Una fila
        assert len(df.columns) > 0  # Tiene columnas

    def test_extract_features_has_required_features(self, predictor):
        """Test que el DataFrame tiene las features esperadas."""
        symptoms = {"abdominal_pain": 5, "diarrhea": 4, "fatigue": 3, "fever": False}
        demographics = {"age": 30, "gender": "F", "disease_duration_years": 3}
        history = {"previous_flares": 2, "last_flare_days_ago": 100}

        df = predictor.extract_features(symptoms, demographics, history)

        # Verificar que tiene las features base
        expected_base_features = [
            'abdominal_pain', 'diarrhea', 'fatigue', 'nausea',
            'age', 'gender', 'disease_duration_years'
        ]
        for feature in expected_base_features:
            assert feature in df.columns, f"Missing feature: {feature}"

        # Verificar que tiene features derivadas (algunas)
        expected_derived = [
            'total_symptom_score', 'gi_score', 'systemic_score',
            'flare_frequency', 'recency_score'
        ]
        for feature in expected_derived:
            assert feature in df.columns, f"Missing derived feature: {feature}"

    def test_extract_features_with_temporal_features(self, predictor):
        """Test con temporal_features proporcionados."""
        symptoms = {"abdominal_pain": 8, "diarrhea": 7, "fatigue": 6, "fever": False}
        demographics = {"age": 35, "gender": "M", "disease_duration_years": 4}
        history = {"previous_flares": 4, "last_flare_days_ago": 60}
        temporal = {
            "pain_trend_7d": 0.15,
            "diarrhea_trend_7d": 0.10,
            "fatigue_trend_7d": 0.05,
            "symptom_volatility_7d": 1.2,
            "symptom_change_rate": 0.08,
            "days_since_low_symptoms": 5
        }

        df = predictor.extract_features(symptoms, demographics, history, temporal)

        # Verificar que las features temporales están presentes
        assert 'pain_trend_7d' in df.columns
        assert 'diarrhea_trend_7d' in df.columns
        assert 'symptom_volatility_7d' in df.columns

    def test_extract_features_handles_different_genders(self, predictor):
        """Test que maneja diferentes valores de género."""
        symptoms = {"abdominal_pain": 5, "diarrhea": 4, "fatigue": 3, "fever": False}
        history = {"previous_flares": 2, "last_flare_days_ago": 100}

        for gender in ["M", "F", "O"]:
            demographics = {"age": 30, "gender": gender, "disease_duration_years": 3}
            df = predictor.extract_features(symptoms, demographics, history)

            assert 'gender' in df.columns
            # Simplemente verificamos que no falla, sin verificar el valor exacto
            assert df['gender'].iloc[0] is not None

    def test_extract_features_correct_feature_count(self, predictor):
        """Test que genera el número correcto de features (34)."""
        symptoms = {"abdominal_pain": 5, "diarrhea": 4, "fatigue": 3, "fever": False}
        demographics = {"age": 30, "gender": "F", "disease_duration_years": 3}
        history = {"previous_flares": 2, "last_flare_days_ago": 100}

        df = predictor.extract_features(symptoms, demographics, history)

        # Debe tener 34 columnas (13 base + 21 derivadas)
        assert len(df.columns) == 34, f"Expected 34 features, got {len(df.columns)}"

    def test_extract_features_no_missing_values(self, predictor):
        """Test que no hay valores nulos en las features."""
        symptoms = {"abdominal_pain": 5, "diarrhea": 4, "fatigue": 3, "fever": False}
        demographics = {"age": 30, "gender": "F", "disease_duration_years": 3}
        history = {"previous_flares": 2, "last_flare_days_ago": 100}

        df = predictor.extract_features(symptoms, demographics, history)

        # No debería haber NaNs
        assert not df.isnull().any().any(), "Found null values in features"

    def test_extract_features_numeric_types(self, predictor):
        """Test que todas las features son numéricas."""
        symptoms = {"abdominal_pain": 5, "diarrhea": 4, "fatigue": 3, "fever": False}
        demographics = {"age": 30, "gender": "F", "disease_duration_years": 3}
        history = {"previous_flares": 2, "last_flare_days_ago": 100}

        df = predictor.extract_features(symptoms, demographics, history)

        # Todas las columnas deben ser numéricas
        import numpy as np
        for col in df.columns:
            assert np.issubdtype(df[col].dtype, np.number), \
                f"Feature {col} is not numeric: {df[col].dtype}"

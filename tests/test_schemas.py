"""Tests unitarios para validación de schemas Pydantic."""
import pytest
from pydantic import ValidationError

from api.schemas import (
    Symptoms,
    Demographics,
    MedicalHistory,
    PredictionRequest,
    FlareRiskPrediction,
)


class TestSymptoms:
    """Tests para el schema de síntomas."""

    def test_valid_symptoms(self):
        """Test con síntomas válidos."""
        symptoms = Symptoms(
            abdominal_pain=5,
            blood_in_stool=False,
            diarrhea=3,
            fatigue=7,
            fever=False,
            nausea=2
        )
        assert symptoms.abdominal_pain == 5
        assert symptoms.diarrhea == 3
        assert symptoms.fatigue == 7

    def test_symptoms_range_validation(self):
        """Test que los valores están en rango 0-10."""
        # Valor negativo debe fallar
        with pytest.raises(ValidationError):
            Symptoms(
                abdominal_pain=-1,
                blood_in_stool=False,
                diarrhea=5,
                fatigue=5,
                fever=False,
                nausea=0
            )

        # Valor mayor a 10 debe fallar
        with pytest.raises(ValidationError):
            Symptoms(
                abdominal_pain=11,
                blood_in_stool=False,
                diarrhea=5,
                fatigue=5,
                fever=False,
                nausea=0
            )

    def test_symptoms_with_defaults(self):
        """Test que los valores por defecto funcionan."""
        symptoms = Symptoms(
            abdominal_pain=5,
            diarrhea=3,
            fatigue=7,
            fever=False
        )
        # blood_in_stool y nausea tienen defaults
        assert symptoms.blood_in_stool is False
        assert symptoms.nausea == 0


class TestDemographics:
    """Tests para el schema de demografía."""

    def test_valid_demographics(self):
        """Test con demografía válida."""
        demo = Demographics(
            age=32,
            gender="F",
            disease_duration_years=5,
            bmi=22.5,
            ibd_type="crohn",
            montreal_location="L3"
        )
        assert demo.age == 32
        assert demo.gender == "F"
        assert demo.ibd_type == "crohn"

    def test_invalid_gender(self):
        """Test que género inválido falla."""
        with pytest.raises(ValidationError):
            Demographics(
                age=30,
                gender="X",  # Debe ser M, F, o O
                disease_duration_years=3,
                ibd_type="crohn",
                montreal_location="L1"
            )

    def test_invalid_ibd_type(self):
        """Test que tipo de EII inválido falla."""
        with pytest.raises(ValidationError):
            Demographics(
                age=30,
                gender="M",
                disease_duration_years=3,
                ibd_type="invalid",  # Debe ser crohn o ulcerative_colitis
                montreal_location="L1"
            )

    def test_montreal_validation_crohn(self):
        """Test que Montreal para Crohn sea L1-L4."""
        # Válido
        demo = Demographics(
            age=30,
            gender="M",
            disease_duration_years=3,
            ibd_type="crohn",
            montreal_location="L1"
        )
        assert demo.montreal_location == "L1"

        # Inválido para Crohn (E1 es para CU)
        with pytest.raises(ValidationError):
            Demographics(
                age=30,
                gender="M",
                disease_duration_years=3,
                ibd_type="crohn",
                montreal_location="E1"
            )

    def test_montreal_validation_cu(self):
        """Test que Montreal para CU sea E1-E3."""
        # Válido
        demo = Demographics(
            age=30,
            gender="F",
            disease_duration_years=3,
            ibd_type="ulcerative_colitis",
            montreal_location="E2"
        )
        assert demo.montreal_location == "E2"

        # Inválido para CU (L1 es para Crohn)
        with pytest.raises(ValidationError):
            Demographics(
                age=30,
                gender="F",
                disease_duration_years=3,
                ibd_type="ulcerative_colitis",
                montreal_location="L1"
            )


class TestMedicalHistory:
    """Tests para el schema de historial."""

    def test_valid_history(self):
        """Test con historial válido."""
        history = MedicalHistory(
            previous_flares=3,
            last_flare_days_ago=120
        )
        assert history.previous_flares == 3
        assert history.last_flare_days_ago == 120

    def test_negative_values_fail(self):
        """Test que valores negativos fallan."""
        with pytest.raises(ValidationError):
            MedicalHistory(
                previous_flares=-1,
                last_flare_days_ago=100
            )


class TestPredictionRequest:
    """Tests para el schema completo de request."""

    def test_valid_prediction_request(self):
        """Test con request completo válido."""
        request = PredictionRequest(
            symptoms=Symptoms(
                abdominal_pain=7,
                blood_in_stool=False,
                diarrhea=6,
                fatigue=5,
                fever=False,
                nausea=3
            ),
            demographics=Demographics(
                age=32,
                gender="F",
                disease_duration_years=5,
                bmi=22.5,
                ibd_type="crohn",
                montreal_location="L3"
            ),
            history=MedicalHistory(
                previous_flares=3,
                last_flare_days_ago=120
            )
        )
        assert request.symptoms.abdominal_pain == 7
        assert request.demographics.ibd_type == "crohn"
        assert request.history.previous_flares == 3

    def test_prediction_request_from_dict(self):
        """Test creación desde diccionario (como viene del JSON)."""
        data = {
            "symptoms": {
                "abdominal_pain": 7,
                "blood_in_stool": False,
                "diarrhea": 6,
                "fatigue": 5,
                "fever": False,
                "nausea": 3
            },
            "demographics": {
                "age": 32,
                "gender": "F",
                "disease_duration_years": 5,
                "bmi": 22.5,
                "ibd_type": "crohn",
                "montreal_location": "L3"
            },
            "history": {
                "previous_flares": 3,
                "last_flare_days_ago": 120
            }
        }
        request = PredictionRequest(**data)
        assert isinstance(request.symptoms, Symptoms)
        assert isinstance(request.demographics, Demographics)
        assert isinstance(request.history, MedicalHistory)


class TestFlareRiskPrediction:
    """Tests para el schema de predicción."""

    def test_valid_prediction(self):
        """Test con predicción válida."""
        prediction = FlareRiskPrediction(
            flare_risk="medium",
            probability=0.65,
            confidence=0.82,
            risk_score=6.5
        )
        assert prediction.flare_risk == "medium"
        assert 0 <= prediction.probability <= 1
        assert 0 <= prediction.confidence <= 1

    def test_invalid_flare_risk(self):
        """Test que nivel de riesgo inválido falla."""
        with pytest.raises(ValidationError):
            FlareRiskPrediction(
                flare_risk="invalid",  # Debe ser low, medium, o high
                probability=0.5,
                confidence=0.8,
                risk_score=5.0
            )

    def test_probability_range(self):
        """Test que probabilidad está en rango 0-1."""
        # Válido
        prediction = FlareRiskPrediction(
            flare_risk="low",
            probability=0.25,
            confidence=0.9,
            risk_score=2.5
        )
        assert prediction.probability == 0.25

        # Inválido (> 1)
        with pytest.raises(ValidationError):
            FlareRiskPrediction(
                flare_risk="high",
                probability=1.5,
                confidence=0.8,
                risk_score=8.0
            )

# 🔌 Guía de Integración - API ML

Esta guía explica cómo integrar el servicio ML en la aplicación web del proyecto.

## 📋 Contexto

El servicio ML es **independiente** de la aplicación web. Funciona como un microservicio stateless que:
- NO tiene base de datos
- NO gestiona usuarios
- Solo recibe datos, procesa y devuelve predicciones

## 🏗️ Arquitectura de Integración

```
┌─────────────────────┐
│   Frontend (Vue)    │
│   localhost:5173    │
└──────────┬──────────┘
           │ HTTP
           ▼
┌─────────────────────┐
│  Backend (FastAPI)  │
│   localhost:8000    │
│                     │
│  ┌──────────────┐   │
│  │ ml_client.py │   │  ← Cliente HTTP para llamar al ML API
│  └──────┬───────┘   │
└─────────┼───────────┘
          │ HTTP
          ▼
┌─────────────────────┐
│   ML API (FastAPI)  │  ← Este proyecto
│   localhost:8001    │
│                     │
│  ┌──────────────┐   │
│  │ Modelos ML   │   │
│  └──────────────┘   │
└─────────────────────┘
```

## 🚀 Setup Inicial

### 1. Verificar que el servicio ML está corriendo

```bash
# En una terminal, inicia el servicio ML
cd crohn-flare-predictor
make serve

# Debería estar en http://localhost:8001
curl http://localhost:8001/health
```

### 2. Configurar variables de entorno en el backend web

```bash
# En crohn-web-app/.env
ML_API_URL=http://localhost:8001
ML_API_TIMEOUT=30
```

## 💻 Implementación en el Backend Web

### Paso 1: Crear cliente HTTP para ML API

Crea el archivo `crohn-web-app/backend/api/ml_client.py`:

```python
"""
Cliente para comunicarse con el servicio ML.
"""
import httpx
from typing import Dict, Any, List
from fastapi import HTTPException, status
import logging

logger = logging.getLogger(__name__)


class MLAPIClient:
    """Cliente HTTP para el servicio ML."""

    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
        self.timeout = 30.0

    async def predict_flare(
        self,
        symptoms: Dict,
        demographics: Dict,
        history: Dict
    ) -> Dict[str, Any]:
        """
        Predecir riesgo de brote para un paciente.

        Args:
            symptoms: Síntomas actuales (abdominal_pain, diarrhea, etc.)
            demographics: Datos demográficos (age, gender, etc.)
            history: Historial médico (previous_flares, medications, etc.)

        Returns:
            {
                "prediction": {
                    "flare_risk": "low|medium|high",
                    "probability": float,
                    "confidence": float
                },
                "factors": {
                    "top_contributors": [...],
                    "symptom_severity_score": float
                },
                "recommendation": str
            }

        Raises:
            HTTPException: Si el servicio ML no está disponible
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/predict",
                    json={
                        "symptoms": symptoms,
                        "demographics": demographics,
                        "history": history
                    }
                )
                response.raise_for_status()
                return response.json()

        except httpx.TimeoutException:
            logger.error("ML API timeout")
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="ML service timeout"
            )
        except httpx.HTTPError as e:
            logger.error(f"ML API error: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="ML service unavailable"
            )

    async def analyze_trends(
        self,
        patient_id: str,
        daily_records: List[Dict]
    ) -> Dict[str, Any]:
        """
        Analizar tendencias de síntomas en el tiempo.

        Args:
            patient_id: ID del paciente
            daily_records: Lista de registros diarios
                [
                    {
                        "date": "2024-11-01",
                        "symptoms": {...}
                    },
                    ...
                ]

        Returns:
            {
                "patient_id": str,
                "analysis_period": {...},
                "trends": {
                    "overall_trend": "improving|stable|worsening",
                    "severity_change": float,
                    "concerning_patterns": [...]
                },
                "risk_assessment": {...},
                "recommendations": [...]
            }
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/analyze/trends",
                    json={
                        "patient_id": patient_id,
                        "daily_records": daily_records,
                        "window_days": 14
                    }
                )
                response.raise_for_status()
                return response.json()

        except httpx.HTTPError as e:
            logger.error(f"ML API trends error: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="ML service unavailable"
            )

    async def batch_predict(
        self,
        patients: List[Dict]
    ) -> Dict[str, Any]:
        """
        Predicciones por lotes (útil para dashboard médico).

        Args:
            patients: Lista de hasta 100 pacientes
                [
                    {
                        "patient_id": str,
                        "symptoms": {...},
                        "demographics": {...},
                        "history": {...}
                    },
                    ...
                ]

        Returns:
            {
                "results": [...],
                "processed_count": int,
                "failed_count": int
            }
        """
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:  # Más timeout
                response = await client.post(
                    f"{self.base_url}/predict/batch",
                    json={"patients": patients}
                )
                response.raise_for_status()
                return response.json()

        except httpx.HTTPError as e:
            logger.error(f"ML API batch error: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="ML service unavailable"
            )


# Singleton instance
ml_client = MLAPIClient()
```

### Paso 2: Usar en endpoints del backend web

Ejemplo de cómo usar el cliente en tus endpoints:

```python
# crohn-web-app/backend/api/symptoms.py
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from db.database import get_db
from db.models import User, DailySymptom, PredictionCache
from .ml_client import ml_client
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/symptoms/daily")
async def record_daily_symptoms(
    symptoms: SymptomsInput,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Usuario registra síntomas del día.
    1. Guardar en BD
    2. Llamar a ML API para predicción
    3. Guardar predicción en cache
    4. Devolver resultado
    """

    # 1. Guardar síntomas en BD
    symptom_record = DailySymptom(
        user_id=current_user.id,
        symptom_date=date.today(),
        abdominal_pain=symptoms.abdominal_pain,
        diarrhea=symptoms.diarrhea,
        fatigue=symptoms.fatigue,
        fever=symptoms.fever,
        weight_change=symptoms.weight_change,
        blood_in_stool=symptoms.blood_in_stool or False,
        nausea=symptoms.nausea or 0
    )
    db.add(symptom_record)
    db.commit()
    db.refresh(symptom_record)

    # 2. Llamar al servicio ML (puede fallar, no bloquear la app)
    prediction = None
    try:
        ml_prediction = await ml_client.predict_flare(
            symptoms=symptoms.dict(),
            demographics={
                "age": current_user.age,
                "gender": current_user.gender,
                "disease_duration_years": current_user.disease_duration_years,
                "bmi": current_user.bmi
            },
            history={
                "previous_flares": current_user.previous_flares,
                "medications": current_user.medications,
                "last_flare_days_ago": calculate_days_since_flare(current_user),
                "surgery_history": current_user.surgery_history,
                "smoking_status": current_user.smoking_status
            }
        )

        # 3. Guardar predicción en cache
        prediction_cache = PredictionCache(
            user_id=current_user.id,
            symptom_record_id=symptom_record.id,
            flare_risk=ml_prediction["prediction"]["flare_risk"],
            probability=ml_prediction["prediction"]["probability"],
            confidence=ml_prediction["prediction"]["confidence"],
            recommendation=ml_prediction["recommendation"],
            factors=ml_prediction["factors"]
        )
        db.add(prediction_cache)
        db.commit()

        prediction = ml_prediction

    except Exception as e:
        # Si ML API falla, continuar sin predicción
        logger.warning(f"ML API unavailable: {e}")
        prediction = None

    # 4. Devolver resultado
    return {
        "symptom_record": symptom_record,
        "prediction": prediction,
        "message": "Symptoms recorded successfully"
    }


@router.get("/trends/{user_id}")
async def get_user_trends(
    user_id: int,
    days: int = 14,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Obtener análisis de tendencias para un usuario."""

    # Verificar permisos
    if current_user.id != user_id and not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Forbidden")

    # Obtener registros diarios
    daily_records = db.query(DailySymptom).filter(
        DailySymptom.user_id == user_id
    ).order_by(DailySymptom.symptom_date.desc()).limit(days).all()

    if len(daily_records) < 7:
        raise HTTPException(
            status_code=400,
            detail="Need at least 7 days of data for trend analysis"
        )

    # Formatear para ML API
    ml_records = [
        {
            "date": record.symptom_date.isoformat(),
            "symptoms": {
                "abdominal_pain": record.abdominal_pain,
                "diarrhea": record.diarrhea,
                "fatigue": record.fatigue,
                "fever": record.fever,
                "weight_change": record.weight_change,
                "blood_in_stool": record.blood_in_stool,
                "nausea": record.nausea or 0
            }
        }
        for record in reversed(daily_records)  # Ordenar cronológicamente
    ]

    # Llamar al servicio ML
    trend_analysis = await ml_client.analyze_trends(
        patient_id=str(user_id),
        daily_records=ml_records
    )

    return trend_analysis
```

## 📊 Schemas de Datos

### Formato de Síntomas

```python
{
    "abdominal_pain": int (0-10),
    "diarrhea": int (0-10),
    "fatigue": int (0-10),
    "fever": bool,
    "weight_change": float,
    "blood_in_stool": bool,
    "nausea": int (0-10)
}
```

### Formato de Demografia

```python
{
    "age": int (0-120),
    "gender": "M" | "F" | "O",
    "disease_duration_years": int,
    "bmi": float (opcional)
}
```

### Formato de Historial

```python
{
    "previous_flares": int,
    "medications": list[str],
    "last_flare_days_ago": int,
    "surgery_history": bool (opcional),
    "smoking_status": "never" | "former" | "current" (opcional)
}
```

## 🔄 Flujos Comunes

### Flujo 1: Registro Diario de Síntomas

```
Usuario completa formulario
    ↓
Frontend → POST /api/symptoms/daily (Backend Web)
    ↓
Backend guarda en BD
    ↓
Backend → POST /predict (ML API)
    ↓
ML API devuelve predicción
    ↓
Backend guarda predicción en cache
    ↓
Backend → Frontend (síntomas + predicción)
    ↓
Mostrar al usuario
```

### Flujo 2: Ver Tendencias (Login o Dashboard)

```
Usuario hace login / abre dashboard
    ↓
Frontend → GET /api/trends/{user_id} (Backend Web)
    ↓
Backend obtiene últimos 14 días de BD
    ↓
Backend → POST /analyze/trends (ML API)
    ↓
ML API analiza tendencias
    ↓
Backend → Frontend (análisis)
    ↓
Mostrar gráficas y alertas
```

## 🚨 Manejo de Errores

**Importante:** El servicio ML puede no estar disponible. La app web debe funcionar sin él.

```python
try:
    prediction = await ml_client.predict_flare(...)
except HTTPException:
    # ML API no disponible
    prediction = None
    # Continuar sin predicción
    logger.warning("ML service unavailable, continuing without prediction")
```

## 🧪 Testing

### Test de integración

```python
# tests/test_ml_integration.py
import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_ml_api_health():
    """Verificar que ML API está disponible."""
    async with AsyncClient(base_url="http://localhost:8001") as client:
        response = await client.get("/health")
        assert response.status_code == 200

@pytest.mark.asyncio
async def test_prediction():
    """Test de predicción."""
    async with AsyncClient(base_url="http://localhost:8001") as client:
        response = await client.post("/predict", json={
            "symptoms": {...},
            "demographics": {...},
            "history": {...}
        })
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert data["prediction"]["flare_risk"] in ["low", "medium", "high"]
```

## 📚 Recursos Adicionales

- **Documentación interactiva:** http://localhost:8001/docs
- **Ejemplos de uso:** `../scripts/test_api.py`
- **Datos de ejemplo:** `../scripts/api_examples.json`

## ❓ FAQ

**P: ¿Qué pasa si el servicio ML está caído?**
R: La app web debe continuar funcionando. Simplemente no se generan predicciones.

**P: ¿Debo guardar las predicciones en mi BD?**
R: Sí, recomendado. Así tienes histórico y no dependes 100% del servicio ML.

**P: ¿Puedo llamar al ML API desde el frontend directamente?**
R: No recomendado. Hazlo desde el backend por seguridad y para manejar errores.

**P: ¿Cómo sé si una predicción es nueva o del cache?**
R: Guarda el timestamp cuando llamas a la API. Si hay dos llamadas el mismo día, usa el cache.

## 📞 Soporte

Para dudas sobre la integración o errores del servicio ML, contactarme directamente.

# 🔌 Guía de Integración - ML API

**Cómo integrar el servicio ML en tu aplicación web FastAPI + Vue + MySQL**

## 📋 Contexto

Este microservicio ML es **stateless e independiente** de la aplicación web principal:
- ❌ NO tiene base de datos propia
- ❌ NO gestiona usuarios
- ❌ NO almacena predicciones
- ✅ Solo recibe datos, procesa y devuelve predicciones

**Tu backend web** es responsable de:
- Autenticar usuarios (JWT)
- Almacenar síntomas en MySQL
- Llamar al ML API cuando sea necesario
- Cachear predicciones en MySQL
- Manejar errores si el ML API falla

---

## 🏗️ Arquitectura

```
┌─────────────────┐
│   Vue.js App    │  Puerto 5173 (dev) / 80 (prod)
│   (Frontend)    │
└────────┬────────┘
         │ HTTP (axios/fetch)
         ▼
┌─────────────────┐
│  FastAPI Web    │  Puerto 8000
│   (Backend)     │
│                 │
│  • JWT Auth     │
│  • MySQL ───────┼──► users, daily_symptoms, meals, etc.
│  • Endpoints    │
│  • ml_client.py │  ← Cliente HTTP para ML API
└────────┬────────┘
         │ HTTP (httpx)
         ▼
┌─────────────────┐
│   ML API        │  Puerto 8001
│  (Este repo)    │
│                 │
│  • /predict     │
│  • /health      │
│  • Modelos ML   │
└─────────────────┘
```

---

## 🚀 Setup Inicial

### 1. Asegúrate de que el ML API está corriendo

```bash
# En el directorio crohn-flare-predictor/
cd crohn-flare-predictor
uv sync
uv run uvicorn api.app:app --reload --host 0.0.0.0 --port 8001

# O con Makefile
make serve

# Verificar que funciona
curl http://localhost:8001/health
# {"status":"healthy","version":"1.0.0"}
```

### 2. Configura variables de entorno en tu backend web

```bash
# En tu archivo .env del backend web
ML_API_URL=http://localhost:8001
ML_API_TIMEOUT=30
```

---

## 💻 Implementación en tu Backend Web

### Paso 1: Crear cliente HTTP para ML API

Crea `backend/api/ml_client.py` en tu proyecto web:

```python
"""
Cliente HTTP para comunicarse con el ML API.
"""
import httpx
from typing import Dict, Any
from fastapi import HTTPException, status
import logging

logger = logging.getLogger(__name__)

class MLAPIClient:
    """Cliente asíncrono para el servicio ML."""

    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
        self.timeout = 30.0

    async def predict_flare(
        self,
        symptoms: Dict,
        demographics: Dict,
        history: Dict,
        temporal_features: Dict = None
    ) -> Dict[str, Any]:
        """
        Predecir riesgo de brote.

        Args:
            symptoms: {abdominal_pain, diarrhea, fatigue, fever, blood_in_stool, nausea}
            demographics: {age, gender, disease_duration_years, bmi, ibd_type, montreal_location}
            history: {previous_flares, last_flare_days_ago}
            temporal_features: (opcional) tendencias calculadas de los últimos 7 días

        Returns:
            {
                "prediction": {"flare_risk": "low|medium|high", "probability": float, ...},
                "factors": {"top_contributors": [...], ...},
                "cluster_info": {...},
                "recommendation": str
            }

        Raises:
            HTTPException: Si el ML API no está disponible o hay error
        """
        try:
            payload = {
                "symptoms": symptoms,
                "demographics": demographics,
                "history": history
            }
            if temporal_features:
                payload["temporal_features"] = temporal_features

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/predict",
                    json=payload
                )
                response.raise_for_status()
                return response.json()

        except httpx.TimeoutException:
            logger.error("ML API timeout")
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="ML service timeout"
            )
        except httpx.HTTPStatusError as e:
            logger.error(f"ML API HTTP error: {e.response.status_code} - {e.response.text}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"ML service error: {e.response.status_code}"
            )
        except Exception as e:
            logger.error(f"ML API unexpected error: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="ML service unavailable"
            )

    async def health_check(self) -> bool:
        """Verificar si el ML API está disponible."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/health")
                return response.status_code == 200
        except:
            return False


# Singleton instance (importa esto en tus endpoints)
ml_client = MLAPIClient()
```

### Paso 2: Usar en tus endpoints

**Ejemplo: Endpoint para registrar síntomas diarios**

```python
# backend/api/symptoms.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from datetime import date
from db.database import get_db
from db.models import User, DailySymptom, FlareP rediction
from api.ml_client import ml_client
from api.auth import get_current_user
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/symptoms", tags=["symptoms"])


@router.post("/daily")
async def record_daily_symptoms(
    symptoms: SymptomsInput,  # Tu Pydantic schema
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Usuario registra síntomas del día.

    Flujo:
    1. Validar y guardar síntomas en BD
    2. Llamar al ML API para predicción
    3. Guardar predicción en cache
    4. Devolver todo al frontend
    """

    # 1. Guardar síntomas en BD
    symptom_record = DailySymptom(
        user_id=current_user.id,
        record_date=date.today(),
        abdominal_pain=symptoms.abdominal_pain,
        diarrhea=symptoms.diarrhea,
        fatigue=symptoms.fatigue,
        fever=symptoms.fever,
        blood_in_stool=symptoms.blood_in_stool,
        nausea=symptoms.nausea,
        wellness_score=symptoms.wellness_score,
        notes=symptoms.notes
    )
    db.add(symptom_record)
    db.commit()
    db.refresh(symptom_record)

    # 2. Llamar al ML API (no bloquear si falla)
    prediction = None
    try:
        ml_prediction = await ml_client.predict_flare(
            symptoms={
                "abdominal_pain": symptoms.abdominal_pain,
                "diarrhea": symptoms.diarrhea,
                "fatigue": symptoms.fatigue,
                "fever": symptoms.fever,
                "blood_in_stool": symptoms.blood_in_stool,
                "nausea": symptoms.nausea
            },
            demographics={
                "age": current_user.age,
                "gender": current_user.gender,
                "disease_duration_years": current_user.disease_duration_years,
                "bmi": current_user.bmi,
                "ibd_type": current_user.ibd_type,
                "montreal_location": current_user.montreal_classification
            },
            history={
                "previous_flares": current_user.previous_flares,
                "last_flare_days_ago": calculate_days_since_flare(current_user)
            }
            # temporal_features opcional - si tienes datos históricos
        )

        # 3. Guardar predicción en cache
        prediction_record = FlarePrediction(
            user_id=current_user.id,
            symptom_record_id=symptom_record.id,
            flare_risk=ml_prediction["prediction"]["flare_risk"],
            probability=ml_prediction["prediction"]["probability"],
            confidence=ml_prediction["prediction"]["confidence"],
            top_contributors=ml_prediction["factors"]["top_contributors"],
            recommendation=ml_prediction["recommendation"],
            cluster_id=ml_prediction.get("cluster_info", {}).get("cluster_id")
        )
        db.add(prediction_record)
        db.commit()

        prediction = ml_prediction

    except HTTPException as e:
        # ML API no disponible - continuar sin predicción
        logger.warning(f"ML API unavailable: {e.detail}")
        prediction = None

    # 4. Devolver resultado
    return {
        "success": True,
        "symptom_record": {
            "id": symptom_record.id,
            "date": symptom_record.record_date.isoformat(),
            "wellness_score": symptom_record.wellness_score
        },
        "prediction": prediction,  # puede ser None si ML API falla
        "message": "Síntomas registrados correctamente"
    }


def calculate_days_since_flare(user: User) -> int:
    """Helper para calcular días desde el último brote."""
    if not user.last_flare_date:
        return 365  # Default si no hay brotes previos
    return (date.today() - user.last_flare_date).days
```

**Ejemplo: Endpoint para el dashboard**

```python
@router.get("/dashboard")
async def get_dashboard(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Obtener datos para el dashboard del usuario.

    Devuelve:
    - Últimos 30 días de síntomas (para gráfica)
    - Predicciones cacheadas
    - Estadísticas del mes
    """

    # Obtener últimos 30 días
    from datetime import timedelta
    start_date = date.today() - timedelta(days=30)

    symptom_records = db.query(DailySymptom).filter(
        DailySymptom.user_id == current_user.id,
        DailySymptom.record_date >= start_date
    ).order_by(DailySymptom.record_date).all()

    # Obtener predicciones del último mes
    predictions = db.query(FlarePrediction).join(
        DailySymptom
    ).filter(
        FlarePrediction.user_id == current_user.id,
        DailySymptom.record_date >= start_date
    ).all()

    # Calcular estadísticas
    total_days = len(symptom_records)
    good_days = sum(1 for s in symptom_records if s.wellness_score >= 8)
    bad_days = sum(1 for s in symptom_records if s.wellness_score <= 4)
    avg_wellness = sum(s.wellness_score for s in symptom_records) / total_days if total_days > 0 else 0

    # Alertas (si hay predicción de alto riesgo reciente)
    alerts = []
    recent_high_risk = [p for p in predictions if p.flare_risk == "high"]
    if recent_high_risk:
        latest = max(recent_high_risk, key=lambda x: x.created_at)
        alerts.append({
            "type": "warning",
            "message": f"Riesgo ALTO de brote ({int(latest.probability * 100)}% probabilidad)",
            "factors": latest.top_contributors,
            "recommendation": latest.recommendation
        })

    return {
        "monthly_data": [
            {
                "date": s.record_date.isoformat(),
                "wellness_score": s.wellness_score,
                "prediction": next(
                    ({"risk": p.flare_risk, "probability": float(p.probability)}
                     for p in predictions if p.symptom_record_id == s.id),
                    None
                )
            }
            for s in symptom_records
        ],
        "summary": {
            "total_days": total_days,
            "good_days": good_days,
            "bad_days": bad_days,
            "avg_wellness": round(avg_wellness, 1)
        },
        "alerts": alerts
    }
```

---

## 🔄 Flujos Principales

### Flujo 1: Usuario registra síntomas

```
Usuario completa formulario (Vue)
         │
         ▼
Frontend → POST /api/symptoms/daily (Backend Web)
  {symptoms, meals, exercise}
         │
         ▼
Backend Web:
  1. Validar datos con Pydantic
  2. Guardar en MySQL (daily_symptoms, meals, exercise_log)
  3. Obtener demographics + history del user
         │
         ▼
Backend Web → POST /predict (ML API)
  {symptoms, demographics, history}
         │
         ▼
ML API procesa y devuelve predicción
         │
         ▼
Backend Web:
  1. Guardar en MySQL (flare_predictions)
  2. Devolver todo al frontend
         │
         ▼
Frontend muestra:
  - ✅ Confirmación
  - 🔮 Predicción de riesgo
  - 💡 Recomendación
```

### Flujo 2: Usuario ve dashboard al hacer login

```
Usuario hace login (Vue)
         │
         ▼
Frontend → GET /api/symptoms/dashboard (Backend Web)
         │
         ▼
Backend Web:
  1. Consultar MySQL (últimos 30 días)
  2. Obtener predicciones cacheadas
  3. Calcular estadísticas
  4. Devolver agregado
         │
         ▼
Frontend renderiza:
  - 📊 Gráfica mensual (1-10)
  - 🔔 Alertas de riesgo alto
  - 📈 Estadísticas
```

---

## 🚨 Manejo de Errores

**MUY IMPORTANTE**: La app web debe funcionar sin el ML API.

```python
# ✅ CORRECTO - Graceful degradation
try:
    prediction = await ml_client.predict_flare(...)
except HTTPException:
    logger.warning("ML service unavailable")
    prediction = None  # Continuar sin predicción

# Devolver al frontend
return {
    "symptom_record": {...},
    "prediction": prediction  # puede ser None
}
```

```python
# ❌ INCORRECTO - Bloquea la app si ML API falla
prediction = await ml_client.predict_flare(...)  # HTTPException mata el request
return {"prediction": prediction}
```

**En el frontend (Vue):**

```javascript
// Manejar prediction que puede ser null
if (response.data.prediction) {
  // Mostrar predicción
  showPredictionModal(response.data.prediction);
} else {
  // ML API no disponible
  showNotification("Síntomas guardados (predicción no disponible)", "info");
}
```

---

## 🧪 Testing

### 1. Test de integración (pytest)

```python
# tests/test_ml_integration.py
import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_ml_api_health():
    """Verificar que ML API responde."""
    async with AsyncClient(base_url="http://localhost:8001") as client:
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

@pytest.mark.asyncio
async def test_ml_api_prediction():
    """Test de predicción básica."""
    async with AsyncClient(base_url="http://localhost:8001") as client:
        response = await client.post("/predict", json={
            "symptoms": {
                "abdominal_pain": 5,
                "blood_in_stool": False,
                "diarrhea": 4,
                "fatigue": 6,
                "fever": False,
                "nausea": 2
            },
            "demographics": {
                "age": 30,
                "gender": "F",
                "disease_duration_years": 3,
                "bmi": 22.0,
                "ibd_type": "crohn",
                "montreal_location": "L3"
            },
            "history": {
                "previous_flares": 2,
                "last_flare_days_ago": 180
            }
        })
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert data["prediction"]["flare_risk"] in ["low", "medium", "high"]
        assert 0 <= data["prediction"]["probability"] <= 1
```

### 2. Test de degradación graceful

```python
@pytest.mark.asyncio
async def test_symptoms_endpoint_when_ml_fails(client, auth_headers):
    """La app debe funcionar si ML API falla."""
    # Simular ML API caído (mock o shutdown real)
    response = await client.post(
        "/api/symptoms/daily",
        json={...},
        headers=auth_headers
    )

    # Debe devolver 200 aunque ML API falle
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["prediction"] is None  # Sin predicción
```

---

## 📊 Schemas de Datos

### Request al ML API

```python
{
    "symptoms": {
        "abdominal_pain": int (0-10),
        "blood_in_stool": bool,
        "diarrhea": int (0-10),
        "fatigue": int (0-10),
        "fever": bool,
        "nausea": int (0-10)
    },
    "demographics": {
        "age": int (0-120),
        "gender": "M" | "F" | "O",
        "disease_duration_years": int (≥0),
        "bmi": float (opcional),
        "ibd_type": "crohn" | "ulcerative_colitis",
        "montreal_location": "L1"|"L2"|"L3"|"L4" (Crohn) o "E1"|"E2"|"E3" (CU)
    },
    "history": {
        "previous_flares": int (≥0),
        "last_flare_days_ago": int (≥0)
    }
}
```

### Response del ML API

```python
{
    "prediction": {
        "flare_risk": "low" | "medium" | "high",
        "probability": float (0-1),
        "confidence": float (0-1),
        "risk_score": float (0-10)
    },
    "factors": {
        "top_contributors": List[str],
        "symptom_severity_score": float
    },
    "cluster_info": {
        "cluster_id": int,
        "cluster_description": str
    },
    "recommendation": str
}
```

---

## 🔐 Seguridad

### 1. No exponer ML API al público

```nginx
# nginx.conf (producción)
location /ml-api/ {
    # Solo accesible desde el backend web (internal)
    internal;
    proxy_pass http://ml-api:8001/;
}
```

### 2. Validar en ambos lados

```python
# Backend web: validar ANTES de llamar a ML API
if symptoms.abdominal_pain < 0 or symptoms.abdominal_pain > 10:
    raise HTTPException(400, "Invalid abdominal_pain value")

# ML API también valida (Pydantic schemas)
```

### 3. Rate limiting

```python
# En tu backend web, limitar predicciones por usuario
# Ej: máximo 10 predicciones/día

@router.post("/symptoms/daily")
@limiter.limit("10/day")  # slowapi o similar
async def record_daily_symptoms(...):
    ...
```

---

## 📚 Recursos Adicionales

- **API Reference completa**: Ver `docs/API_REFERENCE.md`
- **Guía de la app web**: Ver `docs/WEB_APP_GUIDE.md`
- **Ejemplos de uso**: Ver `scripts/test_api.py`
- **Swagger UI**: http://localhost:8001/docs (cuando ML API esté corriendo)

---

## ❓ FAQ

**P: ¿Qué hago si el ML API está caído?**
R: La app debe continuar funcionando normalmente. Simplemente no generes predicciones y muestra un mensaje informativo al usuario.

**P: ¿Debo guardar las predicciones en mi BD?**
R: **Sí, altamente recomendado.** Esto te permite:
- Evitar llamadas redundantes
- Tener histórico de predicciones
- Mostrar predicciones aunque el ML API esté caído
- Analizar tendencias a largo plazo

**P: ¿Puedo llamar al ML API desde Vue directamente?**
R: **No recomendado.** Razones:
- Seguridad: expones el ML API públicamente
- CORS: problemas de cross-origin
- Error handling: más difícil de manejar
- Autenticación: no tienes contexto del usuario

**P: ¿Cómo sé si una predicción es nueva o del cache?**
R: Compara `symptom_record_id` y `created_at`. Si ya hiciste una predicción para ese registro el mismo día, usa el cache.

**P: ¿Cuánto tardan las predicciones?**
R: Típicamente <200ms. Si tarda más, revisa:
- Red (latencia)
- Carga del modelo
- Timeouts configurados

**P: ¿Puedo hacer batch predictions para múltiples usuarios?**
R: Sí, usa el endpoint `/predict/batch`. Útil para:
- Dashboard médico
- Reportes nocturnos
- Análisis de cohortes

---

## 💡 Tips Finales

1. **Empezar simple**: Implementa solo `/predict` primero, añade `/analyze/trends` después

2. **Logging**: Logea todas las llamadas al ML API para debugging
   ```python
   logger.info(f"ML prediction for user {user_id}: {prediction['prediction']['flare_risk']}")
   ```

3. **Monitoring**: Monitorea disponibilidad del ML API
   ```python
   # Endpoint de health check para tu backend
   @router.get("/health")
   async def health():
       ml_status = await ml_client.health_check()
       return {
           "status": "healthy",
           "ml_api": "up" if ml_status else "down"
       }
   ```

4. **Caching inteligente**: No llames al ML API si los síntomas no han cambiado

5. **Feedback loop**: Guarda cuando un médico confirma/rechaza una predicción (para futuras mejoras del modelo)

---

## 📞 Soporte

Para dudas sobre la integración:
- Ver documentación completa en `docs/`
- Revisar ejemplos en `scripts/`
- Contactar directamente (Asier)

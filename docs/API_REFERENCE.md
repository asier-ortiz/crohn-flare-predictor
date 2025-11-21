# 📘 Referencia de API - ML Microservice

Documentación completa de los endpoints del servicio ML para predicción de brotes de EII.

## 🌐 Base URL

- **Desarrollo**: `http://localhost:8001`
- **Producción**: `https://tu-dominio.com/ml-api`

## 🔑 Autenticación

**Actualmente:** Sin autenticación (comunicación server-to-server)

**Futuro:** API Key en header `X-API-Key: your-api-key-here`

---

## 📍 Endpoints

### 1. Health Check

Verificar que el servicio está disponible y el modelo cargado.

**Request:**
```http
GET /health
```

**Response:** `200 OK`
```json
{
  "status": "healthy",
  "version": "1.0.0"
}
```

**Errores:**
- `503 Service Unavailable`: Modelo ML no cargado

---

### 2. Predicción Individual

**El endpoint más importante.** Predice el riesgo de brote para un paciente basándose en síntomas actuales, demografía e historial médico.

**Request:**
```http
POST /predict
Content-Type: application/json
```

```json
{
  "symptoms": {
    "abdominal_pain": 7,
    "blood_in_stool": false,
    "diarrhea": 6,
    "fatigue": 5,
    "fever": false,
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
  },
  "temporal_features": {
    "pain_trend_7d": 0.15,
    "diarrhea_trend_7d": 0.10,
    "fatigue_trend_7d": 0.05,
    "symptom_volatility_7d": 1.2,
    "symptom_change_rate": 0.08,
    "days_since_low_symptoms": 5
  }
}
```

**Campos requeridos:**

| Sección | Campo | Tipo | Rango/Valores | Descripción |
|---------|-------|------|---------------|-------------|
| **symptoms** | `abdominal_pain` | int | 0-10 | Intensidad del dolor abdominal |
| | `blood_in_stool` | bool | - | Presencia de sangre en heces |
| | `diarrhea` | int | 0-10 | Severidad de la diarrea |
| | `fatigue` | int | 0-10 | Nivel de fatiga |
| | `fever` | bool | - | Presencia de fiebre |
| | `nausea` | int | 0-10 | Nivel de náuseas |
| **demographics** | `age` | int | 0-120 | Edad del paciente |
| | `gender` | str | M/F/O | Género |
| | `disease_duration_years` | int | ≥0 | Años desde diagnóstico |
| | `bmi` | float | 10-60 | Índice de masa corporal (opcional) |
| | `ibd_type` | str | crohn/ulcerative_colitis | Tipo de EII |
| | `montreal_location` | str | L1/L2/L3/L4 (Crohn)<br>E1/E2/E3 (CU) | Clasificación de Montreal |
| **history** | `previous_flares` | int | ≥0 | Número de brotes previos |
| | `last_flare_days_ago` | int | ≥0 | Días desde el último brote |
| **temporal_features** (opcional) | `pain_trend_7d` | float | - | Tendencia de dolor últimos 7 días |
| | `diarrhea_trend_7d` | float | - | Tendencia de diarrea últimos 7 días |
| | `fatigue_trend_7d` | float | - | Tendencia de fatiga últimos 7 días |
| | `symptom_volatility_7d` | float | - | Volatilidad de síntomas últimos 7 días |
| | `symptom_change_rate` | float | - | Tasa de cambio de síntomas |
| | `days_since_low_symptoms` | int | ≥0 | Días desde que síntomas estuvieron bajos |

**Response:** `200 OK`
```json
{
  "prediction": {
    "flare_risk": "medium",
    "probability": 0.65,
    "confidence": 0.82,
    "risk_score": 6.5
  },
  "factors": {
    "top_contributors": [
      "abdominal_pain",
      "diarrhea",
      "pain_trend_7d"
    ],
    "symptom_severity_score": 0.58
  },
  "cluster_info": {
    "cluster_id": 1,
    "cluster_confidence": 0.89,
    "model_source": "crohn_cluster_1",
    "cluster_description": "Ileocolónica (L3): Afecta íleon terminal y colon"
  },
  "ibd_info": {
    "ibd_type": "crohn",
    "montreal_classification": "L3"
  },
  "recommendation": "Monitoree sus síntomas de cerca. Considere contactar a su médico si empeoran.",
  "metadata": {
    "prediction_timestamp": "2024-11-21T10:30:00Z",
    "model_version": "2.0.0",
    "api_version": "1.0.0"
  }
}
```

**Interpretación de `flare_risk`:**
- `low`: Probabilidad < 0.40 (riesgo bajo, seguir monitoreando)
- `medium`: Probabilidad 0.40 - 0.70 (riesgo moderado, vigilar de cerca)
- `high`: Probabilidad > 0.70 (riesgo alto, considerar intervención médica)

**Errores:**
- `422 Validation Error`: Datos inválidos o fuera de rango
- `500 Internal Server Error`: Error en la predicción del modelo

**Ejemplo de error 422:**
```json
{
  "detail": [
    {
      "loc": ["body", "symptoms", "abdominal_pain"],
      "msg": "ensure this value is less than or equal to 10",
      "type": "value_error.number.not_le"
    }
  ]
}
```

---

### 3. Predicción por Lotes

Realizar predicciones para múltiples pacientes en una sola petición (máximo 100).

**Request:**
```http
POST /predict/batch
Content-Type: application/json
```

```json
{
  "patients": [
    {
      "patient_id": "P001",
      "symptoms": {...},
      "demographics": {...},
      "history": {...}
    },
    {
      "patient_id": "P002",
      "symptoms": {...},
      "demographics": {...},
      "history": {...}
    }
  ]
}
```

**Response:** `200 OK`
```json
{
  "results": [
    {
      "patient_id": "P001",
      "prediction": {
        "flare_risk": "low",
        "probability": 0.25,
        "confidence": 0.88
      },
      "factors": {...},
      "cluster_info": {...},
      "ibd_info": {...},
      "recommendation": "...",
      "metadata": {...}
    },
    {
      "patient_id": "P002",
      "prediction": {...},
      ...
    }
  ],
  "processed_count": 2,
  "failed_count": 0,
  "errors": []
}
```

**Casos de uso:**
- Dashboard médico con múltiples pacientes
- Reportes batch nocturnos
- Análisis de cohortes

---

### 4. Análisis de Tendencias

Analizar evolución temporal de síntomas (mínimo 7 días de datos).

**Request:**
```http
POST /analyze/trends
Content-Type: application/json
```

```json
{
  "patient_id": "P001",
  "daily_records": [
    {
      "date": "2024-11-01",
      "symptoms": {
        "abdominal_pain": 3,
        "diarrhea": 2,
        "fatigue": 4,
        "fever": false,
        "blood_in_stool": false,
        "nausea": 1
      }
    },
    {
      "date": "2024-11-02",
      "symptoms": {...}
    }
    // ... mínimo 7 registros
  ],
  "window_days": 14
}
```

**Response:** `200 OK`
```json
{
  "patient_id": "P001",
  "analysis_period": {
    "start_date": "2024-11-01",
    "end_date": "2024-11-14",
    "days_analyzed": 14
  },
  "trends": {
    "overall_trend": "stable",
    "severity_change": 0.05,
    "concerning_patterns": [],
    "symptom_volatility": 0.8
  },
  "risk_assessment": {
    "flare_risk": "low",
    "probability": 0.30,
    "confidence": 0.75
  },
  "recommendations": [
    "Continue current management plan",
    "Monitor for changes in symptom patterns"
  ]
}
```

**Valores de `overall_trend`:**
- `improving`: Síntomas mejorando consistentemente
- `stable`: Sin cambios significativos
- `worsening`: Síntomas empeorando (alerta)

**Errores:**
- `400 Bad Request`: Menos de 7 días de datos
- `422 Validation Error`: Formato incorrecto de fechas o síntomas

---

### 5. Información del Modelo

Obtener metadata del modelo activo (versión, métricas, features, etc.).

**Request:**
```http
GET /model/info
```

**Response:** `200 OK`
```json
{
  "model_version": "2.0.0",
  "model_type": "ClusterStratifiedRandomForest",
  "trained_date": "2024-11-15",
  "uses_cluster_models": true,
  "metrics": {
    "accuracy": 0.9922,
    "precision": 0.9524,
    "recall": 1.0000,
    "f1_score": 0.9756,
    "roc_auc": 0.9989
  },
  "features_count": 34,
  "feature_categories": {
    "base_features": 13,
    "derived_features": 21
  },
  "clusters": {
    "crohn": {
      "L1": 0,
      "L2": 1,
      "L3": 1,
      "L4": 2
    },
    "ulcerative_colitis": {
      "E1": 0,
      "E2": 1,
      "E3": 2
    }
  }
}
```

---

## 🔄 CORS

**Origins permitidos** (configurables via `.env`):
- `http://localhost:8000` (Backend web)
- `http://localhost:5173` (Frontend Vue dev)
- `http://localhost:3000` (Frontend alternativo)

Variable de entorno: `CORS_ORIGINS=http://localhost:8000,http://localhost:5173`

---

## ⚠️ Manejo de Errores

### Códigos de Estado HTTP

| Código | Significado | Cuándo ocurre |
|--------|-------------|---------------|
| `200` | Success | Petición procesada correctamente |
| `400` | Bad Request | Datos inválidos (ej: <7 días para trends) |
| `422` | Validation Error | Schema Pydantic inválido |
| `500` | Internal Server Error | Error en modelo ML o código |
| `503` | Service Unavailable | Modelo no cargado en startup |

### Formato de Error Estándar

```json
{
  "detail": "Mensaje de error descriptivo"
}
```

### Ejemplo con Validation Error (422)

```json
{
  "detail": [
    {
      "loc": ["body", "symptoms", "abdominal_pain"],
      "msg": "ensure this value is less than or equal to 10",
      "type": "value_error.number.not_le"
    },
    {
      "loc": ["body", "demographics", "ibd_type"],
      "msg": "unexpected value; permitted: 'crohn', 'ulcerative_colitis'",
      "type": "value_error.str.literal"
    }
  ]
}
```

---

## 📈 Rate Limits

**Actual:** Sin límites (desarrollo)

**Recomendado para producción:**
- **100 requests/minuto** por IP
- **1000 requests/día** por API key
- Considerar rate limit más bajo para `/predict/batch`

---

## 🔄 Versionado de API

**Versión actual:** `v1.0.0`

**Convención Semantic Versioning:**
- **Breaking changes**: `v2.0.0`
- **Nuevas features**: `v1.1.0`
- **Bug fixes**: `v1.0.1`

El versionado se maneja a nivel de header `api_version` en las respuestas.

---

## 📚 Ejemplos de Uso

### Python (httpx)

```python
import httpx
import asyncio

async def predict_flare():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8001/predict",
            json={
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
        )
        return response.json()

result = asyncio.run(predict_flare())
print(f"Risk: {result['prediction']['flare_risk']}")
print(f"Probability: {result['prediction']['probability']}")
```

### cURL

```bash
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "symptoms": {
      "abdominal_pain": 7,
      "blood_in_stool": false,
      "diarrhea": 6,
      "fatigue": 5,
      "fever": false,
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
  }'
```

### JavaScript (Fetch API)

```javascript
async function predictFlare() {
  const response = await fetch('http://localhost:8001/predict', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      symptoms: {
        abdominal_pain: 7,
        blood_in_stool: false,
        diarrhea: 6,
        fatigue: 5,
        fever: false,
        nausea: 3
      },
      demographics: {
        age: 32,
        gender: 'F',
        disease_duration_years: 5,
        bmi: 22.5,
        ibd_type: 'crohn',
        montreal_location: 'L3'
      },
      history: {
        previous_flares: 3,
        last_flare_days_ago: 120
      }
    })
  });

  const data = await response.json();
  console.log(`Risk: ${data.prediction.flare_risk}`);
  console.log(`Probability: ${data.prediction.probability}`);
}
```

---

## 🧪 Testing

Scripts de prueba disponibles en el repositorio:

- **`scripts/test_api.py`** - Tests completos en Python
- **`scripts/test_api.sh`** - Tests rápidos con curl
- **`scripts/evaluate_model.py`** - Evaluación de modelos con casos diversos

```bash
# Test rápido con Python
uv run python scripts/test_api.py

# Test rápido con curl
bash scripts/test_api.sh

# Evaluación completa del modelo
uv run python scripts/evaluate_model.py
```

---

## 📞 Documentación Interactiva

Cuando el servidor está corriendo:

- **Swagger UI**: http://localhost:8001/docs
- **ReDoc**: http://localhost:8001/redoc

Estas interfaces permiten:
- Ver todos los endpoints documentados
- Probar requests directamente desde el navegador
- Ver schemas de datos completos
- Descargar OpenAPI spec (JSON/YAML)

---

## 💡 Mejores Prácticas

### Para el Backend Web

1. **Siempre manejar timeouts**:
```python
async with httpx.AsyncClient(timeout=30.0) as client:
    response = await client.post(...)
```

2. **Cachear predicciones**: Guardar en BD para evitar llamadas redundantes

3. **Graceful degradation**: Si ML API falla, continuar sin predicción

4. **Validar antes de enviar**: Validar datos en backend web antes de llamar al ML API

5. **Logging**: Logear todas las llamadas al ML API para debugging

### Para el Frontend

1. **Nunca llamar directamente**: Siempre a través del backend web

2. **Loading states**: Mostrar spinner mientras se obtiene predicción

3. **Error handling**: Mostrar mensaje amigable si predicción falla

4. **No bloquear UI**: Predicción es opcional, no debe bloquear guardado de síntomas

---

## 📊 Monitoring

Métricas recomendadas para producción:

- **Latencia** de `/predict` (objetivo: <500ms p95)
- **Error rate** (objetivo: <1%)
- **Disponibilidad** (objetivo: 99.9%)
- **Predicciones/día**
- **Distribución de risk levels** (low/medium/high)

Herramientas sugeridas:
- **Prometheus** + **Grafana** para métricas
- **Sentry** para error tracking
- **FastAPI middleware** para logging automático

---

## ❓ FAQ

**P: ¿Puedo llamar al ML API desde el frontend directamente?**
R: No recomendado. Siempre hazlo desde el backend para control de acceso y manejo de errores.

**P: ¿Qué pasa si el ML API está caído?**
R: El backend web debe continuar funcionando. Simplemente no generes predicciones y notifica al usuario.

**P: ¿Debo cachear las predicciones?**
R: Sí, altamente recomendado. Guarda en tabla `flare_predictions` para evitar llamadas redundantes.

**P: ¿Cuánto tarda una predicción?**
R: Típicamente <200ms. Con cluster-stratified puede ser ligeramente más rápido (~150ms).

**P: ¿Puedo predecir sin `temporal_features`?**
R: Sí, son opcionales. El modelo los calculará internamente con valores por defecto.

**P: ¿Qué significa `cluster_id`?**
R: Identifica el submodelo específico usado según Montreal classification (L1-L4 / E1-E3).

---

## 📧 Soporte

- **Documentación completa**: Ver `docs/INTEGRATION.md`
- **Guía de la app web**: Ver `docs/WEB_APP_GUIDE.md`
- **Issues**: https://github.com/tu-usuario/crohn-flare-predictor/issues
- **Contacto directo**: Para dudas sobre integración

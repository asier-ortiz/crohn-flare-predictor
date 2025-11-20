# 📘 Referencia Completa de API

Documentación detallada de todos los endpoints del servicio ML.

## 🌐 Base URL

- **Desarrollo**: `http://localhost:8001`
- **Producción**: `https://tu-dominio.com/ml-api`

## 📍 Endpoints

### 1. Health Check

Verificar estado del servicio.

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

### 2. Información del Modelo

Obtener métricas y metadata del modelo.

**Request:**
```http
GET /model/info
```

**Response:** `200 OK`
```json
{
  "model_version": "1.0.0",
  "trained_date": "2024-01-15",
  "metrics": {
    "accuracy": 0.87,
    "precision": 0.84,
    "recall": 0.89,
    "f1_score": 0.86,
    "roc_auc": 0.91
  },
  "features_count": 45,
  "training_samples": 5000,
  "model_type": "RandomForest"
}
```

---

### 3. Predicción Individual

Predecir riesgo de brote para un paciente.

**Request:**
```http
POST /predict
Content-Type: application/json
```

```json
{
  "symptoms": {
    "abdominal_pain": 7,
    "diarrhea": 6,
    "fatigue": 5,
    "fever": false,
    "weight_change": -2.5,
    "blood_in_stool": false,
    "nausea": 4
  },
  "demographics": {
    "age": 32,
    "gender": "F",
    "disease_duration_years": 5,
    "bmi": 22.5
  },
  "history": {
    "previous_flares": 3,
    "medications": ["mesalamine", "prednisone"],
    "last_flare_days_ago": 120,
    "surgery_history": false,
    "smoking_status": "never"
  }
}
```

**Response:** `200 OK`
```json
{
  "prediction": {
    "flare_risk": "medium",
    "probability": 0.65,
    "confidence": 0.80
  },
  "factors": {
    "top_contributors": [
      "abdominal_pain",
      "diarrhea",
      "previous_flares"
    ],
    "symptom_severity_score": 0.55,
    "trend_indicator": "stable"
  },
  "recommendation": "Monitoree sus sintomas de cerca. Considere contactar a su medico si empeoran."
}
```

**Errores:**
- `422 Validation Error`: Datos inválidos
- `500 Internal Server Error`: Error en predicción

---

### 4. Predicción por Lotes

Predicciones múltiples (máximo 100 pacientes).

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
      "demographics": {...}
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
        "confidence": 0.85
      },
      "factors": {
        "top_contributors": ["general_symptom_pattern"],
        "symptom_severity_score": 0.15
      }
    },
    {
      "patient_id": "P002",
      "prediction": {...},
      "factors": {...}
    }
  ],
  "processed_count": 2,
  "failed_count": 0,
  "errors": null
}
```

---

### 5. Análisis de Tendencias

Analizar evolución de síntomas (mínimo 7 días).

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
        "weight_change": 0.0,
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
    "symptom_correlations": null
  },
  "risk_assessment": {
    "flare_risk": "low",
    "probability": 0.30,
    "confidence": 0.75
  },
  "recommendations": [
    "Continue current management plan"
  ]
}
```

**Errores:**
- `400 Bad Request`: Menos de 7 días de datos

---

## 📊 Schemas de Datos

### Symptoms

| Campo | Tipo | Rango | Requerido | Descripción |
|-------|------|-------|-----------|-------------|
| `abdominal_pain` | int | 0-10 | Sí | Escala de dolor |
| `diarrhea` | int | 0-10 | Sí | Severidad |
| `fatigue` | int | 0-10 | Sí | Nivel de fatiga |
| `fever` | bool | - | Sí | Presencia de fiebre |
| `weight_change` | float | - | Sí | Cambio en kg (negativo = pérdida) |
| `blood_in_stool` | bool | - | No (default: false) | Sangre en heces |
| `nausea` | int | 0-10 | No (default: 0) | Nivel de náuseas |

### Demographics

| Campo | Tipo | Rango | Requerido | Descripción |
|-------|------|-------|-----------|-------------|
| `age` | int | 0-120 | Sí | Edad del paciente |
| `gender` | string | M/F/O | Sí | Género |
| `disease_duration_years` | int | ≥0 | Sí | Años desde diagnóstico |
| `bmi` | float | 10-60 | No | Índice de masa corporal |

### History

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `previous_flares` | int | Número de brotes previos |
| `medications` | array[string] | Lista de medicamentos |
| `last_flare_days_ago` | int | Días desde último brote |
| `surgery_history` | bool | Cirugía previa de EII |
| `smoking_status` | string | never/former/current |

---

## 🔒 Autenticación

**Actualmente:** Sin autenticación (server-to-server)

**Futuro:** API Key en header
```http
X-API-Key: your-api-key-here
```

---

## ⚠️ Manejo de Errores

### Códigos de Estado

| Código | Significado |
|--------|-------------|
| 200 | Success |
| 400 | Bad Request (datos inválidos) |
| 422 | Validation Error (Pydantic) |
| 500 | Internal Server Error |
| 503 | Service Unavailable (modelo no cargado) |

### Formato de Error

```json
{
  "detail": "Mensaje de error descriptivo"
}
```

---

## 📈 Rate Limits

**Actual:** Sin límites

**Recomendado para producción:**
- 100 requests/minuto por IP
- 1000 requests/día por API key

---

## 🔄 Versionado de API

**Actual:** v1.0.0

**Futuros cambios:**
- Breaking changes: v2.0.0
- Nuevas features: v1.1.0
- Bug fixes: v1.0.1

---

## 📚 Ejemplos de Uso

Ver archivos:
- `scripts/test_api.py` - Ejemplos en Python
- `scripts/api_examples.json` - Datos de ejemplo
- `scripts/test_api.sh` - Ejemplos con curl

---

## 🌐 CORS

**Origins permitidos:**
- `http://localhost:8000` (Backend web)
- `http://localhost:5173` (Frontend Vue)

Configurable en `.env` con `CORS_ORIGINS`

---

## 📞 Documentación Interactiva

- **Swagger UI**: http://localhost:8001/docs
- **ReDoc**: http://localhost:8001/redoc

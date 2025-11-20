# Scripts de Prueba de API

Esta carpeta contiene scripts para probar los endpoints de la API.

## Archivos

- `test_api.py` - Script Python completo con todos los tests
- `test_api.sh` - Script Bash con ejemplos curl
- `api_examples.json` - Datos de ejemplo en formato JSON

## Uso

### 1. Iniciar el servidor API

Primero, asegúrate de que el servidor está corriendo:

```bash
make serve
```

O directamente:

```bash
uv run uvicorn api.app:app --reload
```

### 2. Ejecutar los tests

#### Opción A: Con Python (Recomendado)

```bash
# Ejecutar todos los tests
uv run python scripts/test_api.py

# O si estás en el entorno virtual
python scripts/test_api.py
```

Este script probará todos los endpoints y mostrará resultados formateados.

#### Opción B: Con Bash/curl

```bash
./scripts/test_api.sh
```

Requiere `jq` instalado para formatear JSON:
```bash
# macOS
brew install jq

# Linux
sudo apt-get install jq
```

### 3. Ejemplos individuales con curl

#### Health Check
```bash
curl http://localhost:8000/health | jq
```

#### Predicción Individual (Bajo Riesgo)
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "symptoms": {
      "abdominal_pain": 2,
      "diarrhea": 3,
      "fatigue": 2,
      "fever": false,
      "weight_change": 0.0
    },
    "demographics": {
      "age": 28,
      "gender": "F",
      "disease_duration_years": 2
    },
    "history": {
      "previous_flares": 1,
      "medications": ["mesalamine"],
      "last_flare_days_ago": 365
    }
  }' | jq
```

#### Predicción Individual (Alto Riesgo)
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "symptoms": {
      "abdominal_pain": 9,
      "diarrhea": 8,
      "fatigue": 7,
      "fever": true,
      "weight_change": -5.0,
      "blood_in_stool": true
    },
    "demographics": {
      "age": 45,
      "gender": "M",
      "disease_duration_years": 10
    },
    "history": {
      "previous_flares": 6,
      "medications": ["infliximab"],
      "last_flare_days_ago": 45
    }
  }' | jq
```

#### Predicción por Lotes
```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "patients": [
      {
        "patient_id": "P001",
        "symptoms": {
          "abdominal_pain": 3,
          "diarrhea": 2,
          "fatigue": 4,
          "fever": false,
          "weight_change": 0.5
        },
        "demographics": {
          "age": 28,
          "gender": "F",
          "disease_duration_years": 2
        },
        "history": {
          "previous_flares": 1,
          "medications": ["mesalamine"],
          "last_flare_days_ago": 365
        }
      },
      {
        "patient_id": "P002",
        "symptoms": {
          "abdominal_pain": 8,
          "diarrhea": 7,
          "fatigue": 6,
          "fever": true,
          "weight_change": -3.0
        },
        "demographics": {
          "age": 35,
          "gender": "M",
          "disease_duration_years": 8
        },
        "history": {
          "previous_flares": 4,
          "medications": ["adalimumab"],
          "last_flare_days_ago": 60
        }
      }
    ]
  }' | jq
```

#### Usando archivos JSON
```bash
# Extraer un ejemplo del archivo
cat scripts/api_examples.json | jq '.predict_low_risk' > /tmp/request.json

# Hacer la petición
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @/tmp/request.json | jq
```

## Documentación Interactiva

Una vez que el servidor esté corriendo, accede a:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

Estas interfaces permiten probar la API interactivamente desde el navegador.

## Endpoints Disponibles

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/model/info` | GET | Información del modelo |
| `/predict` | POST | Predicción individual |
| `/predict/batch` | POST | Predicción por lotes |
| `/analyze/trends` | POST | Análisis de tendencias |

## Estructura de Respuestas

### Predicción Individual
```json
{
  "prediction": {
    "flare_risk": "medium",
    "probability": 0.65,
    "confidence": 0.80
  },
  "factors": {
    "top_contributors": ["abdominal_pain", "diarrhea"],
    "symptom_severity_score": 0.55
  },
  "recommendation": "Monitoree sus sintomas..."
}
```

### Predicción por Lotes
```json
{
  "results": [
    {
      "patient_id": "P001",
      "prediction": {...},
      "factors": {...}
    }
  ],
  "processed_count": 3,
  "failed_count": 0
}
```

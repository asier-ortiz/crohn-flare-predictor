# 🩺 Crohn Flare Predictor

Sistema de Machine Learning para predecir brotes de enfermedad inflamatoria intestinal (EII) basado en el seguimiento diario de síntomas.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/flaredown/flaredown-autoimmune-symptom-tracker)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)

## 📋 Descripción

Servicio ML independiente (microservicio) para predicción de brotes de enfermedad de Crohn. Este proyecto expone una API REST stateless que analiza síntomas diarios y predice riesgos de brotes.

### ✨ Características

- 🤖 **API REST Stateless**: Servicio ML independiente sin base de datos
- 📊 **Predicción Individual**: Análisis de riesgo basado en síntomas actuales
- 📈 **Análisis de Tendencias**: Evaluación temporal de evolución de síntomas
- 🔀 **Predicciones por Lotes**: Procesamiento múltiple para dashboards
- 🚀 **Deploy Fácil**: Dockerizado y listo para producción
- 📚 **Documentación Completa**: Guías de integración y API reference

### 🎯 Propósito del Proyecto

Este servicio está diseñado para **integrarse con una aplicación web** (FastAPI + Vue) desarrollada por el equipo. El ML funciona como un microservicio independiente que el backend web consume vía HTTP.

## 🗂️ Estructura del Proyecto

```
crohn-flare-predictor/
├── api/                          # API REST
│   ├── app.py                   # FastAPI application
│   ├── ml_model.py              # ML model wrapper
│   └── schemas.py               # Pydantic schemas
├── data/                        # Datos (no versionados)
│   ├── raw/                     # Datos sin procesar (export.csv)
│   └── processed/               # Datos procesados
│       ├── crohn/               # Datasets Crohn
│       └── cu/                  # Datasets Colitis Ulcerosa
├── models/                      # Modelos entrenados
│   ├── crohn/                   # Modelos Crohn (cluster-stratified)
│   └── cu/                      # Modelos CU (cluster-stratified)
├── notebooks/                   # Jupyter notebooks
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_advanced_feature_engineering.ipynb
│   ├── 04_cluster_stratified_training.ipynb
│   └── 05_cluster_stratified_training_cu.ipynb
├── scripts/                     # Scripts auxiliares
│   ├── test_api.py             # Test API con Python
│   ├── test_api.sh             # Test API con curl
│   ├── evaluate_model.py       # Evaluación de modelos
│   └── cleanup_local.sh        # Limpieza de archivos generados
├── docs/                        # Documentación
└── reports/                     # Reportes de evaluación
```

## 📚 Documentación

Documentación completa disponible en [`./docs`](docs/):

- **[Guía de Integración](docs/INTEGRATION.md)** - Para desarrolladores del equipo web
- **[Referencia de API](docs/API_REFERENCE.md)** - Documentación completa de endpoints
- **[Arquitectura](docs/ARCHITECTURE.md)** - Decisiones de diseño
- **[Desarrollo](docs/DEVELOPMENT.md)** - Setup y flujo de trabajo
- **[Deployment](docs/DEPLOYMENT.md)** - Guía de despliegue

## 🚀 Instalación

### Requisitos Previos

- Python 3.10 o superior
- [uv](https://docs.astral.sh/uv/) (gestor de paquetes ultrarrápido)

### Instalación de uv

Si no tienes `uv` instalado:

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Con pip (alternativa)
pip install uv
```

### Pasos de Instalación

1. **Clonar el repositorio**
```bash
git clone https://github.com/tu-usuario/crohn-flare-predictor.git
cd crohn-flare-predictor
```

2. **Instalar dependencias**

```bash
# Solo para API en producción (más rápido)
uv sync

# Para desarrollo completo (incluye notebooks y herramientas)
uv sync --group dev --group notebooks

# Para deep learning (opcional, solo si usas TensorFlow/PyTorch)
uv sync --group deep-learning
```

¡Eso es todo! `uv` creará automáticamente el entorno virtual e instalará las dependencias necesarias en segundos.

3. **Configurar variables de entorno** (opcional)

```bash
cp .env.example .env
# Editar .env con tus valores
```

4. **Descargar datos** (solo para entrenamiento)
- Descargar el dataset desde [Kaggle: Flaredown Autoimmune Symptom Tracker](https://www.kaggle.com/datasets/flaredown/flaredown-autoimmune-symptom-tracker)
- Guardar el archivo `export.csv` en `data/raw/`

### Instalación con Docker

```bash
# Build imagen
docker build -t crohn-ml-api .

# Run
docker run -p 8001:8001 crohn-ml-api

# O con docker-compose
docker-compose up
```

### Comandos Rápidos

El proyecto incluye un **Makefile** con comandos útiles:

```bash
# Ver todos los comandos disponibles
make help

# Setup completo (instalar deps + crear directorios)
make dev

# Iniciar Jupyter Notebook
make notebook

# Levantar API REST
make serve

# Ejecutar tests
make test

# Formatear y verificar código
make check
```

También puedes usar `uv` directamente:

```bash
# Ejecutar cualquier comando en el entorno
uv run python script.py

# Iniciar Jupyter
uv run jupyter notebook

# Ejecutar tests
uv run pytest

# Formatear código
uv run black api/ scripts/

# Levantar API REST
uv run uvicorn api.app:app --reload
```

## 📖 Uso

### Análisis Exploratorio

Ejecutar los notebooks en orden:

```bash
jupyter notebook notebooks/01_exploratory_analysis.ipynb
```

Los notebooks incluyen:
- `01_exploratory_analysis.ipynb`: Análisis inicial de datos y visualizaciones
- `02_feature_engineering.ipynb`: Creación y selección de características
- `03_model_training.ipynb`: Entrenamiento y evaluación de modelos

### API REST

#### Iniciar el servidor

Con `uv`, levantar el servidor es muy simple:

```bash
uv run uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

El servidor estará disponible en `http://localhost:8000`

#### Documentación Interactiva

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 🔌 Endpoints de la API

### Health Check

```http
GET /health
```

Verifica el estado del servicio.

**Respuesta:**
```json
{
  "status": "healthy",
  "version": "1.0.0"
}
```

### Predicción Individual

```http
POST /predict
```

Realiza una predicción de brote basada en síntomas.

**Request Body:**
```json
{
  "symptoms": {
    "abdominal_pain": 7,
    "diarrhea": 5,
    "fatigue": 6,
    "fever": false,
    "weight_change": -2.5
  },
  "demographics": {
    "age": 32,
    "gender": "F",
    "disease_duration_years": 5
  },
  "history": {
    "previous_flares": 3,
    "medications": ["mesalamine"],
    "last_flare_days_ago": 120
  }
}
```

**Respuesta:**
```json
{
  "prediction": {
    "flare_risk": "high",
    "probability": 0.78,
    "confidence": 0.85
  },
  "factors": {
    "top_contributors": [
      "abdominal_pain",
      "symptom_trend",
      "previous_flares"
    ]
  },
  "recommendation": "Consulte con su médico. Se recomienda evaluación temprana."
}
```

### Predicción por Lotes

```http
POST /predict/batch
```

Realiza predicciones múltiples en una sola petición.

**Request Body:**
```json
{
  "patients": [
    {
      "patient_id": "P001",
      "symptoms": {...},
      "demographics": {...}
    },
    {
      "patient_id": "P002",
      "symptoms": {...},
      "demographics": {...}
    }
  ]
}
```

### Análisis de Tendencias

```http
POST /analyze/trends
```

Analiza tendencias temporales de síntomas.

**Request Body:**
```json
{
  "patient_id": "P001",
  "daily_records": [
    {
      "date": "2024-01-01",
      "symptoms": {...}
    },
    {
      "date": "2024-01-02",
      "symptoms": {...}
    }
  ],
  "window_days": 14
}
```

### Obtener Información del Modelo

```http
GET /model/info
```

Retorna información sobre el modelo activo.

**Respuesta:**
```json
{
  "model_version": "1.0.0",
  "trained_date": "2024-01-15",
  "metrics": {
    "accuracy": 0.87,
    "precision": 0.84,
    "recall": 0.89,
    "f1_score": 0.86
  },
  "features_count": 45
}
```

## 🧪 Testing

Ejecutar los tests:

```bash
pytest
```

Con cobertura:

```bash
pytest --cov=api --cov-report=html --cov-report=term
```

## 🔧 Desarrollo

### Formato de Código

```bash
black api/ scripts/
flake8 api/ scripts/
```

### Variables de Entorno

Crear archivo `.env`:

```bash
MODEL_PATH=./models/crohn_predictor.pkl
LOG_LEVEL=INFO
API_VERSION=1.0.0
```

## 📊 Dataset

El proyecto utiliza el dataset de Gastrointestinal Disease de Kaggle, que incluye:
- Datos de seguimiento diario de síntomas
- Información demográfica de pacientes
- Historial de brotes y medicaciones
- Biomarcadores y resultados de laboratorio

## 📝 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## 📧 Contacto

Para preguntas o sugerencias, por favor abre un issue en el repositorio.

## ⚠️ Disclaimer

Este software es solo para fines de investigación y educativos. No debe utilizarse como sustituto del consejo médico profesional, diagnóstico o tratamiento. Siempre consulte con un profesional de la salud calificado.

## 🙏 Agradecimientos

- Dataset proporcionado por la comunidad de Kaggle
- Bibliotecas de código abierto: scikit-learn, pandas, FastAPI
- Comunidad de investigación en EII

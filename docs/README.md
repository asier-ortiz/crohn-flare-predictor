# 📚 Documentación - Crohn Flare Predictor ML API

Documentación completa del microservicio ML para predicción de brotes de enfermedad inflamatoria intestinal.

## 🎯 Propósito

Este microservicio ML está diseñado para integrarse con una aplicación web de seguimiento de EII desarrollada por el equipo. El servicio funciona de manera independiente y stateless, recibiendo datos de síntomas y devolviendo predicciones de riesgo de brotes.

**Stack del proyecto completo:**
- **ML API** (este repo): FastAPI + scikit-learn + Random Forest cluster-stratified
- **Backend Web**: FastAPI + MySQL + JWT Auth
- **Frontend**: Vue.js + HTML5 + CSS3

---

## 📖 Guías Disponibles

### Para Desarrolladores del Equipo Web

#### 1. [**Guía de la Aplicación Web**](WEB_APP_GUIDE.md) 🌟

**Para Cristina, Carlos y todo el equipo de desarrollo web**

Guía completa para desarrollar la aplicación web de seguimiento de EII:
- 📋 Contexto del proyecto TFG
- 🗄️ Esquema de base de datos MySQL (users, daily_symptoms, meals, exercise_log, flare_predictions)
- 📱 Pantallas sugeridas (login, dashboard, registro diario, historial, patrones, perfil)
- 🎨 Mockups y wireframes de UI
- 🔄 Flujos de usuario completos
- 🛠️ Estructura técnica (FastAPI + Vue + MySQL)
- 🚀 Deployment con Docker Compose

**Empieza por aquí si estás desarrollando la app web.**

---

#### 2. [**Guía de Integración**](INTEGRATION.md)

**Cómo integrar el ML API en tu backend FastAPI**

Tutorial paso a paso para conectar tu backend web con este microservicio ML:
- 🏗️ Arquitectura de integración
- 💻 Implementación del cliente HTTP (`ml_client.py`)
- 📝 Ejemplos de endpoints (`/api/symptoms/daily`, `/api/dashboard`)
- 🚨 Manejo de errores y graceful degradation
- 🧪 Tests de integración
- 🔐 Seguridad y rate limiting
- ❓ FAQ y troubleshooting

**Lee esto cuando vayas a implementar las llamadas al ML API.**

---

#### 3. [**Referencia de API**](API_REFERENCE.md)

**Documentación completa de todos los endpoints**

Referencia técnica detallada:
- 📍 Todos los endpoints disponibles:
  - `GET /health` - Health check
  - `POST /predict` - Predicción individual (⭐ más importante)
  - `POST /predict/batch` - Predicciones por lotes
  - `POST /analyze/trends` - Análisis temporal de síntomas
  - `GET /model/info` - Información del modelo
- 📊 Schemas completos de request/response
- 💡 Ejemplos en Python, cURL y JavaScript
- ⚠️ Códigos de error y manejo
- 🧪 Scripts de testing

**Consúltala cuando necesites detalles específicos de un endpoint.**

---

## 🚀 Quick Start

### Levantar el ML API

```bash
# 1. Clonar el repositorio
git clone https://github.com/tu-usuario/crohn-flare-predictor.git
cd crohn-flare-predictor

# 2. Instalar dependencias con uv
uv sync

# 3. Iniciar el servidor
uv run uvicorn api.app:app --reload --host 0.0.0.0 --port 8001

# O con Makefile
make serve

# 4. Verificar que funciona
curl http://localhost:8001/health
# {"status":"healthy","version":"1.0.0"}
```

### Probar el API

```bash
# Ejemplo rápido de predicción
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

### Documentación Interactiva

Con el servidor corriendo:
- **Swagger UI**: http://localhost:8001/docs
- **ReDoc**: http://localhost:8001/redoc

---

## 📂 Estructura del Repositorio

```
crohn-flare-predictor/
├── api/                          # API REST (FastAPI)
│   ├── app.py                   # Aplicación principal
│   ├── ml_model.py              # Wrapper de modelos ML
│   ├── schemas.py               # Pydantic schemas
│   ├── config.py                # Configuración
│   └── constants.py             # Constantes (descripciones de clusters)
├── data/                        # Datos (gitignored)
│   ├── raw/                     # Dataset Kaggle (export.csv)
│   └── processed/               # Datos procesados
│       ├── crohn/               # Datasets Crohn (L1-L4)
│       └── cu/                  # Datasets Colitis Ulcerosa (E1-E3)
├── models/                      # Modelos entrenados
│   ├── crohn/                   # Modelos cluster-stratified Crohn
│   │   ├── cluster_*.pkl        # 3 modelos (L1, L2/L3, L4)
│   │   └── *_metadata.json      # Metadata de modelos
│   └── cu/                      # Modelos cluster-stratified CU
│       ├── cluster_*.pkl        # 3 modelos (E1, E2, E3)
│       └── *_metadata.json      # Metadata de modelos
├── notebooks/                   # Jupyter notebooks (desarrollo ML)
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_advanced_feature_engineering.ipynb
│   ├── 04_cluster_stratified_training.ipynb      # Crohn
│   └── 05_cluster_stratified_training_cu.ipynb    # CU
├── scripts/                     # Scripts auxiliares
│   ├── test_api.py             # Test del API (Python)
│   ├── test_api.sh             # Test del API (curl)
│   ├── evaluate_model.py       # Evaluación de modelos
│   └── cleanup_local.sh        # Limpieza de archivos generados
├── docs/                        # Documentación (aquí estás)
│   ├── README.md               # Este archivo
│   ├── WEB_APP_GUIDE.md        # Guía de la app web (⭐ importante)
│   ├── INTEGRATION.md          # Cómo integrar el ML API
│   └── API_REFERENCE.md        # Referencia técnica de endpoints
├── Dockerfile                   # Para despliegue en contenedor
├── Makefile                     # Comandos útiles (make serve, etc.)
├── pyproject.toml               # Configuración uv y dependencias
└── README.md                    # README principal del proyecto
```

---

## 🎓 Para Estudiantes del TFG

### Flujo de Trabajo Sugerido

1. **Lee primero:** [WEB_APP_GUIDE.md](WEB_APP_GUIDE.md)
   - Entiende el proyecto completo
   - Revisa el esquema de BD
   - Ve los mockups de pantallas

2. **Desarrolla la app web:**
   - Backend FastAPI con MySQL
   - Frontend Vue.js
   - Sistema de autenticación JWT

3. **Integra el ML API:** [INTEGRATION.md](INTEGRATION.md)
   - Implementa `ml_client.py` en tu backend
   - Añade predicciones en `/api/symptoms/daily`
   - Muestra predicciones en el dashboard

4. **Consulta cuando sea necesario:** [API_REFERENCE.md](API_REFERENCE.md)
   - Detalles técnicos de endpoints
   - Schemas exactos
   - Ejemplos de uso

### División del Trabajo

**Sugerencia de roles** (ajustar según equipo):

- **Backend Web (FastAPI + MySQL)**:
  - Setup de BD (usuarios, síntomas, comidas, ejercicio)
  - Autenticación JWT
  - Endpoints CRUD
  - Integración con ML API (`ml_client.py`)

- **Frontend (Vue.js)**:
  - Componentes reutilizables
  - Pantallas (login, dashboard, formularios)
  - Gráficas (Chart.js / ApexCharts)
  - Comunicación con backend (axios)

- **ML / DevOps** (este repositorio):
  - Entrenar modelos (notebooks)
  - Mantener ML API corriendo
  - Docker / deployment
  - Testing de integración

---

## 📊 Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────┐
│                        USUARIO                              │
│                    (Paciente con EII)                       │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   FRONTEND (Vue.js)                         │
│  ┌──────────┬──────────┬──────────┬──────────┬──────────┐  │
│  │  Login   │Dashboard │ Registro │Historial │ Perfil   │  │
│  │          │ Gráficas │  Diario  │          │          │  │
│  └──────────┴──────────┴──────────┴──────────┴──────────┘  │
│                     http://localhost:5173                   │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP/JSON
                         ▼
┌─────────────────────────────────────────────────────────────┐
│               BACKEND WEB (FastAPI)                         │
│  ┌────────────────────────────────────────────────────┐     │
│  │  • JWT Authentication                              │     │
│  │  • Endpoints: /api/symptoms, /api/dashboard, etc. │     │
│  │  • ml_client.py (cliente HTTP)                     │     │
│  └────────────────────┬───────────────────────────────┘     │
│                     http://localhost:8000                   │
└────────┬───────────────┴─────────────────────┬──────────────┘
         │                                     │
         │ SQL                                 │ HTTP/JSON
         ▼                                     ▼
┌──────────────────┐              ┌──────────────────────────┐
│  MySQL Database  │              │  ML API (Este repo)      │
│                  │              │                          │
│  • users         │              │  • POST /predict         │
│  • daily_symptoms│              │  • GET /health           │
│  • meals         │              │  • Modelos RF            │
│  • exercise_log  │              │  • Cluster-stratified    │
│  • predictions   │              │                          │
│                  │              │  http://localhost:8001   │
└──────────────────┘              └──────────────────────────┘
```

---

## 🔬 Sobre el Modelo ML

### Características Técnicas

- **Tipo**: Random Forest Classifier (cluster-stratified)
- **Features**: 34 features totales
  - 13 base features (síntomas + demografía + historial)
  - 21 derived features (agregaciones, tendencias, interacciones)
- **Output**: Riesgo de brote (low/medium/high) con probabilidad y confianza
- **Accuracy**: 99.22% (Crohn), 98.5% (CU)
- **Recall para alto riesgo**: 100% (no se pierde ningún brote real)

### Cluster Stratification

El modelo usa modelos especializados según la clasificación de Montreal:

**Crohn Disease:**
- L1 (ileal) → Cluster 0
- L2 (colónico) → Cluster 1
- L3 (ileocolónico) → Cluster 1
- L4 (gastrointestinal superior) → Cluster 2

**Ulcerative Colitis:**
- E1 (proctitis) → Cluster 0
- E2 (colitis izquierda) → Cluster 1
- E3 (colitis extensa/pancolitis) → Cluster 2

Esto permite predicciones más precisas al adaptar el modelo a diferentes fenotipos de la enfermedad.

---

## 🧪 Testing

### Tests del ML API

```bash
# Test completo con Python
uv run python scripts/test_api.py

# Test rápido con curl
bash scripts/test_api.sh

# Evaluación de modelos (8 casos diversos)
uv run python scripts/evaluate_model.py
```

### Tests de Integración

Ver ejemplos en [INTEGRATION.md](INTEGRATION.md) para:
- Test de health check
- Test de predicción
- Test de degradación graceful (cuando ML API falla)

---

## 🐳 Deployment

### Con Docker

```bash
# Build
docker build -t crohn-ml-api .

# Run
docker run -p 8001:8001 crohn-ml-api
```

### Con Docker Compose (app completa)

Ver ejemplo en [WEB_APP_GUIDE.md](WEB_APP_GUIDE.md) que incluye:
- MySQL
- Backend Web
- ML API
- Frontend

---

## 📝 Dataset

**Fuente**: [Flaredown Autoimmune Symptom Tracker](https://www.kaggle.com/datasets/flaredown/flaredown-autoimmune-symptom-tracker)

El dataset contiene seguimiento diario de síntomas de pacientes con EII y otras enfermedades autoinmunes. Para este proyecto se filtraron solo pacientes con Crohn y Colitis Ulcerosa.

**⚠️ Importante:** El archivo `data/raw/export.csv` (~600MB) no está en git. Descárgalo desde Kaggle para entrenar modelos.

---

## ⚠️ Disclaimer

Este software es solo para fines de investigación y educativos. **NO debe utilizarse como sustituto del consejo médico profesional, diagnóstico o tratamiento.** Siempre consulte con un profesional de la salud calificado.

---

## 🤝 Contribuir

Este es un proyecto TFG académico. Para dudas o mejoras:
- Abre un issue en el repositorio
- Contacta directamente con el equipo

---

## 📧 Contacto

- **Asier** (ML / ML API) - Este repositorio
- **Cristina** (Web App / Frontend)
- **Carlos** (Web App / Backend)

---

## 📚 Enlaces Útiles

### Documentación

- [Guía de la Aplicación Web](WEB_APP_GUIDE.md)
- [Guía de Integración](INTEGRATION.md)
- [Referencia de API](API_REFERENCE.md)

### Recursos Técnicos

- [FastAPI](https://fastapi.tiangolo.com/)
- [Vue.js 3](https://vuejs.org/)
- [scikit-learn](https://scikit-learn.org/)
- [uv Package Manager](https://docs.astral.sh/uv/)
- [Montreal Classification](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2807799/) (clasificación de EII)

### Herramientas

- **Swagger UI**: http://localhost:8001/docs
- **ReDoc**: http://localhost:8001/redoc
- **Scripts**: Ver `scripts/` directory

---

**¡Buena suerte con el TFG! 🎓🚀**

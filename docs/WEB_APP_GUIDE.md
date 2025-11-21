# 🌐 Guía de la Aplicación Web - Proyecto TFG EII

Guía completa para el desarrollo de la aplicación web de seguimiento de Enfermedad Inflamatoria Intestinal (EII).

## 📋 Contexto del Proyecto

Esta aplicación web facilita el seguimiento de síntomas a personas con enfermedad inflamatoria intestinal (Crohn o Colitis Ulcerosa). Los pacientes pueden:

- ✅ **Registrar síntomas diarios** (dolor abdominal, diarrea, fatiga, fiebre, etc.)
- 🍽️ **Registrar alimentación** (qué han comido cada día)
- 🏃 **Registrar actividad física** (si han hecho ejercicio)
- 📊 **Ver gráfica mensual** (del 1 al 10 cómo ha estado cada día)
- 🔮 **Obtener predicciones de brotes** (usando el ML API)
- 📈 **Detectar patrones** (alimentos o rutinas que no sientan bien)

### Stack Tecnológico

- **Backend**: Python + FastAPI
- **Frontend**: Vue.js + HTML5 + CSS3 + JavaScript
- **Estilos**: Bootstrap o TailwindCSS
- **Base de Datos**: MySQL
- **ML API**: Microservicio independiente (este repositorio)
- **Autenticación**: JWT tokens

---

## 🗄️ Esquema de Base de Datos

### Diagrama ER Simplificado

```
┌─────────────┐
│   users     │
├─────────────┤
│ id (PK)     │──┐
│ email       │  │
│ password    │  │
│ name        │  │
│ age         │  │
│ gender      │  │
│ ibd_type    │  │  (crohn/ulcerative_colitis)
│ ...         │  │
└─────────────┘  │
                 │
      ┌──────────┴─────────────┬──────────────┬─────────────┐
      │                        │              │             │
      ▼                        ▼              ▼             ▼
┌──────────────┐   ┌──────────────┐   ┌─────────────┐   ┌─────────────┐
│ daily_       │   │    meals     │   │  exercise_  │   │   flare_    │
│ symptoms     │   │              │   │    log      │   │ predictions │
├──────────────┤   ├──────────────┤   ├─────────────┤   ├─────────────┤
│ id (PK)      │   │ id (PK)      │   │ id (PK)     │   │ id (PK)     │
│ user_id (FK) │   │ user_id (FK) │   │ user_id(FK) │   │ user_id(FK) │
│ record_date  │   │ meal_date    │   │ exercise_dt │   │ symptom_id  │
│ abdominal_   │   │ meal_type    │   │ exercise_   │   │ created_at  │
│   pain       │   │ food_items   │   │   type      │   │ flare_risk  │
│ diarrhea     │   │ notes        │   │ duration    │   │ probability │
│ fatigue      │   │ ...          │   │ intensity   │   │ confidence  │
│ fever        │   │              │   │ ...         │   │ factors     │
│ wellness_    │   │              │   │             │   │ ...         │
│   score      │   │              │   │             │   │             │
│ ...          │   │              │   │             │   │             │
└──────────────┘   └──────────────┘   └─────────────┘   └─────────────┘
```

### Tablas Principales

#### 1. `users` - Datos del usuario y perfil médico

```sql
CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    name VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    -- Datos demográficos
    age INT NOT NULL,
    gender ENUM('M', 'F', 'O') NOT NULL,
    bmi DECIMAL(4,1),

    -- Datos médicos
    ibd_type ENUM('crohn', 'ulcerative_colitis') NOT NULL,
    montreal_classification VARCHAR(10),  -- L1, L2, L3, L4, E1, E2, E3
    disease_duration_years INT NOT NULL,
    diagnosis_date DATE,

    -- Historial médico
    previous_flares INT DEFAULT 0,
    last_flare_date DATE,
    surgery_history BOOLEAN DEFAULT FALSE,
    smoking_status ENUM('never', 'former', 'current') DEFAULT 'never',

    -- Medicación actual (JSON array)
    current_medications JSON,  -- ["mesalamine", "prednisone"]

    INDEX idx_email (email),
    INDEX idx_ibd_type (ibd_type)
);
```

#### 2. `daily_symptoms` - Registro diario de síntomas

```sql
CREATE TABLE daily_symptoms (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    record_date DATE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Síntomas principales (escala 0-10)
    abdominal_pain INT CHECK (abdominal_pain BETWEEN 0 AND 10),
    diarrhea INT CHECK (diarrhea BETWEEN 0 AND 10),
    fatigue INT CHECK (fatigue BETWEEN 0 AND 10),
    nausea INT CHECK (nausea BETWEEN 0 AND 10),

    -- Síntomas booleanos
    fever BOOLEAN DEFAULT FALSE,
    blood_in_stool BOOLEAN DEFAULT FALSE,

    -- Otros
    weight_kg DECIMAL(5,2),
    weight_change DECIMAL(4,2),  -- Cambio respecto al día anterior

    -- Puntuación de bienestar general (1-10)
    wellness_score INT CHECK (wellness_score BETWEEN 1 AND 10),

    -- Notas del paciente
    notes TEXT,

    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    UNIQUE KEY unique_user_date (user_id, record_date),
    INDEX idx_user_date (user_id, record_date)
);
```

#### 3. `meals` - Registro de comidas

```sql
CREATE TABLE meals (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    meal_date DATE NOT NULL,
    meal_time TIME,
    meal_type ENUM('breakfast', 'lunch', 'dinner', 'snack') NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Comida
    food_items TEXT NOT NULL,  -- Lista de alimentos

    -- Reacción (opcional, se llena después)
    caused_symptoms BOOLEAN DEFAULT FALSE,
    symptom_notes TEXT,

    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    INDEX idx_user_date (user_id, meal_date)
);
```

#### 4. `exercise_log` - Registro de ejercicio

```sql
CREATE TABLE exercise_log (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    exercise_date DATE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Ejercicio
    exercise_type VARCHAR(50),  -- walking, running, yoga, swimming, etc.
    duration_minutes INT,
    intensity ENUM('light', 'moderate', 'vigorous'),

    -- Notas
    notes TEXT,

    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    INDEX idx_user_date (user_id, exercise_date)
);
```

#### 5. `flare_predictions` - Caché de predicciones ML

```sql
CREATE TABLE flare_predictions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    symptom_record_id INT NOT NULL,  -- Referencia a daily_symptoms
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Predicción
    flare_risk ENUM('low', 'medium', 'high') NOT NULL,
    probability DECIMAL(4,3) NOT NULL,  -- 0.000 - 1.000
    confidence DECIMAL(4,3) NOT NULL,

    -- Metadata
    top_contributors JSON,  -- ["abdominal_pain", "diarrhea"]
    recommendation TEXT,

    -- Cluster info (si se usa modelo cluster-stratified)
    cluster_id INT,
    cluster_confidence DECIMAL(4,3),

    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (symptom_record_id) REFERENCES daily_symptoms(id) ON DELETE CASCADE,
    INDEX idx_user_created (user_id, created_at)
);
```

---

## 📱 Pantallas Sugeridas

### 1. **Autenticación**

#### Login (`/login`)
- Email
- Contraseña
- "Olvidé mi contraseña"
- Link a registro

#### Registro (`/register`)
- Datos personales (nombre, email, contraseña)
- Datos demográficos (edad, género, BMI)
- Datos médicos:
  - Tipo de EII (Crohn / Colitis Ulcerosa)
  - Clasificación de Montreal (L1-L4 / E1-E3)
  - Años desde diagnóstico
  - Número de brotes previos
  - Fecha del último brote
  - Medicación actual (multi-select)
  - ¿Has tenido cirugías?
  - Estado de fumador

---

### 2. **Dashboard Principal** (`/dashboard`)

Vista principal tras login. Muestra resumen del mes actual.

**Elementos:**

```
┌──────────────────────────────────────────────────────────┐
│  Dashboard - Noviembre 2024                       👤 User │
├──────────────────────────────────────────────────────────┤
│                                                            │
│  📊 Gráfica Mensual (1-10)                                │
│  ┌────────────────────────────────────────────────────┐   │
│  │ 10 │                                                │   │
│  │  9 │                                                │   │
│  │  8 │        ●                                       │   │
│  │  7 │    ●       ●                                   │   │
│  │  6 │                                                │   │
│  │  5 │                    ●   ●       ●               │   │
│  │  4 │                                                │   │
│  │  3 │                                        ●       │   │
│  │  2 │                                                │   │
│  │  1 │                                                │   │
│  │    └────┬────┬────┬────┬────┬────┬────┬────────   │   │
│  │         1    5    10   15   20   25   30          │   │
│  └────────────────────────────────────────────────────┘   │
│                                                            │
│  🔔 Alertas                                                │
│  ┌────────────────────────────────────────────────────┐   │
│  │ ⚠️ Riesgo ALTO de brote (78% probabilidad)         │   │
│  │ Factores: dolor abdominal, tendencia síntomas      │   │
│  │ Recomendación: Consulte con su médico              │   │
│  └────────────────────────────────────────────────────┘   │
│                                                            │
│  📈 Resumen del mes                                        │
│  - Días registrados: 22/30                                │
│  - Días buenos (8-10): 8                                  │
│  - Días malos (1-4): 3                                    │
│  - Promedio de bienestar: 6.8/10                          │
│                                                            │
│  [➕ Registrar síntomas de hoy]                           │
│                                                            │
└──────────────────────────────────────────────────────────┘
```

**Funcionalidades:**
- Gráfica interactiva (hover muestra detalles del día)
- Selector de mes (navegación)
- Botón "Descargar PDF" (genera PDF del mes para médico)
- Alertas de riesgo de brote (si la predicción ML es "high")
- Botón rápido para registrar síntomas del día

---

### 3. **Registro Diario** (`/daily-log`)

Formulario para registrar el día. Organizado en pestañas/secciones.

#### Pestaña 1: Síntomas

```
┌──────────────────────────────────────────────────────────┐
│  Registro Diario - 21 Nov 2024                            │
├──────────────────────────────────────────────────────────┤
│                                                            │
│  [Síntomas] [Comidas] [Ejercicio]                         │
│                                                            │
│  Síntomas Principales (0 = nada, 10 = máximo)            │
│                                                            │
│  Dolor abdominal:  [========>-----] 7                     │
│  Diarrea:          [======>-------] 6                     │
│  Fatiga:           [=====>--------] 5                     │
│  Náuseas:          [===>----------] 3                     │
│                                                            │
│  Otros síntomas:                                          │
│  ☐ Fiebre                                                 │
│  ☑ Sangre en heces                                        │
│                                                            │
│  Peso actual: [___] kg                                    │
│                                                            │
│  ¿Cómo te has sentido hoy en general? (1-10)             │
│  [1] [2] [3] [4] [5] [6] [7] [8] [9] [10]                │
│                                                            │
│  Notas adicionales:                                       │
│  ┌────────────────────────────────────────────────────┐   │
│  │                                                     │   │
│  └────────────────────────────────────────────────────┘   │
│                                                            │
│              [Guardar y ver predicción]                   │
│                                                            │
└──────────────────────────────────────────────────────────┘
```

#### Pestaña 2: Comidas

```
│  [Síntomas] [Comidas] [Ejercicio]                         │
│                                                            │
│  🌅 Desayuno (08:00)                                      │
│  Alimentos: [____________________________________]         │
│  Añadir: [+ Leche] [+ Pan] [+ Huevos] [+ Custom]         │
│                                                            │
│  🌞 Comida (14:00)                                        │
│  Alimentos: [____________________________________]         │
│                                                            │
│  🌙 Cena (21:00)                                          │
│  Alimentos: [____________________________________]         │
│                                                            │
│  🍎 Snacks                                                │
│  [+ Añadir snack]                                         │
│                                                            │
│  ¿Alguna comida causó síntomas?                           │
│  ☐ Sí  Notas: [___________________________]              │
│                                                            │
```

#### Pestaña 3: Ejercicio

```
│  [Síntomas] [Comidas] [Ejercicio]                         │
│                                                            │
│  ¿Hiciste ejercicio hoy?                                  │
│  ◉ Sí  ○ No                                               │
│                                                            │
│  Tipo de ejercicio:                                       │
│  [Caminar ▼]  (caminar, correr, nadar, yoga, gym, etc.)  │
│                                                            │
│  Duración: [__30__] minutos                               │
│                                                            │
│  Intensidad:                                              │
│  ○ Ligera  ◉ Moderada  ○ Vigorosa                         │
│                                                            │
│  Notas:                                                   │
│  ┌────────────────────────────────────────────────────┐   │
│  │                                                     │   │
│  └────────────────────────────────────────────────────┘   │
│                                                            │
```

**Flujo:**
1. Usuario completa formulario
2. Al guardar, backend:
   - Guarda en BD (daily_symptoms, meals, exercise_log)
   - Llama al ML API para predicción
   - Guarda predicción en caché (flare_predictions)
   - Devuelve todo al frontend
3. Frontend muestra modal con predicción:

```
┌──────────────────────────────────────────────────────────┐
│  ✅ Registro guardado                                     │
├──────────────────────────────────────────────────────────┤
│                                                            │
│  🔮 Predicción de brote                                   │
│                                                            │
│  Riesgo: 🟡 MEDIO                                         │
│  Probabilidad: 65%                                        │
│  Confianza: 80%                                           │
│                                                            │
│  Principales factores:                                    │
│  - Dolor abdominal (7/10)                                │
│  - Sangre en heces                                       │
│  - Tendencia últimos 7 días                              │
│                                                            │
│  💡 Recomendación:                                        │
│  Monitoree sus síntomas de cerca. Considere contactar    │
│  a su médico si empeoran.                                │
│                                                            │
│                         [Entendido]                       │
│                                                            │
└──────────────────────────────────────────────────────────┘
```

---

### 4. **Historial** (`/history`)

Vista de todos los registros pasados con filtros.

```
┌──────────────────────────────────────────────────────────┐
│  Historial de Registros                                   │
├──────────────────────────────────────────────────────────┤
│                                                            │
│  Filtros: [Todo ▼] [Nov 2024 ▼] [Buscar: ______]         │
│                                                            │
│  ┌────────────────────────────────────────────────────┐   │
│  │ 📅 21 Nov 2024               Bienestar: 6/10       │   │
│  │ Síntomas: Dolor (7), Diarrea (6), Fatiga (5)       │   │
│  │ Riesgo: 🟡 MEDIO (65%)                             │   │
│  │                              [Ver detalle] [Editar] │   │
│  ├────────────────────────────────────────────────────┤   │
│  │ 📅 20 Nov 2024               Bienestar: 8/10       │   │
│  │ Síntomas: Leves                                    │   │
│  │ Riesgo: 🟢 BAJO (20%)                              │   │
│  │                              [Ver detalle] [Editar] │   │
│  ├────────────────────────────────────────────────────┤   │
│  │ 📅 19 Nov 2024               Bienestar: 7/10       │   │
│  │ ...                                                │   │
│  └────────────────────────────────────────────────────┘   │
│                                                            │
│  [Cargar más...]                                          │
│                                                            │
└──────────────────────────────────────────────────────────┘
```

**Funcionalidades:**
- Filtrar por mes
- Filtrar por nivel de riesgo (todos, alto, medio, bajo)
- Buscar por notas
- Ver detalle completo de un día
- Editar registros pasados
- Exportar a PDF/CSV

---

### 5. **Análisis de Patrones** (`/patterns`)

Detectar correlaciones entre comidas/ejercicio y síntomas.

```
┌──────────────────────────────────────────────────────────┐
│  Análisis de Patrones                                     │
├──────────────────────────────────────────────────────────┤
│                                                            │
│  📊 Tendencia últimos 30 días                             │
│  [Gráfica de líneas con tendencia]                        │
│                                                            │
│  🍽️ Alimentos que podrían causar síntomas                │
│  ┌────────────────────────────────────────────────────┐   │
│  │ 🥛 Leche               → 3 veces seguido de síntomas│   │
│  │ 🍕 Pizza              → 2 veces seguido de síntomas│   │
│  │ 🌶️ Picante            → 2 veces seguido de síntomas│   │
│  └────────────────────────────────────────────────────┘   │
│                                                            │
│  💪 Ejercicio y bienestar                                 │
│  ┌────────────────────────────────────────────────────┐   │
│  │ Días con ejercicio:     Bienestar promedio: 7.8    │   │
│  │ Días sin ejercicio:     Bienestar promedio: 6.2    │   │
│  └────────────────────────────────────────────────────┘   │
│                                                            │
│  📈 Estadísticas del mes                                  │
│  - Mejor racha: 7 días consecutivos buenos               │
│  - Peor semana: 15-21 Nov                                │
│  - Síntoma más frecuente: Fatiga                         │
│                                                            │
└──────────────────────────────────────────────────────────┘
```

---

### 6. **Perfil Médico** (`/profile`)

Editar información médica y descargar informes.

```
┌──────────────────────────────────────────────────────────┐
│  Mi Perfil Médico                                         │
├──────────────────────────────────────────────────────────┤
│                                                            │
│  👤 Datos Personales                                      │
│  Nombre: [____________________]                           │
│  Email: [_____________________]                           │
│  Edad: [__] años                                          │
│  Género: [Femenino ▼]                                     │
│  BMI: [__.__]                                             │
│                                                            │
│  🏥 Información Médica                                    │
│  Tipo de EII: [Crohn ▼]                                   │
│  Clasificación Montreal: [L3 ▼]                           │
│  Años desde diagnóstico: [__]                             │
│  Último brote: [___/___/____]                             │
│  Cirugías previas: ☐ Sí  ☑ No                            │
│                                                            │
│  💊 Medicación Actual                                     │
│  [Mesalamine] [X]                                         │
│  [Prednisone] [X]                                         │
│  [+ Añadir medicamento]                                   │
│                                                            │
│  📄 Informes para el médico                               │
│  [📥 Descargar informe del mes]                           │
│  [📥 Descargar historial completo (PDF)]                  │
│                                                            │
│                    [Guardar cambios]                      │
│                                                            │
└──────────────────────────────────────────────────────────┘
```

---

## 🔄 Flujo de Integración con ML API

### Flujo 1: Registro de síntomas diarios

```
[Usuario completa formulario]
         │
         ▼
[Frontend envía a Backend Web]
  POST /api/symptoms/daily
         │
         ▼
[Backend Web]
  1. Validar datos
  2. Guardar en BD (daily_symptoms, meals, exercise_log)
  3. Obtener demographics + history del user
         │
         ▼
[Backend Web → ML API]
  POST http://localhost:8001/predict
  {
    "symptoms": {...},
    "demographics": {...},
    "history": {...}
  }
         │
         ▼
[ML API procesa y devuelve]
  {
    "prediction": {
      "flare_risk": "medium",
      "probability": 0.65,
      "confidence": 0.80
    },
    "factors": {...},
    "recommendation": "..."
  }
         │
         ▼
[Backend Web]
  1. Guardar predicción en BD (flare_predictions)
  2. Devolver todo al frontend
         │
         ▼
[Frontend muestra]
  - Confirmación de guardado
  - Predicción de riesgo
  - Recomendación
```

### Flujo 2: Dashboard (al hacer login)

```
[Usuario hace login]
         │
         ▼
[Frontend → Backend Web]
  GET /api/dashboard
         │
         ▼
[Backend Web]
  1. Obtener últimos 30 días de daily_symptoms
  2. Obtener predicciones cacheadas
         │
         ▼
[Backend Web → ML API] (opcional)
  POST /analyze/trends
  {
    "patient_id": "123",
    "daily_records": [últimos 14 días],
    "window_days": 14
  }
         │
         ▼
[ML API devuelve análisis de tendencias]
  {
    "trends": {
      "overall_trend": "stable",
      "severity_change": 0.05,
      ...
    },
    "risk_assessment": {...}
  }
         │
         ▼
[Backend Web agrega y devuelve]
  {
    "monthly_data": [...],
    "cached_predictions": [...],
    "trend_analysis": {...},
    "summary": {...}
  }
         │
         ▼
[Frontend renderiza dashboard]
  - Gráfica mensual
  - Alertas
  - Estadísticas
```

---

## 🛠️ Implementación Técnica

### Backend Web (FastAPI)

**Estructura sugerida:**

```
crohn-web-app/
├── backend/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── auth.py              # Login, registro, JWT
│   │   ├── users.py             # Perfil de usuario
│   │   ├── symptoms.py          # Registro diario de síntomas
│   │   ├── meals.py             # Registro de comidas
│   │   ├── exercise.py          # Registro de ejercicio
│   │   ├── dashboard.py         # Dashboard y estadísticas
│   │   ├── patterns.py          # Análisis de patrones
│   │   └── ml_client.py         # Cliente HTTP para ML API
│   ├── db/
│   │   ├── __init__.py
│   │   ├── database.py          # Conexión MySQL
│   │   ├── models.py            # Modelos SQLAlchemy
│   │   └── schemas.py           # Pydantic schemas
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py            # Configuración
│   │   └── security.py          # JWT, hashing
│   └── main.py                  # FastAPI app
├── frontend/                    # Vue.js app
└── requirements.txt
```

### Frontend (Vue.js)

**Estructura sugerida:**

```
frontend/
├── src/
│   ├── components/
│   │   ├── auth/
│   │   │   ├── LoginForm.vue
│   │   │   └── RegisterForm.vue
│   │   ├── dashboard/
│   │   │   ├── MonthlyChart.vue
│   │   │   ├── AlertsCard.vue
│   │   │   └── SummaryCard.vue
│   │   ├── symptoms/
│   │   │   ├── SymptomsForm.vue
│   │   │   ├── MealsForm.vue
│   │   │   └── ExerciseForm.vue
│   │   └── common/
│   │       ├── Navbar.vue
│   │       └── Modal.vue
│   ├── views/
│   │   ├── Dashboard.vue
│   │   ├── DailyLog.vue
│   │   ├── History.vue
│   │   ├── Patterns.vue
│   │   └── Profile.vue
│   ├── services/
│   │   ├── api.js              # Axios config
│   │   ├── auth.js             # Auth service
│   │   └── symptoms.js         # Symptoms service
│   ├── store/                  # Vuex/Pinia store
│   ├── router/                 # Vue Router
│   └── App.vue
└── package.json
```

---

## 📊 Visualización de Datos

### Librerías recomendadas para gráficas:

**Frontend:**
- **Chart.js** - Gráficas simples y bonitas
- **ApexCharts** - Gráficas interactivas avanzadas
- **D3.js** - Control total (más complejo)

**Backend (para PDFs):**
- **matplotlib** - Gráficas estáticas
- **ReportLab** - Generación de PDFs

---

## 🔐 Seguridad

### Autenticación JWT

```python
# backend/api/auth.py
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta

router = APIRouter()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: int = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        # Obtener user de BD
        return user_id
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
```

---

## 🚀 Deployment

### Arquitectura en Producción

```
[Internet]
    │
    ▼
[Nginx]  ← Reverse proxy
    │
    ├──► [Frontend] (Vue SPA)  :80
    │
    ├──► [Backend Web] (FastAPI)  :8000
    │         │
    │         ├──► [MySQL]  :3306
    │         │
    │         └──► [ML API] (FastAPI)  :8001
    │                   │
    │                   └──► [Modelos ML]
    │
    └──► [Static files]
```

### Docker Compose Example

```yaml
version: '3.8'

services:
  mysql:
    image: mysql:8.0
    environment:
      MYSQL_ROOT_PASSWORD: rootpass
      MYSQL_DATABASE: crohn_app
    volumes:
      - mysql_data:/var/lib/mysql
    ports:
      - "3306:3306"

  backend:
    build: ./backend
    ports:
      - "8000:8000"
    depends_on:
      - mysql
      - ml-api
    environment:
      DATABASE_URL: mysql://root:rootpass@mysql:3306/crohn_app
      ML_API_URL: http://ml-api:8001

  ml-api:
    build: ../crohn-flare-predictor
    ports:
      - "8001:8001"
    volumes:
      - ../crohn-flare-predictor/models:/app/models:ro

  frontend:
    build: ./frontend
    ports:
      - "80:80"
    depends_on:
      - backend

volumes:
  mysql_data:
```

---

## 📝 Notas para el Desarrollo

### Prioridades para MVP (Minimum Viable Product)

1. ✅ **Esencial (Fase 1):**
   - Autenticación (login/registro)
   - Registro de síntomas diarios
   - Dashboard con gráfica mensual
   - Integración básica con ML API (/predict)

2. 🔜 **Importante (Fase 2):**
   - Registro de comidas
   - Registro de ejercicio
   - Historial completo
   - Perfil médico editable

3. 💡 **Nice to have (Fase 3):**
   - Análisis de patrones
   - Exportar PDF para médico
   - Análisis de tendencias (ML API /analyze/trends)
   - Notificaciones push

### Tips de Desarrollo

1. **Empezar con datos dummy**: Crear fixtures para testear UI sin ML API
2. **Manejar fallos del ML API gracefully**: App debe funcionar sin predicciones
3. **Validación**: Tanto frontend como backend deben validar inputs
4. **Responsive design**: Diseñar mobile-first (muchos usuarios en móvil)
5. **Accesibilidad**: Usar ARIA labels, contrast ratios correctos
6. **Testing**: Tests unitarios (pytest) + tests E2E (Playwright/Cypress)

---

## 📚 Recursos

- **FastAPI**: https://fastapi.tiangolo.com/
- **Vue.js 3**: https://vuejs.org/
- **SQLAlchemy**: https://www.sqlalchemy.org/
- **Chart.js**: https://www.chartjs.org/
- **Pydantic**: https://docs.pydantic.dev/
- **ML API Documentation**: Ver `API_REFERENCE.md`

---

## 💬 Contacto

Para dudas sobre el ML API o integración, contactar a Asier (este repositorio).

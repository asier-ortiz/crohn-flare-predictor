# 🗄️ Esquema de Base de Datos para Backend Web

Este documento describe el esquema de base de datos recomendado para la aplicación web que consume el servicio ML.

## 📊 Tecnología Recomendada

**PostgreSQL 15+** por:
- ✅ Soporte JSON/JSONB (ideal para almacenar factores de predicción)
- ✅ Excelente para aplicaciones web
- ✅ Compatible con SQLAlchemy
- ✅ Rendimiento y escalabilidad

## 🗂️ Esquema Completo

### Diagrama Entidad-Relación

```
┌─────────────┐
│   users     │
└──────┬──────┘
       │
       │ 1:N
       ▼
┌──────────────────┐
│ daily_symptoms   │
└──────┬───────────┘
       │
       │ 1:1
       ▼
┌──────────────────┐      ┌─────────────────┐
│   predictions    │      │ trend_analyses  │
└──────────────────┘      └─────────────────┘
       │                          │
       │ 1:N                      │ 1:N
       ▼                          ▼
┌──────────────────┐      ┌─────────────────┐
│  notifications   │      │  notifications  │
└──────────────────┘      └─────────────────┘
```

---

## 📋 Tablas Principales

### 1. `users` - Información de Usuarios

```sql
CREATE TABLE users (
    -- Identificación
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,

    -- Información personal
    first_name VARCHAR(100) NOT NULL,
    last_name VARCHAR(100) NOT NULL,
    date_of_birth DATE NOT NULL,
    gender VARCHAR(1) CHECK (gender IN ('M', 'F', 'O')) NOT NULL,

    -- Información médica (Demographics para ML API)
    disease_duration_years INTEGER DEFAULT 0,
    diagnosis_date DATE,
    bmi DECIMAL(4, 1),

    -- Historial médico (History para ML API)
    previous_flares INTEGER DEFAULT 0,
    last_flare_date DATE,
    surgery_history BOOLEAN DEFAULT FALSE,
    surgery_date DATE,
    smoking_status VARCHAR(10) DEFAULT 'never'
        CHECK (smoking_status IN ('never', 'former', 'current')),

    -- Medicación actual (Array para ML API)
    medications TEXT[] DEFAULT '{}',

    -- Estado de cuenta
    is_active BOOLEAN DEFAULT TRUE,
    is_verified BOOLEAN DEFAULT FALSE,
    role VARCHAR(20) DEFAULT 'patient'
        CHECK (role IN ('patient', 'doctor', 'admin')),

    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP,

    -- Configuración
    timezone VARCHAR(50) DEFAULT 'UTC',
    language VARCHAR(5) DEFAULT 'es'
);

-- Índices
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_created_at ON users(created_at);
CREATE INDEX idx_users_is_active ON users(is_active);

-- Trigger para updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
```

### 2. `daily_symptoms` - Registros Diarios de Síntomas

```sql
CREATE TABLE daily_symptoms (
    -- Identificación
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    symptom_date DATE NOT NULL,

    -- Síntomas (formato exacto para ML API)
    abdominal_pain INTEGER NOT NULL CHECK (abdominal_pain BETWEEN 0 AND 10),
    diarrhea INTEGER NOT NULL CHECK (diarrhea BETWEEN 0 AND 10),
    fatigue INTEGER NOT NULL CHECK (fatigue BETWEEN 0 AND 10),
    fever BOOLEAN NOT NULL DEFAULT FALSE,
    weight_change DECIMAL(4, 1) NOT NULL DEFAULT 0.0,
    blood_in_stool BOOLEAN NOT NULL DEFAULT FALSE,
    nausea INTEGER DEFAULT 0 CHECK (nausea BETWEEN 0 AND 10),

    -- Información adicional (no usada por ML API)
    notes TEXT,
    bowel_movements INTEGER,
    sleep_hours DECIMAL(3, 1),
    stress_level INTEGER CHECK (stress_level BETWEEN 0 AND 10),

    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Constraints
    UNIQUE(user_id, symptom_date)
);

-- Índices importantes
CREATE INDEX idx_daily_symptoms_user_date ON daily_symptoms(user_id, symptom_date DESC);
CREATE INDEX idx_daily_symptoms_date ON daily_symptoms(symptom_date DESC);
CREATE INDEX idx_daily_symptoms_user_id ON daily_symptoms(user_id);

-- Trigger para updated_at
CREATE TRIGGER update_daily_symptoms_updated_at
    BEFORE UPDATE ON daily_symptoms
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
```

### 3. `predictions` - Cache de Predicciones ML

```sql
CREATE TABLE predictions (
    -- Identificación
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    symptom_record_id INTEGER REFERENCES daily_symptoms(id) ON DELETE SET NULL,

    -- Resultado de predicción (desde ML API)
    flare_risk VARCHAR(10) NOT NULL CHECK (flare_risk IN ('low', 'medium', 'high')),
    probability DECIMAL(4, 3) NOT NULL CHECK (probability BETWEEN 0 AND 1),
    confidence DECIMAL(4, 3) NOT NULL CHECK (confidence BETWEEN 0 AND 1),

    -- Factores contribuyentes (JSONB para flexibilidad)
    top_contributors TEXT[] DEFAULT '{}',
    symptom_severity_score DECIMAL(4, 3),
    trend_indicator VARCHAR(20),

    -- Recomendación
    recommendation TEXT,

    -- Metadata
    prediction_date DATE NOT NULL DEFAULT CURRENT_DATE,
    ml_model_version VARCHAR(20) DEFAULT '1.0.0',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Unique: una predicción por día por usuario
    UNIQUE(user_id, prediction_date)
);

-- Índices
CREATE INDEX idx_predictions_user_date ON predictions(user_id, prediction_date DESC);
CREATE INDEX idx_predictions_risk ON predictions(flare_risk);
CREATE INDEX idx_predictions_user_risk ON predictions(user_id, flare_risk);
```

### 4. `trend_analyses` - Análisis de Tendencias

```sql
CREATE TABLE trend_analyses (
    -- Identificación
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,

    -- Periodo analizado
    period_start DATE NOT NULL,
    period_end DATE NOT NULL,
    days_analyzed INTEGER NOT NULL,

    -- Resultados del análisis (desde ML API)
    overall_trend VARCHAR(20) NOT NULL
        CHECK (overall_trend IN ('improving', 'stable', 'worsening')),
    severity_change DECIMAL(5, 3) NOT NULL,
    concerning_patterns TEXT[] DEFAULT '{}',

    -- Evaluación de riesgo actual
    current_flare_risk VARCHAR(10) CHECK (current_flare_risk IN ('low', 'medium', 'high')),
    current_probability DECIMAL(4, 3),
    current_confidence DECIMAL(4, 3),

    -- Recomendaciones
    recommendations TEXT[] DEFAULT '{}',

    -- Metadata
    analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ml_model_version VARCHAR(20) DEFAULT '1.0.0',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Índices
CREATE INDEX idx_trend_analyses_user_date ON trend_analyses(user_id, analysis_date DESC);
CREATE INDEX idx_trend_analyses_trend ON trend_analyses(overall_trend);
```

### 5. `notifications` - Sistema de Notificaciones

```sql
CREATE TABLE notifications (
    -- Identificación
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,

    -- Tipo y severidad
    notification_type VARCHAR(50) NOT NULL,
    -- Tipos: 'high_risk_detected', 'trend_worsening', 'blood_detected',
    --        'streak_milestone', 'reminder', 'system'

    severity VARCHAR(20) NOT NULL DEFAULT 'info'
        CHECK (severity IN ('info', 'warning', 'urgent')),

    -- Contenido
    title VARCHAR(200) NOT NULL,
    message TEXT NOT NULL,

    -- Referencias (opcional)
    prediction_id INTEGER REFERENCES predictions(id) ON DELETE SET NULL,
    trend_analysis_id INTEGER REFERENCES trend_analyses(id) ON DELETE SET NULL,

    -- Estado
    is_read BOOLEAN DEFAULT FALSE,
    read_at TIMESTAMP,
    is_sent BOOLEAN DEFAULT FALSE,
    sent_at TIMESTAMP,

    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP
);

-- Índices
CREATE INDEX idx_notifications_user_unread ON notifications(user_id, is_read, created_at DESC);
CREATE INDEX idx_notifications_type ON notifications(notification_type);
CREATE INDEX idx_notifications_severity ON notifications(severity);
```

### 6. `user_preferences` - Preferencias del Usuario

```sql
CREATE TABLE user_preferences (
    id SERIAL PRIMARY KEY,
    user_id INTEGER UNIQUE NOT NULL REFERENCES users(id) ON DELETE CASCADE,

    -- Notificaciones
    email_notifications BOOLEAN DEFAULT TRUE,
    push_notifications BOOLEAN DEFAULT TRUE,
    daily_reminder BOOLEAN DEFAULT TRUE,
    reminder_time TIME DEFAULT '20:00:00',

    -- Alertas
    alert_on_high_risk BOOLEAN DEFAULT TRUE,
    alert_on_trend_worsening BOOLEAN DEFAULT TRUE,
    alert_on_blood_detected BOOLEAN DEFAULT TRUE,

    -- Privacidad
    share_data_for_research BOOLEAN DEFAULT FALSE,

    -- Configuración de análisis
    trend_analysis_frequency VARCHAR(20) DEFAULT 'weekly'
        CHECK (trend_analysis_frequency IN ('daily', 'weekly', 'biweekly', 'monthly')),

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TRIGGER update_user_preferences_updated_at
    BEFORE UPDATE ON user_preferences
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
```

---

## 🔗 Integración con ML API

### Función Helper: Preparar Datos para ML API

```sql
-- Vista que combina datos del usuario para llamar a ML API
CREATE OR REPLACE VIEW v_user_ml_data AS
SELECT
    u.id as user_id,
    -- Demographics
    EXTRACT(YEAR FROM AGE(u.date_of_birth)) as age,
    u.gender,
    u.disease_duration_years,
    u.bmi,
    -- History
    u.previous_flares,
    CASE
        WHEN u.last_flare_date IS NOT NULL
        THEN CURRENT_DATE - u.last_flare_date
        ELSE 9999
    END as last_flare_days_ago,
    u.medications,
    u.surgery_history,
    u.smoking_status
FROM users u
WHERE u.is_active = TRUE;
```

### Queries Comunes

#### 1. Obtener datos para predicción individual

```sql
-- Obtener último registro de síntomas + datos del usuario
SELECT
    ds.*,
    vml.age,
    vml.gender,
    vml.disease_duration_years,
    vml.bmi,
    vml.previous_flares,
    vml.last_flare_days_ago,
    vml.medications,
    vml.surgery_history,
    vml.smoking_status
FROM daily_symptoms ds
JOIN v_user_ml_data vml ON vml.user_id = ds.user_id
WHERE ds.user_id = $1
  AND ds.symptom_date = $2;
```

#### 2. Obtener datos para análisis de tendencias

```sql
-- Últimos 14 días de síntomas para un usuario
SELECT
    symptom_date as date,
    abdominal_pain,
    diarrhea,
    fatigue,
    fever,
    weight_change,
    blood_in_stool,
    nausea
FROM daily_symptoms
WHERE user_id = $1
  AND symptom_date >= CURRENT_DATE - INTERVAL '14 days'
ORDER BY symptom_date ASC;
```

#### 3. Verificar si necesita análisis de tendencias

```sql
-- Usuarios que necesitan análisis de tendencias
SELECT DISTINCT ds.user_id
FROM daily_symptoms ds
LEFT JOIN trend_analyses ta ON
    ta.user_id = ds.user_id
    AND ta.analysis_date >= CURRENT_DATE - INTERVAL '1 day'
WHERE ta.id IS NULL
  AND (
    SELECT COUNT(*)
    FROM daily_symptoms ds2
    WHERE ds2.user_id = ds.user_id
      AND ds2.symptom_date >= CURRENT_DATE - INTERVAL '14 days'
  ) >= 7;
```

---

## 📊 Ejemplo con SQLAlchemy (Python)

### Modelos SQLAlchemy

```python
# backend/db/models.py
from sqlalchemy import Column, Integer, String, Date, Boolean, DECIMAL, ARRAY, Text, ForeignKey, CheckConstraint
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)
    email = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)

    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    date_of_birth = Column(Date, nullable=False)
    gender = Column(String(1), nullable=False)

    disease_duration_years = Column(Integer, default=0)
    bmi = Column(DECIMAL(4, 1))
    previous_flares = Column(Integer, default=0)
    last_flare_date = Column(Date)
    surgery_history = Column(Boolean, default=False)
    smoking_status = Column(String(10), default='never')
    medications = Column(ARRAY(Text), default=[])

    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relaciones
    daily_symptoms = relationship("DailySymptom", back_populates="user", cascade="all, delete-orphan")
    predictions = relationship("Prediction", back_populates="user", cascade="all, delete-orphan")

    @property
    def age(self):
        from datetime import date
        today = date.today()
        return today.year - self.date_of_birth.year - (
            (today.month, today.day) < (self.date_of_birth.month, self.date_of_birth.day)
        )

    @property
    def last_flare_days_ago(self):
        from datetime import date
        if self.last_flare_date:
            return (date.today() - self.last_flare_date).days
        return 9999


class DailySymptom(Base):
    __tablename__ = 'daily_symptoms'

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    symptom_date = Column(Date, nullable=False)

    abdominal_pain = Column(Integer, nullable=False)
    diarrhea = Column(Integer, nullable=False)
    fatigue = Column(Integer, nullable=False)
    fever = Column(Boolean, nullable=False, default=False)
    weight_change = Column(DECIMAL(4, 1), nullable=False, default=0.0)
    blood_in_stool = Column(Boolean, nullable=False, default=False)
    nausea = Column(Integer, default=0)

    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relaciones
    user = relationship("User", back_populates="daily_symptoms")
    prediction = relationship("Prediction", back_populates="symptom_record", uselist=False)

    __table_args__ = (
        CheckConstraint('abdominal_pain BETWEEN 0 AND 10'),
        CheckConstraint('diarrhea BETWEEN 0 AND 10'),
        CheckConstraint('fatigue BETWEEN 0 AND 10'),
        CheckConstraint('nausea BETWEEN 0 AND 10'),
    )


class Prediction(Base):
    __tablename__ = 'predictions'

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    symptom_record_id = Column(Integer, ForeignKey('daily_symptoms.id', ondelete='SET NULL'))

    flare_risk = Column(String(10), nullable=False)
    probability = Column(DECIMAL(4, 3), nullable=False)
    confidence = Column(DECIMAL(4, 3), nullable=False)

    top_contributors = Column(ARRAY(Text), default=[])
    symptom_severity_score = Column(DECIMAL(4, 3))
    recommendation = Column(Text)

    prediction_date = Column(Date, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relaciones
    user = relationship("User", back_populates="predictions")
    symptom_record = relationship("DailySymptom", back_populates="prediction")
```

### Ejemplo de Uso

```python
# backend/api/symptoms.py
from datetime import date
from sqlalchemy.orm import Session
from db.models import User, DailySymptom, Prediction
from ml_client import ml_client

async def record_and_predict(
    user_id: int,
    symptoms_data: dict,
    db: Session
):
    """
    Registrar síntomas y obtener predicción del servicio ML.
    """

    # 1. Obtener usuario
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise ValueError("Usuario no encontrado")

    # 2. Guardar síntomas
    symptom_record = DailySymptom(
        user_id=user_id,
        symptom_date=date.today(),
        **symptoms_data  # abdominal_pain, diarrhea, etc.
    )
    db.add(symptom_record)
    db.flush()  # Para obtener el ID

    # 3. Preparar datos para ML API
    ml_request = {
        "symptoms": symptoms_data,
        "demographics": {
            "age": user.age,
            "gender": user.gender,
            "disease_duration_years": user.disease_duration_years,
            "bmi": float(user.bmi) if user.bmi else None
        },
        "history": {
            "previous_flares": user.previous_flares,
            "medications": user.medications,
            "last_flare_days_ago": user.last_flare_days_ago,
            "surgery_history": user.surgery_history,
            "smoking_status": user.smoking_status
        }
    }

    # 4. Llamar a ML API
    try:
        ml_response = await ml_client.predict_flare(**ml_request)

        # 5. Guardar predicción en cache
        prediction = Prediction(
            user_id=user_id,
            symptom_record_id=symptom_record.id,
            flare_risk=ml_response["prediction"]["flare_risk"],
            probability=ml_response["prediction"]["probability"],
            confidence=ml_response["prediction"]["confidence"],
            top_contributors=ml_response["factors"]["top_contributors"],
            symptom_severity_score=ml_response["factors"]["symptom_severity_score"],
            recommendation=ml_response["recommendation"],
            prediction_date=date.today()
        )
        db.add(prediction)

        # 6. Crear alerta si riesgo alto
        if ml_response["prediction"]["flare_risk"] == "high":
            create_notification(
                user_id=user_id,
                notification_type="high_risk_detected",
                severity="urgent",
                title="Riesgo Alto Detectado",
                message=ml_response["recommendation"],
                prediction_id=prediction.id,
                db=db
            )

    except Exception as e:
        # Si ML API falla, continuar sin predicción
        ml_response = None
        logger.warning(f"ML API error: {e}")

    db.commit()

    return {
        "symptom_record": symptom_record,
        "prediction": ml_response
    }
```

---

## 🔄 Migraciones con Alembic

### Setup inicial

```bash
# Instalar Alembic
uv add alembic

# Inicializar
alembic init migrations

# Configurar alembic.ini
# sqlalchemy.url = postgresql://user:pass@localhost/crohn_db
```

### Crear migración

```bash
# Auto-generar desde modelos
alembic revision --autogenerate -m "Initial schema"

# Aplicar
alembic upgrade head
```

---

## 📈 Optimizaciones y Mejores Prácticas

### 1. Índices Compuestos

```sql
-- Para queries frecuentes
CREATE INDEX idx_symptoms_user_date_desc
    ON daily_symptoms(user_id, symptom_date DESC);

CREATE INDEX idx_predictions_user_risk_date
    ON predictions(user_id, flare_risk, prediction_date DESC);
```

### 2. Particionamiento (para escala grande)

```sql
-- Particionar daily_symptoms por fecha
CREATE TABLE daily_symptoms_2024 PARTITION OF daily_symptoms
    FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');
```

### 3. Políticas de Retención

```sql
-- Eliminar notificaciones antiguas leídas
DELETE FROM notifications
WHERE is_read = TRUE
  AND read_at < CURRENT_DATE - INTERVAL '30 days';

-- Archivar síntomas antiguos (>2 años)
-- Mover a tabla de archivo en vez de eliminar
```

---

## 🔒 Seguridad

### 1. Row Level Security (RLS)

```sql
ALTER TABLE daily_symptoms ENABLE ROW LEVEL SECURITY;

CREATE POLICY user_own_symptoms ON daily_symptoms
    FOR ALL
    USING (user_id = current_setting('app.current_user_id')::INTEGER);
```

### 2. Encriptación de Datos Sensibles

```sql
-- Usar pgcrypto para campos sensibles
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Ejemplo (si hay datos médicos muy sensibles)
ALTER TABLE users ADD COLUMN medical_notes_encrypted BYTEA;
```

---

## 📝 Resumen para el Equipo

### Flujo de Datos

1. **Usuario registra síntomas** → Guardar en `daily_symptoms`
2. **Backend llama a ML API** → Obtener predicción
3. **Guardar predicción** en `predictions` (cache)
4. **Si riesgo alto** → Crear `notification`
5. **Cada X días** → Análisis de tendencias en `trend_analyses`

### Queries Principales

- `GET /symptoms/today` → Último registro de `daily_symptoms`
- `POST /symptoms/record` → INSERT en `daily_symptoms` + llamar ML API
- `GET /predictions/latest` → Último registro de `predictions`
- `GET /trends/week` → Últimos 7 días de `daily_symptoms`
- `GET /notifications/unread` → Notificaciones sin leer

### Consideraciones

- ✅ Una predicción por día por usuario (UNIQUE constraint)
- ✅ Guardar respuesta completa de ML API para histórico
- ✅ No depender 100% de ML API (puede fallar)
- ✅ Indices en campos usados en WHERE/JOIN
- ✅ Triggers para updated_at automático

# Esquema de Base de Datos - Crohn Flare Predictor

Este documento describe el esquema de base de datos necesario para la aplicación web de seguimiento de EII (Enfermedad Inflamatoria Intestinal).

## Tablas Principales

El esquema está diseñado para ser simple y fácil de implementar. Consta de 5 tablas principales:

1. **users** - Información de usuarios y autenticación
2. **user_profiles** - Datos demográficos y médicos del paciente
3. **daily_records** - Registros diarios de síntomas, alimentos y ejercicio
4. **predictions** - Historial de predicciones realizadas
5. **user_sessions** - Gestión de sesiones (opcional)

---

## SQL de Creación de Tablas

### 1. Tabla de Usuarios

```sql
CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    email VARCHAR(255) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP NULL,
    is_active BOOLEAN DEFAULT TRUE,
    INDEX idx_email (email)
);
```

**Descripción**: Almacena la información básica de autenticación de usuarios.

---

### 2. Tabla de Perfiles de Usuario

```sql
CREATE TABLE user_profiles (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    age INT NOT NULL,
    gender ENUM('M', 'F', 'other') NOT NULL,
    bmi DECIMAL(4,1),
    disease_duration_years DECIMAL(4,1),
    ibd_type ENUM('crohn', 'uc') NOT NULL,
    montreal_location VARCHAR(10),
    previous_flares INT DEFAULT 0,
    last_flare_days_ago INT,
    surgery_history BOOLEAN DEFAULT FALSE,
    smoking_status ENUM('never', 'former', 'current') DEFAULT 'never',
    cumulative_flare_days INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    INDEX idx_user_id (user_id)
);
```

**Descripción**: Contiene los datos demográficos y médicos de cada paciente. Estos datos se usan para las predicciones.

**Campos importantes**:
- `ibd_type`: "crohn" o "uc" (colitis ulcerosa)
- `montreal_location`: Clasificación de Montreal para Crohn (L1, L2, L3, L4)
- `smoking_status`: Nunca fumó, ex-fumador, fumador actual

---

### 3. Tabla de Medicamentos del Usuario

```sql
CREATE TABLE user_medications (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    medication_name VARCHAR(255) NOT NULL,
    start_date DATE,
    end_date DATE NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    INDEX idx_user_id (user_id),
    INDEX idx_active (user_id, is_active)
);
```

**Descripción**: Lista de medicamentos que el usuario está tomando actualmente o ha tomado.

**Nota**: Los nombres pueden estar en español o inglés (ej: "mesalazina", "Humira", "azatioprina")

---

### 4. Tabla de Registros Diarios

```sql
CREATE TABLE daily_records (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    record_date DATE NOT NULL,

    -- Síntomas (escala 0-10 o boolean)
    abdominal_pain INT CHECK (abdominal_pain BETWEEN 0 AND 10),
    diarrhea INT CHECK (diarrhea BETWEEN 0 AND 10),
    fatigue INT CHECK (fatigue BETWEEN 0 AND 10),
    nausea INT CHECK (nausea BETWEEN 0 AND 10),
    fever BOOLEAN DEFAULT FALSE,
    blood_in_stool BOOLEAN DEFAULT FALSE,
    weight_change DECIMAL(4,1) DEFAULT 0.0,

    -- Estilo de vida
    exercise ENUM('none', 'moderate', 'high') DEFAULT 'none',

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    UNIQUE KEY unique_user_date (user_id, record_date),
    INDEX idx_user_date (user_id, record_date)
);
```

**Descripción**: Registros diarios de síntomas y ejercicio.

**Campos importantes**:
- Síntomas numéricos: escala de 0-10 (0 = sin síntoma, 10 = máximo)
- `exercise`: none (sin ejercicio), moderate (moderado), high (intenso)
- `weight_change`: cambio de peso en kg (negativo = pérdida)

---

### 5. Tabla de Alimentos Consumidos

```sql
CREATE TABLE daily_foods (
    id INT AUTO_INCREMENT PRIMARY KEY,
    daily_record_id INT NOT NULL,
    food_text VARCHAR(500) NOT NULL,
    meal_type ENUM('breakfast', 'lunch', 'dinner', 'snack') NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (daily_record_id) REFERENCES daily_records(id) ON DELETE CASCADE,
    INDEX idx_daily_record (daily_record_id)
);
```

**Descripción**: Lista de alimentos consumidos cada día (texto libre).

**Ejemplo de datos**:
- "pizza con queso"
- "café con leche"
- "ensalada verde"

La API categoriza automáticamente estos textos en grupos (lácteos, gluten, café, etc.)

---

### 6. Tabla de Predicciones

```sql
CREATE TABLE predictions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Resultado de la predicción
    risk_level ENUM('low', 'medium', 'high') NOT NULL,
    probability DECIMAL(4,2) NOT NULL,
    confidence DECIMAL(4,2) NOT NULL,

    -- Información adicional
    model_version VARCHAR(50),
    ibd_type VARCHAR(20),
    cluster_id INT NULL,

    -- Tendencias (si hay suficiente historial)
    trend_direction ENUM('worsening', 'improving', 'stable') NULL,
    severity_change DECIMAL(4,2) NULL,

    -- Datos completos de la respuesta (JSON)
    full_response JSON,

    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    INDEX idx_user_date (user_id, prediction_date),
    INDEX idx_risk_level (risk_level)
);
```

**Descripción**: Historial de todas las predicciones realizadas para cada usuario.

**Campos importantes**:
- `risk_level`: bajo, moderado, alto
- `probability`: probabilidad decimal (0.00-1.00)
- `full_response`: JSON completo de la respuesta de la API (para consultas detalladas)

---

## Creación de la Base de Datos Completa

```sql
-- Crear la base de datos
CREATE DATABASE IF NOT EXISTS crohn_tracker CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

USE crohn_tracker;

-- Crear todas las tablas en orden (debido a foreign keys)

-- 1. Usuarios (primero porque otras tablas dependen de ella)
CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    email VARCHAR(255) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP NULL,
    is_active BOOLEAN DEFAULT TRUE,
    INDEX idx_email (email)
);

-- 2. Perfiles de usuario
CREATE TABLE user_profiles (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    age INT NOT NULL,
    gender ENUM('M', 'F', 'other') NOT NULL,
    bmi DECIMAL(4,1),
    disease_duration_years DECIMAL(4,1),
    ibd_type ENUM('crohn', 'uc') NOT NULL,
    montreal_location VARCHAR(10),
    previous_flares INT DEFAULT 0,
    last_flare_days_ago INT,
    surgery_history BOOLEAN DEFAULT FALSE,
    smoking_status ENUM('never', 'former', 'current') DEFAULT 'never',
    cumulative_flare_days INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    INDEX idx_user_id (user_id)
);

-- 3. Medicamentos
CREATE TABLE user_medications (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    medication_name VARCHAR(255) NOT NULL,
    start_date DATE,
    end_date DATE NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    INDEX idx_user_id (user_id),
    INDEX idx_active (user_id, is_active)
);

-- 4. Registros diarios
CREATE TABLE daily_records (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    record_date DATE NOT NULL,
    abdominal_pain INT CHECK (abdominal_pain BETWEEN 0 AND 10),
    diarrhea INT CHECK (diarrhea BETWEEN 0 AND 10),
    fatigue INT CHECK (fatigue BETWEEN 0 AND 10),
    nausea INT CHECK (nausea BETWEEN 0 AND 10),
    fever BOOLEAN DEFAULT FALSE,
    blood_in_stool BOOLEAN DEFAULT FALSE,
    weight_change DECIMAL(4,1) DEFAULT 0.0,
    exercise ENUM('none', 'moderate', 'high') DEFAULT 'none',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    UNIQUE KEY unique_user_date (user_id, record_date),
    INDEX idx_user_date (user_id, record_date)
);

-- 5. Alimentos consumidos
CREATE TABLE daily_foods (
    id INT AUTO_INCREMENT PRIMARY KEY,
    daily_record_id INT NOT NULL,
    food_text VARCHAR(500) NOT NULL,
    meal_type ENUM('breakfast', 'lunch', 'dinner', 'snack') NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (daily_record_id) REFERENCES daily_records(id) ON DELETE CASCADE,
    INDEX idx_daily_record (daily_record_id)
);

-- 6. Predicciones
CREATE TABLE predictions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    risk_level ENUM('low', 'medium', 'high') NOT NULL,
    probability DECIMAL(4,2) NOT NULL,
    confidence DECIMAL(4,2) NOT NULL,
    model_version VARCHAR(50),
    ibd_type VARCHAR(20),
    cluster_id INT NULL,
    trend_direction ENUM('worsening', 'improving', 'stable') NULL,
    severity_change DECIMAL(4,2) NULL,
    full_response JSON,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    INDEX idx_user_date (user_id, prediction_date),
    INDEX idx_risk_level (risk_level)
);
```

---

## Notas de Implementación

### Seguridad
- **NUNCA** guardes contraseñas en texto plano. Usa `password_hash` con bcrypt o similar.
- Implementa validación de email antes de insertar en la tabla `users`.

### Consultas Comunes

#### Obtener últimos 7 días de registros para un usuario
```sql
SELECT * FROM daily_records
WHERE user_id = ?
  AND record_date >= DATE_SUB(CURDATE(), INTERVAL 7 DAY)
ORDER BY record_date DESC;
```

#### Obtener alimentos de un día específico
```sql
SELECT df.food_text, df.meal_type
FROM daily_foods df
INNER JOIN daily_records dr ON df.daily_record_id = dr.id
WHERE dr.user_id = ? AND dr.record_date = ?;
```

#### Obtener medicamentos activos de un usuario
```sql
SELECT medication_name
FROM user_medications
WHERE user_id = ? AND is_active = TRUE;
```

#### Obtener historial de predicciones (últimas 30)
```sql
SELECT risk_level, probability, prediction_date, trend_direction
FROM predictions
WHERE user_id = ?
ORDER BY prediction_date DESC
LIMIT 30;
```

---

## Flujo de Datos Típico

1. **Usuario registra síntomas diarios**
   - INSERT en `daily_records`
   - INSERT en `daily_foods` (múltiples filas si comió varias cosas)

2. **Usuario solicita predicción**
   - SELECT últimos 7-14 días de `daily_records`
   - SELECT alimentos asociados de `daily_foods`
   - SELECT medicamentos activos de `user_medications`
   - SELECT datos demográficos de `user_profiles`
   - Enviar todo a la API `/predict?format=simple`
   - INSERT resultado en `predictions`

3. **Dashboard muestra datos**
   - SELECT últimas 30 predicciones para gráfica
   - SELECT registro de hoy para "estado actual"
   - SELECT insights de lifestyle de última predicción

---

## Consideraciones de Performance

- **Índices**: Ya incluidos en las definiciones (idx_user_date, idx_email, etc.)
- **Particionamiento**: Si la aplicación crece mucho, considera particionar `daily_records` y `predictions` por año
- **Archivado**: Después de 2-3 años, considera mover registros antiguos a tabla de archivo

---

## Ejemplo de Datos de Prueba

```sql
-- Usuario de prueba
INSERT INTO users (email, password_hash, full_name)
VALUES ('test@example.com', '$2b$12$...hash...', 'Usuario Test');

-- Perfil del usuario
INSERT INTO user_profiles (user_id, age, gender, bmi, ibd_type, disease_duration_years, montreal_location, previous_flares, last_flare_days_ago, surgery_history, smoking_status, cumulative_flare_days)
VALUES (1, 32, 'F', 22.5, 'crohn', 5.0, 'L3', 3, 120, FALSE, 'never', 45);

-- Medicamentos
INSERT INTO user_medications (user_id, medication_name, is_active)
VALUES
    (1, 'mesalazina', TRUE),
    (1, 'azatioprina', TRUE);

-- Registro diario
INSERT INTO daily_records (user_id, record_date, abdominal_pain, diarrhea, fatigue, nausea, fever, blood_in_stool, weight_change, exercise)
VALUES (1, '2025-11-22', 4, 3, 3, 2, FALSE, FALSE, -0.2, 'moderate');

-- Alimentos del día
INSERT INTO daily_foods (daily_record_id, food_text, meal_type)
VALUES
    (1, 'café con leche', 'breakfast'),
    (1, 'tostadas', 'breakfast'),
    (1, 'ensalada cesar', 'lunch'),
    (1, 'pollo a la plancha', 'dinner');
```

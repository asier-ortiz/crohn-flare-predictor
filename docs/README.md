# 📚 Documentación del Proyecto

Documentación completa del servicio ML para predicción de brotes de IBD (Enfermedad Inflamatoria Intestinal).

## 📖 Guías por Rol

### 👨‍💻 Para Desarrolladores Web (Consumidores de la API)

Si necesitas integrar este servicio ML en tu aplicación:

1. **[Guía de Integración](INTEGRATION.md)** ⭐ Empieza aquí
   - Cómo consumir la API desde tu aplicación web
   - Ejemplos de código
   - Mejores prácticas

2. **[Referencia de API](API_REFERENCE.md)**
   - Documentación completa de endpoints
   - Schemas de request/response
   - Códigos de error

3. **[Esquema de Base de Datos](DATABASE_SCHEMA.md)** (Referencia)
   - Schema recomendado para tu backend
   - Cómo almacenar predicciones y síntomas

### 🔬 Para Desarrollo ML

Si vas a trabajar en el modelo o entrenar nuevos modelos:

1. **[Guía de Desarrollo](DEVELOPMENT.md)** ⭐ Empieza aquí
   - Setup del entorno local
   - Flujo de trabajo con notebooks
   - Re-entrenamiento de modelos

2. **[Implementación Cluster-Stratified](CLUSTER_STRATIFIED_IMPLEMENTATION.md)**
   - Arquitectura de modelos cluster-stratified
   - Mapeo Montreal Classification → Clusters
   - Features derivadas (34 features totales)

3. **[Arquitectura](architecture.md)**
   - Decisiones de diseño
   - ¿Por qué un servicio independiente?
   - Stateless vs Stateful

### 🚀 Para DevOps/Despliegue

1. **[Deployment](deployment.md)**
   - Cómo desplegar en producción
   - Docker y configuración
   - Variables de entorno

## 🎯 ¿Qué es este proyecto?

Este es un **servicio ML independiente** (microservicio) que expone una API REST para predicción de brotes de IBD basado en:
- Síntomas diarios del paciente
- Historial médico
- Features derivadas (agregaciones, temporales, interacciones)
- Modelos cluster-stratified por fenotipo de enfermedad

### ✅ Responsabilidades del Servicio

- Entrenar y mantener modelos ML
- Exponer predicciones vía API REST
- Clasificación automática por clusters (Montreal)
- Análisis de tendencias temporales
- Predicciones individuales y por lotes

### ❌ NO es Responsabilidad del Servicio

- Gestión de usuarios (login, registro)
- Almacenamiento de datos de pacientes
- Frontend/UI
- Base de datos persistente

## 📂 Estructura del Proyecto

```
crohn-flare-predictor/
├── api/                    # API FastAPI
│   ├── app.py             # Aplicación principal
│   ├── ml_model.py        # Lógica de predicción cluster-stratified
│   └── schemas.py         # Validación Pydantic
├── notebooks/             # Análisis y entrenamiento
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_advanced_feature_engineering.ipynb
│   ├── 04_cluster_stratified_training.ipynb
│   └── 05_cluster_stratified_training_cu.ipynb
├── models/                # Modelos entrenados (.pkl)
│   ├── crohn/
│   └── cu/
├── scripts/               # Scripts de utilidad
│   ├── test_api.py
│   └── evaluate_model.py
├── docs/                  # Esta documentación
└── tests/                 # Tests unitarios
```

## 🚀 Quick Start

### Levantar el Servicio

```bash
# 1. Instalar dependencias
uv sync

# 2. Iniciar API
make serve

# 3. Verificar que funciona
curl http://localhost:8001/health
```

### Explorar la API

Una vez corriendo, accede a:
- **Swagger UI**: http://localhost:8001/docs
- **ReDoc**: http://localhost:8001/redoc

## 🔗 Links Útiles

- **Swagger Docs**: http://localhost:8001/docs (cuando el servidor esté corriendo)
- **Kaggle Dataset**: [Flaredown Autoimmune Symptom Tracker](https://www.kaggle.com/datasets/flaredown/flaredown-autoimmune-symptom-tracker)

## 📊 Características del Modelo

### Modelos Cluster-Stratified

- **34 features totales**: 13 base + 21 derivadas
- **Modelos separados** por tipo de IBD (Crohn / UC)
- **3 clusters** por fenotipo de enfermedad (basado en Montreal Classification)
- **Global fallback** cuando cluster-specific no está disponible
- **99.22% accuracy** en Crohn, **100% recall** para riesgo alto

### Features Derivadas (21)

1. **Agregaciones de Síntomas** (5): total_symptom_score, gi_score, systemic_score, red_flag_score, symptom_count
2. **Temporales** (7): pain_trend_7d, diarrhea_trend_7d, fatigue_trend_7d, volatility, change_rate, days_since_low
3. **Historial** (4): flare_frequency, recency_score, disease_burden, young_longduration
4. **Interacciones** (5): pain_diarrhea_combo, blood_and_pain, vulnerable_state, severity_category, gi_dominant

## 📞 Soporte

Para problemas o preguntas sobre:
- **API ML**: Contacta al equipo de ML
- **Integración/Backend Web**: Consulta la guía de integración
- **Despliegue**: Ver deployment.md

## 📄 Licencia

MIT License - Ver archivo LICENSE en la raíz del proyecto.

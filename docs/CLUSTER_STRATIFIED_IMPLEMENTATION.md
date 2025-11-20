# 🎯 Cluster-Stratified Model Implementation

## Resumen Ejecutivo

Este documento describe la implementación de **modelos estratificados por fenotipo de paciente** para mejorar las predicciones de brotes de Crohn.

### Motivación

La enfermedad de Crohn tiene diferentes localizaciones y presentaciones clínicas (fenotipos):
- **Patrón Ileal (L1)**: Predomina dolor abdominal
- **Patrón Colónico (L2)**: Predomina diarrea con sangre
- **Patrón Ileocolónico (L3)**: Sintomatología mixta

**Hipótesis**: Entrenar modelos específicos para cada fenotipo mejorará las predicciones al capturar patrones únicos de cada subgrupo.

---

## 📋 Implementación

### 1. Clustering de Fenotipos (Notebook 01)

- **Algoritmo**: KMeans con k=3 clusters
- **Features de clustering**:
  - Síntomas: `abdominal_pain`, `blood_in_stool`, `diarrhea`, `fatigue`, `fever`, `nausea`
  - Ratios derivados: `pain_diarrhea_ratio`, `blood_freq`
- **Validación**: Silhouette Score ~0.27
- **Resultado**: 1,261 pacientes clustered
  - Cluster 0 (21.5%): Patrón Ileal-like
  - Cluster 1 (64.8%): Patrón Ileocolónico-like
  - Cluster 2 (13.7%): Patrón con alto sangrado

### 2. Modelos Estratificados (Notebook 04)

**Entrenamiento:**
- **3 modelos RF** (uno por cluster)
- **SMOTE Moderado** aplicado a cada cluster
- **Mismos hiperparámetros** que modelo global

**Archivos generados:**
```
models/
├── rf_severity_classifier_cluster_0.pkl   # Modelo Cluster 0
├── rf_severity_classifier_cluster_1.pkl   # Modelo Cluster 1
├── rf_severity_classifier_cluster_2.pkl   # Modelo Cluster 2
├── cluster_kmeans.pkl                     # KMeans para inferencia
├── cluster_scaler.pkl                     # StandardScaler
├── cluster_models_metadata.json           # Metadata de modelos
└── cluster_metadata.json                  # Metadata de clustering
```

### 3. API Actualizada

**Nueva clase**: `ClusterStratifiedPredictor` en `api/ml_model.py`

**Flujo de predicción:**
1. **Inferir cluster** del usuario basándose en síntomas
2. **Cargar modelo** específico del cluster
3. **Predecir** usando modelo especializado

**Respuesta API extendida:**
```json
{
  "prediction": {
    "flare_risk": "high",
    "probability": 0.82,
    "confidence": 0.75,
    "probabilities": {
      "low": 0.05,
      "medium": 0.13,
      "high": 0.82
    },
    "cluster_id": 1,              // ← NUEVO
    "cluster_confidence": 0.89    // ← NUEVO
  },
  "factors": {...},
  "recommendation": "..."
}
```

**Compatibilidad:**
- La API **automáticamente detecta** si existen modelos por cluster
- Si no existen, **fallback** al modelo global
- **Backward compatible**: clientes existentes funcionan sin cambios

---

## 🚀 Uso

### Entrenar Modelos por Cluster

```bash
# Descargar dataset de Kaggle a data/raw/export.csv
# Luego ejecutar:

make train-clusters
```

Esto ejecutará:
1. Notebook 01: Clustering de fenotipos
2. Notebook 02: Feature engineering
3. Notebook 04: Entrenamiento estratificado

### Iniciar API con Modelos por Cluster

```bash
make serve
```

La API automáticamente:
- ✅ Detecta modelos por cluster
- ✅ Los carga y usa para predicciones
- ✅ Incluye `cluster_id` y `cluster_confidence` en respuestas

### Usar Modelo Global (Fallback)

Si quieres forzar el uso del modelo global:

```bash
# Renombrar/mover los modelos por cluster
mv models/cluster_kmeans.pkl models/cluster_kmeans.pkl.bak

# Levantar API
make serve
```

La API detectará que no hay modelos por cluster y usará el global.

---

## 📊 Comparación de Métricas

**A completar después de entrenar con datos reales:**

| Métrica | Modelo Global | Modelos por Cluster | Mejora |
|---------|---------------|---------------------|--------|
| Accuracy | TBD | TBD | TBD |
| F1 Macro | TBD | TBD | TBD |
| F1 High | TBD | TBD | TBD |
| Recall High | TBD | TBD | TBD |

**Ubicación del reporte:** `reports/cluster_stratified_training_report.json`

---

## 🔍 Inferencia de Cluster

### Cómo Funciona

El sistema infiere automáticamente el cluster del paciente:

1. **Extrae features de síntomas** del request
2. **Normaliza** con StandardScaler entrenado
3. **Predice cluster** con KMeans
4. **Calcula confianza** basándose en distancias a centroides

### Confianza del Cluster

```python
confidence = (dist_segundo_cluster - dist_cluster_asignado) / dist_segundo_cluster
```

- **Alta confianza (>0.7)**: Paciente claramente pertenece al cluster
- **Media confianza (0.4-0.7)**: Cluster probable pero con solapamiento
- **Baja confianza (<0.4)**: Paciente en frontera entre clusters

### Ejemplo

**Input:**
```json
{
  "symptoms": {
    "abdominal_pain": 8,
    "diarrhea": 3,
    "blood_in_stool": false,
    "fatigue": 5,
    "fever": false,
    "nausea": 6
  },
  "demographics": {...},
  "history": {...}
}
```

**Inferencia:**
- **Cluster inferido**: 0 (Patrón Ileal - alto dolor, baja diarrea)
- **Confianza**: 0.85 (alta)
- **Modelo usado**: `rf_severity_classifier_cluster_0.pkl`

---

## 🧪 Testing

### Probar API con Cluster

```bash
# Levantar servidor
make serve

# En otra terminal, probar endpoint
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "symptoms": {
      "abdominal_pain": 8,
      "diarrhea": 3,
      "fatigue": 5,
      "fever": false,
      "blood_in_stool": false,
      "nausea": 4
    },
    "demographics": {
      "age": 35,
      "gender": "F",
      "disease_duration_years": 5
    },
    "history": {
      "previous_flares": 2,
      "last_flare_days_ago": 180
    }
  }'
```

**Respuesta esperada:**
```json
{
  "prediction": {
    "flare_risk": "medium",
    "probability": 0.72,
    "confidence": 0.58,
    "probabilities": {...},
    "cluster_id": 0,
    "cluster_confidence": 0.85
  },
  ...
}
```

---

## 📁 Estructura de Archivos

```
crohn-flare-predictor/
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb      # Clustering inicial
│   ├── 02_feature_engineering.ipynb       # Feature engineering
│   ├── 03_model_training.ipynb           # Modelo global
│   └── 04_cluster_stratified_training.ipynb  # ← NUEVO (modelos por cluster)
│
├── api/
│   ├── ml_model.py                       # ← ACTUALIZADO (ClusterStratifiedPredictor)
│   ├── app.py                            # ← ACTUALIZADO (soporte clusters)
│   └── schemas.py                        # ← ACTUALIZADO (cluster_id, cluster_confidence)
│
├── models/
│   ├── rf_severity_classifier.pkl         # Modelo global (fallback)
│   ├── rf_severity_classifier_cluster_0.pkl  # ← NUEVO
│   ├── rf_severity_classifier_cluster_1.pkl  # ← NUEVO
│   ├── rf_severity_classifier_cluster_2.pkl  # ← NUEVO
│   ├── cluster_kmeans.pkl                    # ← NUEVO
│   ├── cluster_scaler.pkl                    # ← NUEVO
│   ├── cluster_models_metadata.json          # ← NUEVO
│   └── cluster_metadata.json                 # ← NUEVO
│
├── data/processed/
│   ├── ml_dataset.csv                    # Dataset ML
│   ├── user_clusters.csv                 # Asignación de clusters
│   └── cluster_profiles.csv              # Perfiles de clusters
│
├── reports/
│   └── cluster_stratified_training_report.json  # ← NUEVO (comparación métricas)
│
└── docs/
    └── CLUSTER_STRATIFIED_IMPLEMENTATION.md     # ← Este documento
```

---

## 🔧 Troubleshooting

### API no usa modelos por cluster

**Síntomas:**
- `cluster_id` es `null` en respuestas
- Logs muestran: "Using global predictor"

**Solución:**
1. Verificar que existan los archivos:
   ```bash
   ls models/cluster_*.pkl models/cluster_kmeans.pkl
   ```
2. Revisar logs de la API al iniciar
3. Verificar permisos de lectura

### Error al cargar KMeans

**Error:**
```
Error loading cluster models: No module named 'sklearn.cluster'
```

**Solución:**
```bash
uv sync
```

### Cluster confidence siempre baja

**Causa**: Pacientes en fronteras entre clusters (normal)

**Solución**: No es un error. Indica que el paciente tiene características de múltiples fenotipos. El modelo aún predice correctamente.

---

## 📚 Referencias

- **Notebook 01**: `notebooks/01_exploratory_analysis.ipynb` - Clustering methodology
- **Notebook 04**: `notebooks/04_cluster_stratified_training.ipynb` - Training details
- **API Docs**: http://localhost:8001/docs (cuando el servidor está activo)

---

## 🎯 Próximos Pasos

1. **Entrenar con datos reales** y evaluar métricas
2. **Comparar** modelo global vs cluster-stratified
3. **Analizar** casos donde cluster-stratified funciona mejor
4. **Iterar** sobre número de clusters (¿k=4? ¿k=5?)
5. **Explorar** features adicionales para clustering (ej: Montreal classification si está disponible)

---

## 💡 Notas Importantes

- **Sin datos, sin modelos**: Necesitas dataset de Flaredown de Kaggle
- **Descargar dataset**: https://www.kaggle.com/datasets/amanik000/gastrointestinal-disease-dataset
- **Colocar en**: `data/raw/export.csv`
- **Ejecutar**: `make train-clusters`

**⚠️ Disclaimer**: Este proyecto es para investigación y educación. No usar para diagnóstico médico real.

---

**Autor**: Claude Assistant + Asier Ortiz García
**Fecha**: Noviembre 2025
**Versión**: 1.0

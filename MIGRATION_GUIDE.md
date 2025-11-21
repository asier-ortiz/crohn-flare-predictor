# Guía de Migración: Reorganización de Notebooks

Esta guía te ayudará a reorganizar los notebooks manualmente en tu máquina local.

## Resumen de Cambios

- **5 notebooks** → **3 notebooks**
- Notebook 01: Añadir análisis medications/surgery/smoking
- Notebook 02: Fusionar 02 + 03 (base + derivadas)
- Notebook 03: Unificar 04 + 05 (training Crohn + CU)

## Paso 1: Instalar dependencias

```bash
pip install imbalanced-learn
```

## Paso 2: Ejecutar scripts de migración

Los scripts están en este mismo directorio:
- `scripts/migration/update_notebook_01.py`
- `scripts/migration/merge_notebooks_02_03.py`
- `scripts/migration/create_notebook_03.py`

```bash
cd ~/Documentos/GitHub/crohn-flare-predictor

python scripts/migration/update_notebook_01.py
python scripts/migration/merge_notebooks_02_03.py
python scripts/migration/create_notebook_03.py
```

## Paso 3: Ejecutar notebooks

```bash
# Ejecutar en orden (puede tardar 20-30 minutos total)
jupyter nbconvert --to notebook --execute notebooks/01_exploratory_analysis.ipynb --output 01_exploratory_analysis.ipynb

jupyter nbconvert --to notebook --execute notebooks/02_feature_engineering.ipynb --output 02_feature_engineering.ipynb

jupyter nbconvert --to notebook --execute notebooks/03_model_training.ipynb --output 03_model_training.ipynb
```

## Paso 4: Eliminar notebooks viejos

```bash
rm notebooks/03_advanced_feature_engineering.ipynb
rm notebooks/04_cluster_stratified_training.ipynb
rm notebooks/05_cluster_stratified_training_cu.ipynb
```

## Paso 5: Commit y push

```bash
git add -A
git commit -m "refactor: reorganize notebooks and add Flaredown dataset analysis

- Reorganized 5 notebooks into 3
- Added medications/surgery/smoking analysis to notebook 01
- Merged notebooks 02 + 03 into unified feature engineering
- Unified notebooks 04 + 05 into single training notebook
- Retrained all models (Crohn + CU)"

git push origin main
```

## Verificación

Después de la migración deberías tener:

```
notebooks/
├── 01_exploratory_analysis.ipynb (26 celdas)
├── 02_feature_engineering.ipynb (31 celdas)
└── 03_model_training.ipynb (11 celdas)

models/
├── crohn/
│   ├── rf_severity_classifier_global.pkl
│   ├── rf_severity_classifier_cluster_0.pkl
│   ├── rf_severity_classifier_cluster_1.pkl
│   ├── rf_severity_classifier_cluster_2.pkl
│   ├── cluster_kmeans.pkl
│   └── cluster_scaler.pkl
└── cu/
    ├── rf_severity_classifier_global.pkl
    ├── rf_severity_classifier_cluster_0.pkl
    ├── rf_severity_classifier_cluster_1.pkl
    ├── cluster_kmeans.pkl
    └── cluster_scaler.pkl
```

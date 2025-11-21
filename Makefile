.PHONY: help install sync clean test format lint serve notebook docker-build docker-run
.PHONY: kill-serve run-notebook-02 run-notebook-03 pipeline reports show-latest-report

# Variables
PYTHON := uv run python
PYTEST := uv run pytest
BLACK := uv run black
FLAKE8 := uv run flake8
JUPYTER := uv run jupyter
UVICORN := uv run uvicorn

help: ## Mostrar este mensaje de ayuda
	@echo "Comandos disponibles:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Instalar uv (gestor de paquetes)
	@echo "Instalando uv..."
	@curl -LsSf https://astral.sh/uv/install.sh | sh || pip install uv

sync: ## Sincronizar e instalar todas las dependencias
	@echo "Instalando dependencias con uv..."
	uv sync
	@echo "✅ Dependencias instaladas correctamente"

clean: ## Limpiar archivos temporales y cache
	@echo "Limpiando archivos temporales..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".coverage" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Limpieza completada"

test: ## Ejecutar tests de integración (requiere servidor activo en :8001)
	@echo "🧪 Ejecutando tests de integración..."
	@echo "⚠️  Asegúrate de que el servidor esté corriendo: make serve"
	@echo ""
	$(PYTEST) --no-cov scripts/test_api.py
	@echo ""
	@echo "✅ Tests de integración completados"

test-unit: ## Ejecutar tests unitarios con cobertura
	@echo "🧪 Ejecutando tests unitarios con cobertura..."
	$(PYTEST) --cov=api --cov-report=html --cov-report=term
	@echo "📊 Reporte de cobertura generado en htmlcov/index.html"

test-integration: test ## Alias para 'test' (tests de integración)

format: ## Formatear código con Black
	@echo "Formateando código..."
	$(BLACK) api/ scripts/
	@echo "✅ Código formateado"

lint: ## Verificar código con flake8
	@echo "Verificando código..."
	$(FLAKE8) api/ scripts/
	@echo "✅ Verificación completada"

check: format lint ## Ejecutar formato y lint (sin tests)

check-all: format lint test ## Ejecutar formato, lint y tests (requiere servidor)

serve: ## Iniciar servidor API en modo desarrollo
	@echo "🚀 Iniciando servidor API..."
	@echo "Documentación disponible en:"
	@echo "  - Swagger UI: http://localhost:8001/docs"
	@echo "  - ReDoc: http://localhost:8001/redoc"
	$(UVICORN) api.app:app --reload --host 0.0.0.0 --port 8001

kill-serve: ## Detener servidor API (libera puerto 8001)
	@echo "🛑 Deteniendo servidor en puerto 8001..."
	@lsof -ti :8001 | xargs kill -9 2>/dev/null || echo "✅ Puerto 8001 ya está libre"

serve-prod: ## Iniciar servidor API en modo producción
	@echo "🚀 Iniciando servidor API (producción)..."
	$(UVICORN) api.app:app --host 0.0.0.0 --port 8001 --workers 4

test-api: ## Probar endpoints de la API (requiere servidor activo)
	@echo "🧪 Probando API..."
	$(PYTHON) scripts/test_api.py

test-api-curl: ## Probar API con curl (requiere jq instalado)
	@echo "🧪 Probando API con curl..."
	./scripts/test_api.sh

evaluate: ## Evaluar precisión del modelo (requiere servidor activo)
	@echo "🔬 Evaluando modelo..."
	$(PYTHON) scripts/evaluate_model.py

notebook: ## Iniciar Jupyter Notebook (con entorno virtual)
	@echo "📓 Iniciando Jupyter Notebook..."
	@echo "⚙️  Usando entorno virtual de uv"
	uv run jupyter notebook

lab: ## Iniciar Jupyter Lab (con entorno virtual)
	@echo "📓 Iniciando Jupyter Lab..."
	@echo "⚙️  Usando entorno virtual de uv"
	uv run jupyter lab

train: ## Ejecutar entrenamiento completo (notebooks 02 y 03)
	@echo "🤖 Ejecutando pipeline de entrenamiento..."
	@echo "📊 Paso 1/2: Feature Engineering (notebook 02)..."
	uv run jupyter nbconvert --to notebook --execute notebooks/02_feature_engineering.ipynb --inplace
	@echo "✅ Features generadas"
	@echo "📊 Paso 2/2: Model Training (notebook 03)..."
	uv run jupyter nbconvert --to notebook --execute notebooks/03_model_training.ipynb --inplace
	@echo "✅ Modelo entrenado y guardado en models/"

run-notebook-02: ## Ejecutar solo notebook 02 (Feature Engineering)
	@echo "📊 Ejecutando Feature Engineering..."
	uv run jupyter nbconvert --to notebook --execute notebooks/02_feature_engineering.ipynb --inplace
	@echo "✅ Dataset procesado guardado en data/processed/"

run-notebook-03: ## Ejecutar solo notebook 03 (Model Training)
	@echo "🤖 Ejecutando Model Training..."
	uv run jupyter nbconvert --to notebook --execute notebooks/03_model_training.ipynb --inplace
	@echo "✅ Modelo guardado en models/"

run-notebook-04: ## Ejecutar solo notebook 04 (Cluster-Stratified Training)
	@echo "🎯 Ejecutando Cluster-Stratified Model Training..."
	uv run jupyter nbconvert --to notebook --execute notebooks/04_cluster_stratified_training.ipynb --inplace
	@echo "✅ Modelos por cluster guardados en models/"

train-clusters: ## Entrenar modelos estratificados por cluster (notebooks 01, 02, 03, 04) [LEGACY]
	@echo "🔬 Ejecutando pipeline completo con cluster stratification..."
	@echo "📊 Paso 1/3: Exploratory Analysis + Clustering (notebook 01)..."
	uv run jupyter nbconvert --to notebook --execute notebooks/01_exploratory_analysis.ipynb --inplace
	@echo "✅ Clusters generados"
	@echo "📊 Paso 2/3: Feature Engineering (notebook 02)..."
	uv run jupyter nbconvert --to notebook --execute notebooks/02_feature_engineering.ipynb --inplace
	@echo "✅ Features generadas"
	@echo "📊 Paso 3/3: Cluster-Stratified Training (notebook 04)..."
	uv run jupyter nbconvert --to notebook --execute notebooks/04_cluster_stratified_training.ipynb --inplace
	@echo "✅ Modelos por cluster entrenados"
	@echo ""
	@echo "📋 Modelos generados:"
	@echo "  - models/rf_severity_classifier_cluster_0.pkl"
	@echo "  - models/rf_severity_classifier_cluster_1.pkl"
	@echo "  - models/rf_severity_classifier_cluster_2.pkl"
	@echo "  - models/cluster_kmeans.pkl (para inferencia)"
	@echo "  - models/cluster_scaler.pkl"
	@echo ""
	@echo "💡 Siguiente paso: make serve (la API usará automáticamente los modelos por cluster)"

run-notebook-05: ## Ejecutar solo notebook 05 (Cluster-Stratified Training CU)
	@echo "🎯 Ejecutando Cluster-Stratified Model Training (CU)..."
	uv run jupyter nbconvert --to notebook --execute notebooks/05_cluster_stratified_training_cu.ipynb --inplace
	@echo "✅ Modelos CU por cluster guardados en models/cu/"

train-crohn: ## Entrenar pipeline completo para Crohn (notebooks 01, 02, 03, 04)
	@echo "🔬 Pipeline completo: CROHN"
	@echo "═══════════════════════════════════════"
	@echo "📊 Paso 1/4: Exploratory Analysis + Clustering (Crohn)..."
	uv run jupyter nbconvert --to notebook --execute notebooks/01_exploratory_analysis.ipynb --inplace
	@echo "✅ Clusters Crohn generados"
	@echo "📊 Paso 2/4: Feature Engineering (Crohn)..."
	uv run jupyter nbconvert --to notebook --execute notebooks/02_feature_engineering.ipynb --inplace
	@echo "✅ Features base generadas"
	@echo "📊 Paso 3/4: Advanced Feature Engineering (34 features)..."
	uv run jupyter nbconvert --to notebook --execute notebooks/03_advanced_feature_engineering.ipynb --inplace
	@echo "✅ Features derivadas generadas (21 nuevas)"
	@echo "📊 Paso 4/4: Cluster-Stratified Training (Crohn)..."
	uv run jupyter nbconvert --to notebook --execute notebooks/04_cluster_stratified_training.ipynb --inplace
	@echo "✅ Modelos Crohn entrenados (34 features)"
	@echo ""
	@echo "📋 Modelos generados en models/crohn/:"
	@echo "  - rf_severity_classifier_cluster_0.pkl"
	@echo "  - rf_severity_classifier_cluster_1.pkl"
	@echo "  - rf_severity_classifier_cluster_2.pkl"
	@echo "  - rf_severity_classifier_global.pkl"
	@echo "  - cluster_kmeans.pkl"
	@echo "  - cluster_scaler.pkl"

train-cu: ## Entrenar pipeline completo para CU (notebooks 01, 02, 03, 05)
	@echo "🔬 Pipeline completo: COLITIS ULCEROSA"
	@echo "═══════════════════════════════════════"
	@echo "📊 Paso 1/4: Exploratory Analysis + Clustering (CU)..."
	uv run jupyter nbconvert --to notebook --execute notebooks/01_exploratory_analysis.ipynb --inplace
	@echo "✅ Clusters CU generados"
	@echo "📊 Paso 2/4: Feature Engineering (CU)..."
	uv run jupyter nbconvert --to notebook --execute notebooks/02_feature_engineering.ipynb --inplace
	@echo "✅ Features base generadas"
	@echo "📊 Paso 3/4: Advanced Feature Engineering (34 features)..."
	uv run jupyter nbconvert --to notebook --execute notebooks/03_advanced_feature_engineering.ipynb --inplace
	@echo "✅ Features derivadas generadas (21 nuevas)"
	@echo "📊 Paso 4/4: Cluster-Stratified Training (CU)..."
	uv run jupyter nbconvert --to notebook --execute notebooks/05_cluster_stratified_training_cu.ipynb --inplace
	@echo "✅ Modelos CU entrenados (34 features)"
	@echo ""
	@echo "📋 Modelos generados en models/cu/:"
	@echo "  - rf_severity_classifier_cluster_0.pkl"
	@echo "  - rf_severity_classifier_cluster_1.pkl"
	@echo "  - rf_severity_classifier_cluster_2.pkl"
	@echo "  - rf_severity_classifier_global.pkl"
	@echo "  - cluster_kmeans.pkl"
	@echo "  - cluster_scaler.pkl"

train-all: ## Entrenar TODOS los modelos (Crohn + CU)
	@echo "🚀 PIPELINE COMPLETO: CROHN + CU"
	@echo "════════════════════════════════════════════════════════"
	@echo ""
	@echo "📊 Fase 1/2: Entrenando modelos CROHN..."
	@echo "───────────────────────────────────────────────────────"
	@$(MAKE) train-crohn
	@echo ""
	@echo "📊 Fase 2/2: Entrenando modelos CU..."
	@echo "───────────────────────────────────────────────────────"
	@$(MAKE) train-cu
	@echo ""
	@echo "════════════════════════════════════════════════════════"
	@echo "✅ TODOS LOS MODELOS ENTRENADOS"
	@echo "════════════════════════════════════════════════════════"
	@echo ""
	@echo "📋 Modelos disponibles:"
	@echo "  🔹 CROHN:  models/crohn/rf_severity_classifier_cluster_*.pkl"
	@echo "  🔹 CU:     models/cu/rf_severity_classifier_cluster_*.pkl"
	@echo ""
	@echo "💡 Siguiente paso:"
	@echo "   make serve  (La API cargará automáticamente ambos tipos)"

setup-data: ## Crear estructura de directorios para datos
	@echo "📁 Creando estructura de directorios..."
	mkdir -p data/raw data/processed models/crohn models/cu reports/evaluations docs/figures
	@echo "✅ Directorios creados:"
	@echo "   - data/raw data/processed"
	@echo "   - models/crohn models/cu"
	@echo "   - reports/evaluations"
	@echo "   - docs/figures"
	@echo "⚠️  Recuerda descargar el dataset de Kaggle en data/raw/"

dev: sync setup-data ## Setup completo para desarrollo
	@echo "✅ Entorno de desarrollo listo"
	@echo "Ejecuta 'make notebook' para empezar a analizar datos"
	@echo "Ejecuta 'make serve' para levantar la API"

pipeline: train ## Pipeline completo: entrenar modelo
	@echo ""
	@echo "✅ Pipeline de entrenamiento completado"
	@echo ""
	@echo "📋 Próximos pasos:"
	@echo "  1. make serve          - Iniciar servidor API"
	@echo "  2. make evaluate       - Evaluar modelo (en otra terminal)"
	@echo "  3. make info           - Ver estado del proyecto"

# Docker commands (si decides usar Docker más adelante)
docker-build: ## Construir imagen Docker
	docker build -t crohn-flare-predictor:latest .

docker-run: ## Ejecutar contenedor Docker
	docker run -p 8000:8000 crohn-flare-predictor:latest

# Comandos de utilidad
add: ## Agregar nueva dependencia (uso: make add PKG=nombre-paquete)
	uv add $(PKG)

remove: ## Remover dependencia (uso: make remove PKG=nombre-paquete)
	uv remove $(PKG)

update: ## Actualizar todas las dependencias
	uv sync --upgrade

lock: ## Actualizar lockfile sin instalar
	uv lock

shell: ## Abrir shell en el entorno virtual
	uv run bash

python: ## Abrir Python REPL en el entorno
	$(PYTHON)

info: ## Mostrar información del proyecto
	@echo "📋 Información del Proyecto"
	@echo "═══════════════════════════"
	@echo "Nombre: crohn-flare-predictor"
	@echo "Versión: 1.0.0"
	@echo "Python: $(shell $(PYTHON) --version)"
	@echo ""
	@echo "📊 Estado del Modelo:"
	@if [ -f models/rf_severity_classifier.pkl ]; then \
		echo "  ✅ Modelo entrenado: models/rf_severity_classifier.pkl"; \
		ls -lh models/rf_severity_classifier.pkl | awk '{print "     Tamaño: " $$5 " - Modificado: " $$6 " " $$7 " " $$8}'; \
	else \
		echo "  ❌ No hay modelo entrenado (ejecuta: make train)"; \
	fi
	@echo ""
	@echo "📂 Datasets:"
	@if [ -f data/processed/ml_dataset.csv ]; then \
		echo "  ✅ Dataset procesado: data/processed/ml_dataset.csv"; \
		ls -lh data/processed/ml_dataset.csv | awk '{print "     Tamaño: " $$5}'; \
	else \
		echo "  ❌ Dataset no procesado (ejecuta: make run-notebook-02)"; \
	fi
	@echo ""
	@echo "📦 Dependencias principales:"
	@uv pip list | grep -E "(fastapi|scikit-learn|pandas|uvicorn|imbalanced-learn)" || true

reports: ## Ver últimos reportes de evaluación
	@echo "📊 Últimos Reportes de Evaluación"
	@echo "════════════════════════════════"
	@if [ -d reports/evaluations ]; then \
		ls -lt reports/evaluations/*.json 2>/dev/null | head -5 | while read -r line; do \
			file=$$(echo $$line | awk '{print $$NF}'); \
			date=$$(echo $$line | awk '{print $$6, $$7, $$8}'); \
			echo "  📄 $$(basename $$file) - $$date"; \
		done || echo "  ⚠️  No hay reportes aún (ejecuta: make evaluate)"; \
	else \
		echo "  ⚠️  Directorio reports/evaluations no existe"; \
	fi
	@echo ""
	@echo "💡 Para ver un reporte: cat reports/evaluations/evaluation_YYYYMMDD_HHMMSS.json | jq"

show-latest-report: ## Mostrar último reporte de evaluación
	@echo "📊 Último Reporte de Evaluación"
	@echo "═══════════════════════════════"
	@latest=$$(ls -t reports/evaluations/*.json 2>/dev/null | head -1); \
	if [ -n "$$latest" ]; then \
		echo "📄 Archivo: $$(basename $$latest)"; \
		echo ""; \
		cat "$$latest" | $(PYTHON) -m json.tool; \
	else \
		echo "⚠️  No hay reportes disponibles"; \
		echo "💡 Ejecuta: make evaluate (requiere servidor activo)"; \
	fi

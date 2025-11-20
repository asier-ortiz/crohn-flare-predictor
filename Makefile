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

test: ## Ejecutar tests con pytest
	@echo "Ejecutando tests..."
	$(PYTEST)

test-cov: ## Ejecutar tests con cobertura
	@echo "Ejecutando tests con cobertura..."
	$(PYTEST) --cov=src --cov-report=html --cov-report=term
	@echo "📊 Reporte de cobertura generado en htmlcov/index.html"

format: ## Formatear código con Black
	@echo "Formateando código..."
	$(BLACK) src/ api/ tests/ scripts/
	@echo "✅ Código formateado"

lint: ## Verificar código con flake8
	@echo "Verificando código..."
	$(FLAKE8) src/ api/ tests/ scripts/
	@echo "✅ Verificación completada"

check: format lint test ## Ejecutar formato, lint y tests

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

predict: ## Ejecutar predicción de ejemplo
	@echo "🔮 Ejecutando predicción..."
	$(PYTHON) -m src.model --predict

setup-data: ## Crear estructura de directorios para datos
	@echo "📁 Creando estructura de directorios..."
	mkdir -p data/raw data/processed models logs reports/evaluations docs/figures
	@echo "✅ Directorios creados:"
	@echo "   - data/raw data/processed"
	@echo "   - models logs"
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

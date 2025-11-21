#!/bin/bash
# Script para limpiar archivos antiguos/temporales del repositorio local
# Ejecutar desde la raíz del proyecto: bash scripts/cleanup_local.sh

echo "🧹 Limpieza de archivos locales antiguos/temporales"
echo "=================================================="
echo ""

# Confirmar antes de proceder
read -p "⚠️  Esto eliminará archivos locales (datos, modelos antiguos, reportes). ¿Continuar? (y/N): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Operación cancelada"
    exit 0
fi

echo ""
echo "🗑️  Eliminando archivos..."
echo ""

# 1. Directorio logs vacío
if [ -d "logs" ]; then
    rm -rf logs
    echo "✅ Eliminado: logs/"
fi

# 2. README.md vacío en data/
if [ -f "data/README.md" ]; then
    rm -f data/README.md
    echo "✅ Eliminado: data/README.md"
fi

# 3. api/requirements.txt vacío
if [ -f "api/requirements.txt" ]; then
    rm -f api/requirements.txt
    echo "✅ Eliminado: api/requirements.txt"
fi

# 4. scripts/api_examples.json (no usado)
if [ -f "scripts/api_examples.json" ]; then
    rm -f scripts/api_examples.json
    echo "✅ Eliminado: scripts/api_examples.json"
fi

# 5. Modelos antiguos en la raíz de models/ (antes de reorganización crohn/cu)
echo ""
echo "📦 Limpiando modelos antiguos en models/ (raíz)..."
cd models 2>/dev/null
if [ $? -eq 0 ]; then
    # Eliminar archivos antiguos, mantener subdirectorios crohn/ y cu/
    rm -f cluster_kmeans.pkl
    rm -f cluster_metadata.json
    rm -f cluster_models_metadata.json
    rm -f cluster_scaler.pkl
    rm -f rf_severity_classifier.pkl
    rm -f rf_severity_classifier_cluster_*.pkl
    rm -f rf_severity_classifier_metadata.json
    echo "✅ Eliminados modelos antiguos de models/ (raíz)"
    cd ..
fi

# 6. Archivos temporales/duplicados en data/processed/
echo ""
echo "📂 Limpiando archivos temporales en data/processed/..."
if [ -d "data/processed" ]; then
    cd data/processed

    # Eliminar archivos duplicados en la raíz (ya están en crohn/ y cu/)
    rm -f cluster_profiles.csv
    rm -f ml_dataset.csv
    rm -f ml_dataset_metadata.json
    rm -f user_clusters.csv

    # Eliminar muestras temporales
    rm -f crohn_sample_10k.csv
    rm -f sample_50000.csv
    rm -f crohn_filtered.csv
    rm -f cu_filtered.csv

    echo "✅ Eliminados archivos temporales de data/processed/"
    cd ../..
fi

# 7. Reportes antiguos en reports/ (raíz)
echo ""
echo "📊 Limpiando reportes antiguos en reports/..."
if [ -d "reports" ]; then
    cd reports

    # Eliminar reportes JSON en la raíz, mantener reports/evaluations/
    rm -f cluster_stratified_training_report.json
    rm -f crohn_cluster_stratified_training_report.json
    rm -f cu_cluster_stratified_training_report.json

    echo "✅ Eliminados reportes antiguos de reports/ (raíz)"

    # Opcional: limpiar evaluaciones antiguas (descomentar si quieres)
    # echo "  ⚠️  Manteniendo evaluaciones en reports/evaluations/"
    # echo "  💡 Para limpiar evaluaciones: rm reports/evaluations/*.json"

    cd ..
fi

echo ""
echo "════════════════════════════════════════════════════════"
echo "✅ LIMPIEZA COMPLETADA"
echo "════════════════════════════════════════════════════════"
echo ""
echo "📂 Estructura limpia:"
echo "  ✅ models/crohn/       (modelos Crohn)"
echo "  ✅ models/cu/          (modelos UC)"
echo "  ✅ data/processed/crohn/   (datos Crohn)"
echo "  ✅ data/processed/cu/      (datos UC)"
echo "  ✅ reports/evaluations/    (evaluaciones)"
echo ""
echo "🗑️  Eliminados:"
echo "  ❌ logs/"
echo "  ❌ models/*.pkl (9 archivos antiguos)"
echo "  ❌ data/processed/*.csv (archivos temporales/duplicados)"
echo "  ❌ reports/*.json (reportes antiguos en raíz)"
echo "  ❌ Archivos vacíos/no usados"
echo ""
echo "💡 Siguiente paso: git add -A && git commit -m 'chore: clean local files'"
echo ""

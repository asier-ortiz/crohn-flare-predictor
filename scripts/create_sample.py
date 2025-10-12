#!/usr/bin/env python3
"""
Script para crear un sample del dataset de Flaredown
Útil para compartir y debugging sin enviar el archivo completo (600MB+)
"""

import pandas as pd
import os
import sys


def create_sample(input_path, output_dir, max_rows=50000, target_size_mb=25):
    """
    Crea un sample del dataset que sea compartible

    Args:
        input_path: Ruta al CSV original
        output_dir: Directorio donde guardar el sample
        max_rows: Número máximo de filas inicial
        target_size_mb: Tamaño objetivo en MB
    """

    print("=" * 70)
    print("CREACIÓN DE SAMPLE DEL DATASET FLAREDOWN")
    print("=" * 70)

    # Verificar que existe el archivo
    if not os.path.exists(input_path):
        print(f"❌ Error: No se encuentra el archivo {input_path}")
        print(f"\n🔍 Verificando ubicación...")
        print(f"   Directorio actual: {os.getcwd()}")
        print(f"   Ruta buscada: {os.path.abspath(input_path)}")
        sys.exit(1)

    print(f"\n📂 Archivo original: {input_path}")
    original_size = os.path.getsize(input_path) / (1024 * 1024)
    print(f"💾 Tamaño original: {original_size:.2f} MB")

    # Leer muestra inicial
    print(f"\n📊 Cargando primeras {max_rows:,} filas...")
    df = pd.read_csv(input_path, nrows=max_rows)

    print(f"✓ Cargadas {len(df):,} filas × {df.shape[1]} columnas")

    # Mostrar información básica
    print(f"\n📋 COLUMNAS DEL DATASET:")
    for i, col in enumerate(df.columns, 1):
        dtype = df[col].dtype
        unique = df[col].nunique()
        nulls = df[col].isnull().sum()
        print(f"  {i:2d}. {col:30s} | Tipo: {str(dtype):10s} | Únicos: {unique:6,} | Nulos: {nulls:5,}")

    # Mostrar primeras filas
    print(f"\n📈 PRIMERAS 5 FILAS:")
    print(df.head())

    # Información adicional
    print(f"\n📊 INFORMACIÓN DETALLADA:")
    print(df.info())

    # Crear directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)

    # Guardar sample inicial
    output_filename = f'sample_{len(df)}.csv'
    output_path = os.path.join(output_dir, output_filename)
    df.to_csv(output_path, index=False)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n✓ Sample guardado: {output_path}")
    print(f"💾 Tamaño: {size_mb:.2f} MB")

    # Si es muy grande, reducir
    if size_mb > target_size_mb:
        print(f"\n⚠️ El archivo supera {target_size_mb}MB. Reduciendo...")

        # Calcular cuántas filas necesitamos
        rows_needed = int(len(df) * (target_size_mb / size_mb) * 0.9)  # 90% del target para margen
        print(f"📉 Reduciendo a ~{rows_needed:,} filas...")

        df_small = df.sample(rows_needed, random_state=42)

        output_filename_small = f'sample_{rows_needed}.csv'
        output_path_small = os.path.join(output_dir, output_filename_small)
        df_small.to_csv(output_path_small, index=False)

        size_mb_small = os.path.getsize(output_path_small) / (1024 * 1024)
        print(f"✓ Sample reducido guardado: {output_path_small}")
        print(f"💾 Nuevo tamaño: {size_mb_small:.2f} MB")

        # Eliminar el sample grande
        os.remove(output_path)
        output_path = output_path_small
        size_mb = size_mb_small
        final_rows = rows_needed
    else:
        final_rows = len(df)

    # Resumen final
    print("\n" + "=" * 70)
    print("RESUMEN")
    print("=" * 70)
    print(f"✓ Sample creado exitosamente")
    print(f"📁 Ubicación: {output_path}")
    print(f"📊 Filas: {final_rows:,}")
    print(f"💾 Tamaño final: {size_mb:.2f} MB")
    print(f"📦 Compresión: {(1 - size_mb / original_size) * 100:.1f}% del original")

    # Sugerencias
    print(f"\n💡 PRÓXIMOS PASOS:")
    print(f"  1. Puedes compartir este archivo: {output_path}")
    print(f"  2. O comprimirlo más: zip {output_path.replace('.csv', '.zip')} {output_path}")
    print(f"  3. Usar en notebooks para desarrollo rápido")

    return output_path


if __name__ == '__main__':
    # Obtener rutas absolutas independientemente de desde dónde se ejecute
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # Subir un nivel desde scripts/

    # Configuración con rutas absolutas
    INPUT_FILE = os.path.join(project_root, 'data', 'raw', 'export.csv')
    OUTPUT_DIR = os.path.join(project_root, 'data', 'processed')
    MAX_ROWS = 50000  # Número inicial de filas a cargar
    TARGET_SIZE_MB = 25  # Tamaño objetivo del sample

    print(f"\n📂 Directorio del proyecto: {project_root}")
    print(f"📂 Buscando archivo en: {INPUT_FILE}")

    # Crear sample
    try:
        sample_path = create_sample(
            input_path=INPUT_FILE,
            output_dir=OUTPUT_DIR,
            max_rows=MAX_ROWS,
            target_size_mb=TARGET_SIZE_MB
        )
        print(f"\n✅ Proceso completado exitosamente")

    except Exception as e:
        print(f"\n❌ Error durante la creación del sample:")
        print(f"   {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
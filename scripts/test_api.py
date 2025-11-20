#!/usr/bin/env python
"""
Script para probar los endpoints de la API.
"""
import httpx
import json
from datetime import date, timedelta

BASE_URL = "http://localhost:8001"


def test_predict_single():
    """Probar predicción individual."""
    print("\n" + "="*60)
    print("TEST: /predict - Predicción Individual")
    print("="*60)

    payload = {
        "symptoms": {
            "abdominal_pain": 7,
            "diarrhea": 6,
            "fatigue": 5,
            "fever": False,
            "weight_change": -2.5,
            "blood_in_stool": False,
            "nausea": 4
        },
        "demographics": {
            "age": 32,
            "gender": "F",
            "disease_duration_years": 5,
            "bmi": 22.5
        },
        "history": {
            "previous_flares": 3,
            "medications": ["mesalamine", "prednisone"],
            "last_flare_days_ago": 120,
            "surgery_history": False,
            "smoking_status": "never"
        }
    }

    print("\n📤 Request:")
    print(json.dumps(payload, indent=2))

    response = httpx.post(f"{BASE_URL}/predict", json=payload)

    print(f"\n📥 Response (Status: {response.status_code}):")
    if response.status_code == 200:
        result = response.json()
        print(json.dumps(result, indent=2))

        print("\n📊 Resumen:")
        print(f"  • Riesgo: {result['prediction']['flare_risk'].upper()}")
        print(f"  • Probabilidad: {result['prediction']['probability']*100:.1f}%")
        print(f"  • Confianza: {result['prediction']['confidence']*100:.1f}%")
        print(f"  • Factores principales: {', '.join(result['factors']['top_contributors'])}")
        print(f"  • Recomendación: {result['recommendation']}")
    else:
        print(response.text)


def test_predict_high_risk():
    """Probar predicción de alto riesgo."""
    print("\n" + "="*60)
    print("TEST: /predict - Caso de Alto Riesgo")
    print("="*60)

    payload = {
        "symptoms": {
            "abdominal_pain": 9,
            "diarrhea": 8,
            "fatigue": 7,
            "fever": True,
            "weight_change": -5.0,
            "blood_in_stool": True,
            "nausea": 7
        },
        "demographics": {
            "age": 45,
            "gender": "M",
            "disease_duration_years": 10
        },
        "history": {
            "previous_flares": 6,
            "medications": ["infliximab", "azathioprine"],
            "last_flare_days_ago": 45,
            "surgery_history": True,
            "smoking_status": "former"
        }
    }

    print("\n📤 Request:")
    print(json.dumps(payload, indent=2))

    response = httpx.post(f"{BASE_URL}/predict", json=payload)

    print(f"\n📥 Response (Status: {response.status_code}):")
    if response.status_code == 200:
        result = response.json()
        print(json.dumps(result, indent=2))

        print("\n📊 Resumen:")
        print(f"  • Riesgo: {result['prediction']['flare_risk'].upper()}")
        print(f"  • Probabilidad: {result['prediction']['probability']*100:.1f}%")
        print(f"  • Confianza: {result['prediction']['confidence']*100:.1f}%")
    else:
        print(response.text)


def test_predict_batch():
    """Probar predicción por lotes."""
    print("\n" + "="*60)
    print("TEST: /predict/batch - Predicción por Lotes")
    print("="*60)

    payload = {
        "patients": [
            {
                "patient_id": "P001",
                "symptoms": {
                    "abdominal_pain": 3,
                    "diarrhea": 2,
                    "fatigue": 4,
                    "fever": False,
                    "weight_change": 0.5,
                    "blood_in_stool": False,
                    "nausea": 1
                },
                "demographics": {
                    "age": 28,
                    "gender": "F",
                    "disease_duration_years": 2
                },
                "history": {
                    "previous_flares": 1,
                    "medications": ["mesalamine"],
                    "last_flare_days_ago": 365
                }
            },
            {
                "patient_id": "P002",
                "symptoms": {
                    "abdominal_pain": 8,
                    "diarrhea": 7,
                    "fatigue": 6,
                    "fever": True,
                    "weight_change": -3.0,
                    "blood_in_stool": True,
                    "nausea": 5
                },
                "demographics": {
                    "age": 35,
                    "gender": "M",
                    "disease_duration_years": 8
                },
                "history": {
                    "previous_flares": 4,
                    "medications": ["adalimumab", "methotrexate"],
                    "last_flare_days_ago": 60
                }
            },
            {
                "patient_id": "P003",
                "symptoms": {
                    "abdominal_pain": 5,
                    "diarrhea": 5,
                    "fatigue": 5,
                    "fever": False,
                    "weight_change": -1.0,
                    "blood_in_stool": False,
                    "nausea": 3
                },
                "demographics": {
                    "age": 42,
                    "gender": "F",
                    "disease_duration_years": 6
                },
                "history": {
                    "previous_flares": 2,
                    "medications": ["budesonide"],
                    "last_flare_days_ago": 180
                }
            }
        ]
    }

    print(f"\n📤 Request: {len(payload['patients'])} pacientes")

    response = httpx.post(f"{BASE_URL}/predict/batch", json=payload)

    print(f"\n📥 Response (Status: {response.status_code}):")
    if response.status_code == 200:
        result = response.json()
        print(json.dumps(result, indent=2))

        print("\n📊 Resumen:")
        print(f"  • Procesados: {result['processed_count']}")
        print(f"  • Fallidos: {result['failed_count']}")

        print("\n  Resultados por paciente:")
        for patient_result in result['results']:
            pid = patient_result['patient_id']
            risk = patient_result['prediction']['flare_risk']
            prob = patient_result['prediction']['probability'] * 100
            print(f"    - {pid}: {risk.upper()} ({prob:.1f}%)")
    else:
        print(response.text)


def test_analyze_trends():
    """Probar análisis de tendencias."""
    print("\n" + "="*60)
    print("TEST: /analyze/trends - Análisis de Tendencias")
    print("="*60)

    # Generar 14 días de datos con tendencia empeorando
    today = date.today()
    daily_records = []

    for i in range(14):
        record_date = today - timedelta(days=14-i)
        # Síntomas empeorando gradualmente
        severity_factor = i / 14

        daily_records.append({
            "date": record_date.isoformat(),
            "symptoms": {
                "abdominal_pain": int(3 + severity_factor * 5),
                "diarrhea": int(2 + severity_factor * 6),
                "fatigue": int(3 + severity_factor * 4),
                "fever": i > 10,
                "weight_change": -severity_factor * 2,
                "blood_in_stool": i > 12,
                "nausea": int(2 + severity_factor * 4)
            }
        })

    payload = {
        "patient_id": "P001",
        "daily_records": daily_records,
        "window_days": 14
    }

    print(f"\n📤 Request: {len(daily_records)} días de datos")
    print(f"  Período: {daily_records[0]['date']} a {daily_records[-1]['date']}")

    response = httpx.post(f"{BASE_URL}/analyze/trends", json=payload)

    print(f"\n📥 Response (Status: {response.status_code}):")
    if response.status_code == 200:
        result = response.json()
        print(json.dumps(result, indent=2))

        print("\n📊 Resumen:")
        print(f"  • Tendencia: {result['trends']['overall_trend'].upper()}")
        print(f"  • Cambio severidad: {result['trends']['severity_change']:+.2f}")
        print(f"  • Patrones preocupantes: {len(result['trends']['concerning_patterns'])}")
        if result['trends']['concerning_patterns']:
            for pattern in result['trends']['concerning_patterns']:
                print(f"    - {pattern}")
        print(f"  • Riesgo actual: {result['risk_assessment']['flare_risk'].upper()}")
        print(f"  • Recomendaciones:")
        for rec in result['recommendations']:
            print(f"    - {rec}")
    else:
        print(response.text)


def test_model_info():
    """Obtener información del modelo."""
    print("\n" + "="*60)
    print("TEST: /model/info - Información del Modelo")
    print("="*60)

    response = httpx.get(f"{BASE_URL}/model/info")

    print(f"\n📥 Response (Status: {response.status_code}):")
    if response.status_code == 200:
        result = response.json()
        print(json.dumps(result, indent=2))

        print("\n📊 Resumen:")
        print(f"  • Versión: {result['model_version']}")
        print(f"  • Tipo: {result['model_type']}")
        print(f"  • Fecha entrenamiento: {result['trained_date']}")
        print(f"  • Características: {result['features_count']}")
        print(f"  • Métricas:")
        print(f"    - Accuracy: {result['metrics']['accuracy']:.2%}")
        print(f"    - Precision: {result['metrics']['precision']:.2%}")
        print(f"    - Recall: {result['metrics']['recall']:.2%}")
        print(f"    - F1-Score: {result['metrics']['f1_score']:.2%}")
        if result['metrics']['roc_auc']:
            print(f"    - ROC-AUC: {result['metrics']['roc_auc']:.2%}")
    else:
        print(response.text)


def main():
    """Ejecutar todos los tests."""
    print("\n🚀 Iniciando pruebas de API")
    print("Asegúrate de que el servidor está corriendo en http://localhost:8001")

    try:
        # Verificar que el servidor está activo
        response = httpx.get(f"{BASE_URL}/health")
        if response.status_code != 200:
            print("❌ Error: El servidor no está respondiendo correctamente")
            return
        print("✅ Servidor activo\n")

        # Ejecutar tests
        test_model_info()
        test_predict_single()
        test_predict_high_risk()
        test_predict_batch()
        test_analyze_trends()

        print("\n" + "="*60)
        print("✅ Todas las pruebas completadas")
        print("="*60)

    except httpx.ConnectError:
        print("\n❌ Error: No se puede conectar al servidor")
        print("Asegúrate de ejecutar: make serve")
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()

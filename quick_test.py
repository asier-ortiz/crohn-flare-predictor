#!/usr/bin/env python
"""
Script rápido para probar la API ML.
Uso: uv run python quick_test.py
"""
import httpx
import json


BASE_URL = "http://localhost:8001"


def test_health():
    """Test health check."""
    print("\n" + "="*60)
    print("TEST 1: Health Check")
    print("="*60)

    response = httpx.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")

    if response.status_code == 200:
        print("✅ API is healthy!")
    else:
        print("❌ API not healthy")
        return False

    return True


def test_low_risk_prediction():
    """Test predicción de bajo riesgo."""
    print("\n" + "="*60)
    print("TEST 2: Predicción - Bajo Riesgo")
    print("="*60)

    payload = {
        "symptoms": {
            "abdominal_pain": 2,
            "diarrhea": 1,
            "fatigue": 3,
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
            "previous_flares": 0,
            "medications": ["mesalamine"],
            "last_flare_days_ago": 365
        }
    }

    response = httpx.post(f"{BASE_URL}/predict", json=payload)

    if response.status_code == 200:
        result = response.json()
        print(f"\n📊 Resultado:")
        print(f"  Riesgo: {result['prediction']['flare_risk'].upper()}")
        print(f"  Probabilidad: {result['prediction']['probability']*100:.1f}%")
        print(f"  Confianza: {result['prediction']['confidence']*100:.1f}%")
        print(f"  Factores: {', '.join(result['factors']['top_contributors'])}")
        print(f"\n💡 Recomendación:")
        print(f"  {result['recommendation']}")
        print("\n✅ Test pasado!")
        return True
    else:
        print(f"❌ Error {response.status_code}: {response.text}")
        return False


def test_high_risk_prediction():
    """Test predicción de alto riesgo."""
    print("\n" + "="*60)
    print("TEST 3: Predicción - Alto Riesgo")
    print("="*60)

    payload = {
        "symptoms": {
            "abdominal_pain": 9,
            "diarrhea": 8,
            "fatigue": 7,
            "fever": True,
            "weight_change": -4.0,
            "blood_in_stool": True,
            "nausea": 6
        },
        "demographics": {
            "age": 42,
            "gender": "M",
            "disease_duration_years": 8
        },
        "history": {
            "previous_flares": 5,
            "medications": ["infliximab", "prednisone"],
            "last_flare_days_ago": 30,
            "surgery_history": True
        }
    }

    response = httpx.post(f"{BASE_URL}/predict", json=payload)

    if response.status_code == 200:
        result = response.json()
        print(f"\n📊 Resultado:")
        print(f"  Riesgo: {result['prediction']['flare_risk'].upper()}")
        print(f"  Probabilidad: {result['prediction']['probability']*100:.1f}%")
        print(f"  Confianza: {result['prediction']['confidence']*100:.1f}%")
        print(f"  Factores: {', '.join(result['factors']['top_contributors'])}")
        print(f"\n💡 Recomendación:")
        print(f"  {result['recommendation']}")
        print("\n✅ Test pasado!")
        return True
    else:
        print(f"❌ Error {response.status_code}: {response.text}")
        return False


def test_batch_prediction():
    """Test predicción por lotes."""
    print("\n" + "="*60)
    print("TEST 4: Predicción por Lotes")
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
                    "weight_change": 0.0
                },
                "demographics": {
                    "age": 30,
                    "gender": "F",
                    "disease_duration_years": 3
                },
                "history": {
                    "previous_flares": 1,
                    "medications": ["mesalamine"],
                    "last_flare_days_ago": 200
                }
            },
            {
                "patient_id": "P002",
                "symptoms": {
                    "abdominal_pain": 7,
                    "diarrhea": 6,
                    "fatigue": 5,
                    "fever": True,
                    "weight_change": -2.0
                },
                "demographics": {
                    "age": 45,
                    "gender": "M",
                    "disease_duration_years": 10
                },
                "history": {
                    "previous_flares": 4,
                    "medications": ["adalimumab"],
                    "last_flare_days_ago": 60
                }
            }
        ]
    }

    response = httpx.post(f"{BASE_URL}/predict/batch", json=payload)

    if response.status_code == 200:
        result = response.json()
        print(f"\n📊 Procesados: {result['processed_count']}")
        print(f"❌ Fallidos: {result['failed_count']}")
        print(f"\n📋 Resultados:")

        for patient in result['results']:
            print(f"\n  Paciente {patient['patient_id']}:")
            print(f"    Riesgo: {patient['prediction']['flare_risk'].upper()}")
            print(f"    Probabilidad: {patient['prediction']['probability']*100:.1f}%")

        print("\n✅ Test pasado!")
        return True
    else:
        print(f"❌ Error {response.status_code}: {response.text}")
        return False


def main():
    """Ejecutar todos los tests."""
    print("\n🚀 Iniciando pruebas de la API ML")
    print("Asegúrate de que el servidor está corriendo en http://localhost:8001")

    try:
        results = []

        # Ejecutar tests
        results.append(("Health Check", test_health()))
        results.append(("Predicción Bajo Riesgo", test_low_risk_prediction()))
        results.append(("Predicción Alto Riesgo", test_high_risk_prediction()))
        results.append(("Predicción por Lotes", test_batch_prediction()))

        # Resumen
        print("\n" + "="*60)
        print("📊 RESUMEN DE TESTS")
        print("="*60)

        for test_name, passed in results:
            status = "✅ PASADO" if passed else "❌ FALLADO"
            print(f"{test_name:.<40} {status}")

        total_passed = sum(1 for _, passed in results if passed)
        print(f"\nTotal: {total_passed}/{len(results)} tests pasados")

        if total_passed == len(results):
            print("\n🎉 ¡Todos los tests pasaron exitosamente!")
        else:
            print("\n⚠️ Algunos tests fallaron. Revisa los errores arriba.")

    except httpx.ConnectError:
        print("\n❌ ERROR: No se puede conectar al servidor")
        print("Asegúrate de ejecutar: make serve")
        print("O: uv run uvicorn api.app:app --reload --port 8001")


if __name__ == "__main__":
    main()

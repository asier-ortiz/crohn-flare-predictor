#!/usr/bin/env python
"""
Script para evaluar la precisión del modelo/sistema de predicción.

Uso:
    uv run python scripts/evaluate_model.py

Este script genera un reporte de métricas del sistema actual.
"""
import httpx
import json
from typing import List, Dict, Tuple
from datetime import date


BASE_URL = "http://localhost:8001"


# Dataset de casos de prueba con etiquetas reales
# Incluye casos de Crohn (L1-L4) y UC (E1-E3)
TEST_CASES = [
    # === CASOS DE BAJO RIESGO ===

    # Crohn L1 (ileal) - Bajo riesgo
    {
        "input": {
            "symptoms": {
                "abdominal_pain": 2,
                "diarrhea": 1,
                "fatigue": 3,
                "fever": False,
                "weight_change": 0.0,
                "blood_in_stool": False,
                "nausea": 1
            },
            "demographics": {
                "age": 28,
                "gender": "M",
                "disease_duration_years": 2,
                "bmi": 24.0,
                "ibd_type": "crohn",
                "montreal_location": "L1"
            },
            "history": {
                "previous_flares": 1,
                "medications": ["mesalamine"],
                "last_flare_days_ago": 365,
                "surgery_history": False,
                "smoking_status": "never"
            }
        },
        "expected_risk": "low",
        "case_description": "Crohn L1: Síntomas leves, bien controlado"
    },

    # UC E1 (proctitis) - Bajo riesgo
    {
        "input": {
            "symptoms": {
                "abdominal_pain": 1,
                "diarrhea": 2,
                "fatigue": 2,
                "fever": False,
                "weight_change": 0.5,
                "blood_in_stool": False,
                "nausea": 0
            },
            "demographics": {
                "age": 25,
                "gender": "F",
                "disease_duration_years": 1,
                "bmi": 22.0,
                "ibd_type": "ulcerative_colitis",
                "montreal_location": "E1"
            },
            "history": {
                "previous_flares": 0,
                "medications": ["mesalamine"],
                "last_flare_days_ago": 500,
                "surgery_history": False,
                "smoking_status": "never"
            }
        },
        "expected_risk": "low",
        "case_description": "UC E1 (proctitis): Muy leve, sin brotes previos"
    },

    # === CASOS DE MEDIO RIESGO ===

    # Crohn L2 (colonic) - Medio riesgo
    {
        "input": {
            "symptoms": {
                "abdominal_pain": 5,
                "diarrhea": 4,
                "fatigue": 5,
                "fever": False,
                "weight_change": -1.5,
                "blood_in_stool": False,
                "nausea": 3
            },
            "demographics": {
                "age": 35,
                "gender": "F",
                "disease_duration_years": 5,
                "bmi": 21.5,
                "ibd_type": "crohn",
                "montreal_location": "L2"
            },
            "history": {
                "previous_flares": 2,
                "medications": ["mesalamine", "prednisone"],
                "last_flare_days_ago": 180,
                "surgery_history": False,
                "smoking_status": "former"
            }
        },
        "expected_risk": "medium",
        "case_description": "Crohn L2: Síntomas moderados, algunos brotes previos"
    },

    # UC E2 (left-sided) - Medio riesgo
    {
        "input": {
            "symptoms": {
                "abdominal_pain": 6,
                "diarrhea": 5,
                "fatigue": 6,
                "fever": False,
                "weight_change": -2.0,
                "blood_in_stool": True,
                "nausea": 4
            },
            "demographics": {
                "age": 40,
                "gender": "M",
                "disease_duration_years": 7,
                "bmi": 23.0,
                "ibd_type": "ulcerative_colitis",
                "montreal_location": "E2"
            },
            "history": {
                "previous_flares": 3,
                "medications": ["mesalamine", "azathioprine"],
                "last_flare_days_ago": 150,
                "surgery_history": False,
                "smoking_status": "never"
            }
        },
        "expected_risk": "medium",
        "case_description": "UC E2: Moderado, sangre ocasional, historial de brotes"
    },

    # === CASOS DE ALTO RIESGO ===

    # Crohn L3 (ileocolonic) - Alto riesgo
    {
        "input": {
            "symptoms": {
                "abdominal_pain": 7,
                "diarrhea": 6,
                "fatigue": 5,
                "fever": False,
                "weight_change": -2.5,
                "blood_in_stool": True,
                "nausea": 4
            },
            "demographics": {
                "age": 32,
                "gender": "F",
                "disease_duration_years": 5.5,
                "bmi": 22.5,
                "ibd_type": "crohn",
                "montreal_location": "L3"
            },
            "history": {
                "previous_flares": 3,
                "medications": ["infliximab", "azathioprine"],
                "last_flare_days_ago": 120,
                "surgery_history": False,
                "smoking_status": "never"
            }
        },
        "expected_risk": "high",
        "case_description": "Crohn L3: Síntomas severos, sangre, brote reciente"
    },

    # UC E3 (extensive/pancolitis) - Alto riesgo
    {
        "input": {
            "symptoms": {
                "abdominal_pain": 8,
                "diarrhea": 9,
                "fatigue": 7,
                "fever": True,
                "weight_change": -4.5,
                "blood_in_stool": True,
                "nausea": 6
            },
            "demographics": {
                "age": 45,
                "gender": "M",
                "disease_duration_years": 8,
                "bmi": 21.0,
                "ibd_type": "ulcerative_colitis",
                "montreal_location": "E3"
            },
            "history": {
                "previous_flares": 5,
                "medications": ["mesalamine", "prednisone"],
                "last_flare_days_ago": 45,
                "surgery_history": False,
                "smoking_status": "former"
            }
        },
        "expected_risk": "high",
        "case_description": "UC E3 (pancolitis): Muy severo, fiebre, pérdida peso"
    },

    # Crohn L4 (upper GI) - Alto riesgo
    {
        "input": {
            "symptoms": {
                "abdominal_pain": 8,
                "diarrhea": 7,
                "fatigue": 7,
                "fever": True,
                "weight_change": -3.5,
                "blood_in_stool": True,
                "nausea": 7
            },
            "demographics": {
                "age": 50,
                "gender": "F",
                "disease_duration_years": 12,
                "bmi": 19.5,
                "ibd_type": "crohn",
                "montreal_location": "L4"
            },
            "history": {
                "previous_flares": 6,
                "medications": ["infliximab", "methotrexate"],
                "last_flare_days_ago": 30,
                "surgery_history": True,
                "smoking_status": "never"
            }
        },
        "expected_risk": "high",
        "case_description": "Crohn L4 (upper GI): Severo, fiebre, cirugía previa"
    },

    # Crohn L3 con historial extenso - Alto riesgo
    {
        "input": {
            "symptoms": {
                "abdominal_pain": 9,
                "diarrhea": 9,
                "fatigue": 8,
                "fever": True,
                "weight_change": -5.0,
                "blood_in_stool": True,
                "nausea": 8
            },
            "demographics": {
                "age": 55,
                "gender": "M",
                "disease_duration_years": 18,
                "bmi": 18.5,
                "ibd_type": "crohn",
                "montreal_location": "L3"
            },
            "history": {
                "previous_flares": 10,
                "medications": ["adalimumab", "methotrexate", "prednisone"],
                "last_flare_days_ago": 15,
                "surgery_history": True,
                "smoking_status": "former"
            }
        },
        "expected_risk": "high",
        "case_description": "Crohn L3: Muy severo, enfermedad larga, múltiples brotes"
    },
]


def evaluate_predictions() -> Dict:
    """
    Evaluar el modelo con casos de prueba.

    Returns:
        Dict con métricas de evaluación
    """
    print("\n" + "="*70)
    print("🔬 EVALUACIÓN DEL SISTEMA DE PREDICCIÓN")
    print("="*70)

    # Verificar que la API esté disponible
    try:
        response = httpx.get(f"{BASE_URL}/health", timeout=5.0)
        if response.status_code != 200:
            print("❌ La API no está disponible")
            return None
    except Exception as e:
        print(f"❌ Error conectando a la API: {e}")
        print("Asegúrate de ejecutar: make serve")
        return None

    print(f"✅ API disponible en {BASE_URL}")
    print(f"📊 Evaluando {len(TEST_CASES)} casos de prueba...\n")

    results = []
    confusion_matrix = {
        "low": {"low": 0, "medium": 0, "high": 0},
        "medium": {"low": 0, "medium": 0, "high": 0},
        "high": {"low": 0, "medium": 0, "high": 0},
    }

    for i, test_case in enumerate(TEST_CASES, 1):
        print(f"Caso {i}/{len(TEST_CASES)}: {test_case['case_description']}")

        try:
            # Hacer predicción
            response = httpx.post(
                f"{BASE_URL}/predict",
                json=test_case["input"],
                timeout=10.0
            )

            if response.status_code == 200:
                prediction = response.json()
                predicted_risk = prediction["prediction"]["flare_risk"]
                expected_risk = test_case["expected_risk"]

                # Actualizar matriz de confusión
                confusion_matrix[expected_risk][predicted_risk] += 1

                # Verificar si es correcto
                is_correct = predicted_risk == expected_risk
                status = "✅" if is_correct else "❌"

                print(f"  Esperado: {expected_risk.upper()}")
                print(f"  Predicho: {predicted_risk.upper()} {status}")

                # Mostrar distribución de probabilidades
                pred_obj = prediction['prediction']
                if 'probabilities' in pred_obj and pred_obj['probabilities']:
                    probs = pred_obj['probabilities']
                    print(f"  Distribución:")
                    print(f"    LOW:    {probs.get('low', 0)*100:5.1f}%")
                    print(f"    MEDIUM: {probs.get('medium', 0)*100:5.1f}%")
                    print(f"    HIGH:   {probs.get('high', 0)*100:5.1f}%")
                    print(f"  Confianza: {pred_obj['confidence']*100:.1f}% (diferencia entre top 2)")
                else:
                    print(f"  Probabilidad clase predicha: {pred_obj['probability']*100:.1f}%")

                results.append({
                    "case": i,
                    "expected": expected_risk,
                    "predicted": predicted_risk,
                    "correct": is_correct,
                    "probability": pred_obj["probability"],
                    "confidence": pred_obj["confidence"],
                    "probabilities": pred_obj.get("probabilities", {})
                })
            else:
                print(f"  ❌ Error en API: {response.status_code}")
                results.append({
                    "case": i,
                    "expected": test_case["expected_risk"],
                    "predicted": None,
                    "correct": False,
                    "error": response.status_code
                })

        except Exception as e:
            print(f"  ❌ Error: {e}")
            results.append({
                "case": i,
                "expected": test_case["expected_risk"],
                "predicted": None,
                "correct": False,
                "error": str(e)
            })

        print()

    # Calcular métricas
    metrics = calculate_metrics(results, confusion_matrix)

    return {
        "results": results,
        "confusion_matrix": confusion_matrix,
        "metrics": metrics
    }


def calculate_metrics(results: List[Dict], confusion_matrix: Dict) -> Dict:
    """Calcular métricas de evaluación."""

    total_cases = len([r for r in results if r["predicted"] is not None])
    correct_predictions = sum(1 for r in results if r["correct"])

    if total_cases == 0:
        return {}

    accuracy = correct_predictions / total_cases

    # Métricas por clase
    metrics_by_class = {}

    for risk_level in ["low", "medium", "high"]:
        # True Positives
        tp = confusion_matrix[risk_level][risk_level]

        # False Positives (predicho como esta clase pero era otra)
        fp = sum(confusion_matrix[other][risk_level]
                for other in ["low", "medium", "high"] if other != risk_level)

        # False Negatives (era esta clase pero se predijo otra)
        fn = sum(confusion_matrix[risk_level][other]
                for other in ["low", "medium", "high"] if other != risk_level)

        # True Negatives
        tn = sum(confusion_matrix[other1][other2]
                for other1 in ["low", "medium", "high"]
                for other2 in ["low", "medium", "high"]
                if other1 != risk_level and other2 != risk_level)

        # Calcular precision, recall, f1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        metrics_by_class[risk_level] = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "support": tp + fn  # Casos reales de esta clase
        }

    # Promedio macro
    macro_precision = sum(m["precision"] for m in metrics_by_class.values()) / 3
    macro_recall = sum(m["recall"] for m in metrics_by_class.values()) / 3
    macro_f1 = sum(m["f1_score"] for m in metrics_by_class.values()) / 3

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "by_class": metrics_by_class
    }


def print_report(evaluation: Dict):
    """Imprimir reporte de evaluación."""

    if not evaluation:
        return

    metrics = evaluation["metrics"]
    confusion_matrix = evaluation["confusion_matrix"]

    print("\n" + "="*70)
    print("📊 RESULTADOS DE EVALUACIÓN")
    print("="*70)

    # Accuracy general
    print(f"\n🎯 Accuracy General: {metrics['accuracy']*100:.1f}%")
    print(f"   ({sum(1 for r in evaluation['results'] if r['correct'])}/{len(evaluation['results'])} predicciones correctas)")

    # Matriz de confusión
    print("\n📈 Matriz de Confusión:")
    print("\n                 Predicho →")
    print("       Real ↓    LOW    MEDIUM   HIGH")
    print("       " + "-"*40)
    for actual in ["low", "medium", "high"]:
        print(f"       {actual.upper():8}", end="")
        for predicted in ["low", "medium", "high"]:
            count = confusion_matrix[actual][predicted]
            print(f"  {count:4}", end="")
        print()

    # Métricas por clase
    print("\n📊 Métricas por Clase de Riesgo:")
    print("\n  Clase    Precision  Recall   F1-Score  Support")
    print("  " + "-"*50)

    for risk_level in ["low", "medium", "high"]:
        m = metrics["by_class"][risk_level]
        print(f"  {risk_level.upper():8}  "
              f"{m['precision']*100:5.1f}%    "
              f"{m['recall']*100:5.1f}%   "
              f"{m['f1_score']*100:5.1f}%    "
              f"{m['support']}")

    # Promedios
    print("\n  " + "-"*50)
    print(f"  Macro Avg "
          f"{metrics['macro_precision']*100:5.1f}%    "
          f"{metrics['macro_recall']*100:5.1f}%   "
          f"{metrics['macro_f1']*100:5.1f}%")

    # Interpretación
    print("\n💡 Interpretación:")

    if metrics["accuracy"] >= 0.85:
        print("  ✅ Excelente precisión general (≥85%)")
    elif metrics["accuracy"] >= 0.70:
        print("  ⚠️  Precisión aceptable (70-85%)")
    else:
        print("  ❌ Precisión baja (<70%) - Necesita mejora")

    # Análisis por clase
    for risk_level in ["low", "medium", "high"]:
        m = metrics["by_class"][risk_level]
        if m["f1_score"] < 0.70:
            print(f"  ⚠️  Clase '{risk_level}' tiene F1 bajo ({m['f1_score']*100:.1f}%) - Revisar casos")

    print("\n" + "="*70)


def save_report(evaluation: Dict, filename: str = None):
    """Guardar reporte en JSON con timestamp."""

    if not evaluation:
        return

    import json
    from datetime import datetime
    import os

    # Create reports directory if it doesn't exist
    os.makedirs("reports/evaluations", exist_ok=True)

    # Generate filename with timestamp if not provided
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"reports/evaluations/evaluation_{timestamp}.json"

    report = {
        "date": datetime.now().isoformat(),
        "test_cases_count": len(TEST_CASES),
        "metrics": evaluation["metrics"],
        "confusion_matrix": evaluation["confusion_matrix"],
        "detailed_results": evaluation["results"]
    }

    with open(filename, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n💾 Reporte guardado en: {filename}")


def main():
    """Función principal."""

    print("\n🤖 Sistema de Evaluación de Predicción de Brotes de Crohn")
    print("   Asegúrate de que la API esté corriendo (make serve)\n")

    # Ejecutar evaluación
    evaluation = evaluate_predictions()

    if evaluation:
        # Imprimir reporte
        print_report(evaluation)

        # Guardar reporte
        save_report(evaluation)

        print("\n✨ Evaluación completada!")
        print("\n💡 Nota: Estas métricas corresponden al modelo RandomForest entrenado.")
        print("   Si el modelo no está cargado, el sistema usa predicciones basadas en reglas como fallback.")
        print("   Revisa los logs del servidor para confirmar qué método se está usando.")
    else:
        print("\n❌ No se pudo completar la evaluación")
        print("   Verifica que el servidor esté corriendo con: make serve")


if __name__ == "__main__":
    main()

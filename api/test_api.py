"""
Test script for Crohn Flare Predictor API

This script demonstrates how to interact with the API endpoints.
Run the API first: uvicorn api.app:app --reload

Usage:
    python api/test_api.py
"""

import requests
import json
from pathlib import Path

# API base URL
BASE_URL = "http://localhost:8000"

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "=" * 80)
    print(f"{title}")
    print("=" * 80)


def test_health():
    """Test the health check endpoint"""
    print_section("1. Testing Health Check Endpoint")

    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    return response.status_code == 200


def test_root():
    """Test the root endpoint"""
    print_section("2. Testing Root Endpoint")

    response = requests.get(f"{BASE_URL}/")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    return response.status_code == 200


def test_model_info():
    """Test the model info endpoint"""
    print_section("3. Testing Model Info Endpoint")

    response = requests.get(f"{BASE_URL}/model/info")
    print(f"Status Code: {response.status_code}")
    data = response.json()

    # Print key info
    print(f"\nModel Type: {data['model_type']}")
    print(f"Model Name: {data['model_name']}")
    print(f"Number of Features: {data['n_features']}")
    print(f"Optimal Threshold (F2): {data['optimal_threshold_f2']:.4f}")
    print(f"Test Recall: {data['test_recall']:.4f}")
    print(f"Test Precision: {data['test_precision']:.4f}")
    print(f"Test F2-Score: {data['test_f2']:.4f}")
    print(f"Date Created: {data['date_created']}")

    return response.status_code == 200


def test_single_prediction(example_name="no_flare_example"):
    """Test single prediction endpoint"""
    print_section(f"4. Testing Single Prediction - {example_name}")

    # Load example data
    examples_path = Path(__file__).parent / "example_requests.json"
    with open(examples_path, 'r') as f:
        examples = json.load(f)

    # Get the specific example
    example_data = examples['examples'][example_name]

    # Create request payload
    payload = {
        "features": example_data['features'],
        "user_id": "test_user_001"
    }

    print(f"\nExample: {example_data['description']}")
    print(f"Key features:")
    print(f"  - Severity mean: {example_data['features']['severity_mean']:.2f}")
    print(f"  - Severity max: {example_data['features']['severity_max']:.2f}")
    print(f"  - Abdominal pain: {example_data['features']['abdominal_pain']:.2f}")
    print(f"  - Diarrhea: {example_data['features']['diarrhea']:.2f}")
    print(f"  - Severity change %: {example_data['features']['severity_change_pct']:.2f}")

    # Make request
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    print(f"\nStatus Code: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print(f"\nPrediction Results:")
        print(f"  - Is Flare: {result['is_flare']}")
        print(f"  - Flare Probability: {result['flare_probability']:.4f}")
        print(f"  - Risk Level: {result['risk_level']}")
        print(f"  - Confidence: {result['confidence']}")
        print(f"  - Threshold Used: {result['threshold_used']:.4f}")
        print(f"  - Recommendation: {result['recommendation']}")
    else:
        print(f"Error: {response.text}")

    return response.status_code == 200


def test_batch_prediction():
    """Test batch prediction endpoint"""
    print_section("5. Testing Batch Prediction")

    # Load example data
    examples_path = Path(__file__).parent / "example_requests.json"
    with open(examples_path, 'r') as f:
        examples = json.load(f)

    # Get batch example
    payload = examples['batch_example']

    print(f"\nBatch size: {len(payload['predictions'])} predictions")

    # Make request
    response = requests.post(f"{BASE_URL}/predict/batch", json=payload)
    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print(f"\nBatch Results:")
        print(f"  - Total Predictions: {result['total_predictions']}")
        print(f"  - Flare Count: {result['flare_count']}")
        print(f"  - Average Flare Probability: {result['average_flare_probability']:.4f}")

        print(f"\nIndividual Predictions:")
        for i, pred in enumerate(result['predictions']):
            print(f"  Prediction {i+1}:")
            print(f"    - Is Flare: {pred['is_flare']}")
            print(f"    - Probability: {pred['flare_probability']:.4f}")
            print(f"    - Risk Level: {pred['risk_level']}")
    else:
        print(f"Error: {response.text}")

    return response.status_code == 200


def test_custom_threshold():
    """Test prediction with custom threshold"""
    print_section("6. Testing Custom Threshold")

    # Load example data
    examples_path = Path(__file__).parent / "example_requests.json"
    with open(examples_path, 'r') as f:
        examples = json.load(f)

    example_data = examples['examples']['moderate_risk_example']

    # Test with different thresholds
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]

    for threshold in thresholds:
        payload = {
            "features": example_data['features'],
            "user_id": "test_user_threshold",
            "threshold": threshold
        }

        response = requests.post(f"{BASE_URL}/predict", json=payload)

        if response.status_code == 200:
            result = response.json()
            print(f"\nThreshold: {threshold:.1f}")
            print(f"  - Is Flare: {result['is_flare']}")
            print(f"  - Probability: {result['flare_probability']:.4f}")
            print(f"  - Risk Level: {result['risk_level']}")

    return True


def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("CROHN FLARE PREDICTOR API - TEST SUITE")
    print("=" * 80)
    print("\nMake sure the API is running: uvicorn api.app:app --reload")
    print("API Base URL:", BASE_URL)

    try:
        # Run tests
        results = {
            "Health Check": test_health(),
            "Root Endpoint": test_root(),
            "Model Info": test_model_info(),
            "Single Prediction (No Flare)": test_single_prediction("no_flare_example"),
            "Single Prediction (Flare)": test_single_prediction("flare_example"),
            "Single Prediction (Moderate)": test_single_prediction("moderate_risk_example"),
            "Batch Prediction": test_batch_prediction(),
            "Custom Threshold": test_custom_threshold()
        }

        # Print summary
        print_section("TEST SUMMARY")
        passed = sum(results.values())
        total = len(results)

        for test_name, result in results.items():
            status = "✓ PASSED" if result else "✗ FAILED"
            print(f"{status}: {test_name}")

        print(f"\nTotal: {passed}/{total} tests passed")

        if passed == total:
            print("\n🎉 All tests passed!")
        else:
            print(f"\n⚠️  {total - passed} test(s) failed")

    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Cannot connect to API")
        print("Make sure the API is running:")
        print("  uvicorn api.app:app --reload")
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")


if __name__ == "__main__":
    main()

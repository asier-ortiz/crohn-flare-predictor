# Crohn Flare Predictor API

FastAPI-based REST API for predicting inflammatory bowel disease (IBD) flares using machine learning.

## Quick Start

### 1. Install Dependencies

```bash
# Make sure you're in the project root directory
pip install fastapi uvicorn pydantic
```

Or install all dependencies:

```bash
pip install -r requirements.txt
```

### 2. Start the API

```bash
# From project root directory
uvicorn api.app:app --reload --port 8000
```

The API will be available at: http://localhost:8000

### 3. Access Interactive Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### 4. Run Tests

```bash
# In another terminal (while API is running)
python api/test_api.py
```

## API Endpoints

### Health & Info

#### `GET /`
Welcome message and API overview

**Response:**
```json
{
  "name": "Crohn Flare Predictor API",
  "version": "1.0.0",
  "status": "ready",
  "endpoints": {...}
}
```

#### `GET /health`
Health check and model status

**Response:**
```json
{
  "status": "healthy",
  "message": "API is running and model is loaded",
  "timestamp": "2025-10-13T12:00:00",
  "model_loaded": true,
  "model_type": "XGBoost"
}
```

#### `GET /model/info`
Get model metadata and performance metrics

**Response:**
```json
{
  "model_type": "XGBoost",
  "model_name": "XGBoost (threshold tuned F2)",
  "n_features": 43,
  "optimal_threshold_f2": 0.1024,
  "test_recall": 1.0,
  "test_precision": 0.9977,
  "test_f2": 0.9995,
  "test_pr_auc": 0.9999,
  "date_created": "2025-10-13 11:48:31",
  "features": [...]
}
```

### Predictions

#### `POST /predict`
Single prediction from symptom features

**Request Body:**
```json
{
  "features": {
    "severity_mean": 1.5,
    "severity_max": 3.0,
    "severity_sum": 18.0,
    "symptom_count": 12,
    "abdominal_pain": 2.0,
    "blood_in_stool": 0.0,
    "diarrhea": 3.0,
    "fatigue": 2.0,
    // ... (43 features total)
  },
  "user_id": "optional_user_id",
  "threshold": 0.5  // optional custom threshold
}
```

**Response:**
```json
{
  "is_flare": true,
  "flare_probability": 0.8523,
  "confidence": "High",
  "threshold_used": 0.1024,
  "risk_level": "Critical",
  "recommendation": "URGENT: High risk of flare detected...",
  "timestamp": "2025-10-13T12:00:00"
}
```

#### `POST /predict/batch`
Batch predictions (up to 1000 at once)

**Request Body:**
```json
{
  "predictions": [
    { /* features object 1 */ },
    { /* features object 2 */ }
  ],
  "threshold": 0.5  // optional
}
```

**Response:**
```json
{
  "predictions": [
    { /* prediction result 1 */ },
    { /* prediction result 2 */ }
  ],
  "total_predictions": 2,
  "flare_count": 1,
  "average_flare_probability": 0.4523
}
```

## Feature Requirements

The API expects **43 features** in the following order:

### Core Symptom Aggregates
- `severity_mean` (0-4): Mean symptom severity
- `severity_max` (0-4): Maximum symptom severity
- `severity_sum` (≥0): Sum of all symptom severities
- `symptom_count` (≥1): Number of symptoms tracked

### Individual Symptom Severities (0-4)
- `abdominal_pain`
- `blood_in_stool`
- `diarrhea`
- `fatigue`
- `fever`
- `joint_pain`
- `nausea`
- `other`
- `weight_loss`

### Baseline and Change
- `severity_baseline_7d` (0-4): 7-day baseline severity
- `severity_change_pct`: Percentage change from baseline

### Rolling Window Features
**3-day windows:**
- `severity_mean_3d`, `severity_max_3d`, `symptom_count_3d`

**7-day windows:**
- `severity_mean_7d`, `severity_max_7d`, `symptom_count_7d`

**14-day windows:**
- `severity_mean_14d`, `severity_max_14d`, `symptom_count_14d`

### Trend Features
- `severity_trend_3d`: 3-day severity trend (slope)
- `severity_std_3d`: 3-day std deviation
- `severity_trend_7d`: 7-day severity trend
- `severity_std_7d`: 7-day std deviation

### Symptom-Specific Rolling (7-day)
- `diarrhea_7d`
- `abdominal_pain_7d`
- `blood_in_stool_7d`
- `fatigue_7d`

### Temporal Features
- `day_of_week` (0-6): 0=Monday, 6=Sunday
- `day_of_month` (1-31)
- `month` (1-12)
- `is_weekend` (0 or 1)
- `days_since_start` (≥1): Days since patient started tracking

### Lag Features
- `severity_mean_lag1`, `is_flare_lag1`: Previous day
- `severity_mean_lag2`, `is_flare_lag2`: 2 days ago
- `severity_mean_lag3`, `is_flare_lag3`: 3 days ago

## Example Usage

### Python (requests library)

```python
import requests
import json

# Load example data
with open('api/example_requests.json', 'r') as f:
    examples = json.load(f)

# Get example features
features = examples['examples']['flare_example']['features']

# Make prediction
response = requests.post(
    'http://localhost:8000/predict',
    json={
        'features': features,
        'user_id': 'patient_123'
    }
)

result = response.json()
print(f"Flare Risk: {result['risk_level']}")
print(f"Probability: {result['flare_probability']:.2%}")
print(f"Recommendation: {result['recommendation']}")
```

### cURL

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d @api/example_requests.json
```

### JavaScript (fetch)

```javascript
const response = await fetch('http://localhost:8000/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    features: { /* 43 features */ },
    user_id: 'patient_123'
  })
});

const result = await response.json();
console.log(`Risk: ${result.risk_level}`);
```

## Risk Level Classifications

| Probability | Risk Level | Recommendation |
|-------------|------------|----------------|
| < 0.2 | **Low** | Continue current treatment plan |
| 0.2 - 0.5 | **Medium** | Moderate risk, monitor symptoms |
| 0.5 - 0.8 | **High** | High risk, consider contacting provider |
| ≥ 0.8 | **Critical** | URGENT: Contact healthcare provider |

## Confidence Levels

Based on distance from classification threshold:

- **Low**: Prediction is close to threshold (±0.1)
- **Medium**: Moderate distance from threshold (0.1-0.3)
- **High**: Far from threshold (>0.3)

## Model Information

- **Algorithm**: XGBoost (Gradient Boosting)
- **Optimal Threshold**: 0.1024 (F2-optimized for high recall)
- **Test Performance**:
  - Recall: 100% (detects all flares)
  - Precision: 99.77%
  - F2-Score: 0.9996
- **Training Data**: 10,686 daily records from 325 Crohn's/IBD patients
- **Date Created**: October 2025

## Development

### Project Structure

```
api/
├── app.py                    # Main FastAPI application
├── schemas.py                # Pydantic request/response models
├── example_requests.json     # Example payloads for testing
├── test_api.py              # Test script
└── README.md                # This file
```

### Running in Production

For production deployment, use a production ASGI server:

```bash
# Install production server
pip install gunicorn

# Run with gunicorn (4 workers)
gunicorn api.app:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Docker Deployment (Optional)

Create a `Dockerfile`:

```dockerfile
FROM python:3.13-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:

```bash
docker build -t crohn-flare-api .
docker run -p 8000:8000 crohn-flare-api
```

## Troubleshooting

### Model Not Loaded Error

**Error**: `503 Service Unavailable - Model not loaded`

**Solution**: Ensure the trained model files exist:
- `models/xgboost_flare_predictor.pkl`
- `models/scaler.pkl`
- `models/model_metadata.json`

Run the training notebook first: `notebooks/04_model_training.ipynb`

### Import Errors

**Error**: `ModuleNotFoundError: No module named 'fastapi'`

**Solution**: Install dependencies:
```bash
pip install fastapi uvicorn pydantic
```

### Port Already in Use

**Error**: `Address already in use`

**Solution**: Use a different port:
```bash
uvicorn api.app:app --reload --port 8001
```

## License

See project root LICENSE file.

## Support

For issues and questions:
- Check `/docs` endpoint for interactive API documentation
- Review `example_requests.json` for valid request formats
- Run `test_api.py` to verify setup

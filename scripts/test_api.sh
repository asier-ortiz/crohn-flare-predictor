#!/bin/bash

# Script para probar los endpoints de la API con curl
# Asegúrate de que el servidor está corriendo: make serve

BASE_URL="http://localhost:8000"
EXAMPLES_FILE="scripts/api_examples.json"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}  Crohn Flare Predictor API Tests${NC}"
echo -e "${BLUE}======================================${NC}\n"

# 1. Health Check
echo -e "${GREEN}1. Health Check${NC}"
echo -e "${YELLOW}GET /health${NC}"
curl -s -X GET "$BASE_URL/health" | jq '.'
echo -e "\n"

# 2. Model Info
echo -e "${GREEN}2. Model Information${NC}"
echo -e "${YELLOW}GET /model/info${NC}"
curl -s -X GET "$BASE_URL/model/info" | jq '.'
echo -e "\n"

# 3. Predict - Low Risk
echo -e "${GREEN}3. Predicción Individual - Bajo Riesgo${NC}"
echo -e "${YELLOW}POST /predict${NC}"
curl -s -X POST "$BASE_URL/predict" \
  -H "Content-Type: application/json" \
  -d @- <<'EOF' | jq '.'
{
  "symptoms": {
    "abdominal_pain": 2,
    "diarrhea": 3,
    "fatigue": 2,
    "fever": false,
    "weight_change": 0.0,
    "blood_in_stool": false,
    "nausea": 1
  },
  "demographics": {
    "age": 28,
    "gender": "F",
    "disease_duration_years": 2,
    "bmi": 22.0
  },
  "history": {
    "previous_flares": 1,
    "medications": ["mesalamine"],
    "last_flare_days_ago": 365,
    "surgery_history": false,
    "smoking_status": "never"
  }
}
EOF
echo -e "\n"

# 4. Predict - High Risk
echo -e "${GREEN}4. Predicción Individual - Alto Riesgo${NC}"
echo -e "${YELLOW}POST /predict${NC}"
curl -s -X POST "$BASE_URL/predict" \
  -H "Content-Type: application/json" \
  -d @- <<'EOF' | jq '.'
{
  "symptoms": {
    "abdominal_pain": 9,
    "diarrhea": 8,
    "fatigue": 7,
    "fever": true,
    "weight_change": -5.0,
    "blood_in_stool": true,
    "nausea": 7
  },
  "demographics": {
    "age": 45,
    "gender": "M",
    "disease_duration_years": 10,
    "bmi": 19.5
  },
  "history": {
    "previous_flares": 6,
    "medications": ["infliximab", "azathioprine"],
    "last_flare_days_ago": 45,
    "surgery_history": true,
    "smoking_status": "former"
  }
}
EOF
echo -e "\n"

# 5. Batch Prediction
echo -e "${GREEN}5. Predicción por Lotes${NC}"
echo -e "${YELLOW}POST /predict/batch${NC}"
curl -s -X POST "$BASE_URL/predict/batch" \
  -H "Content-Type: application/json" \
  -d @- <<'EOF' | jq '.'
{
  "patients": [
    {
      "patient_id": "P001",
      "symptoms": {
        "abdominal_pain": 3,
        "diarrhea": 2,
        "fatigue": 4,
        "fever": false,
        "weight_change": 0.5,
        "blood_in_stool": false,
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
        "fever": true,
        "weight_change": -3.0,
        "blood_in_stool": true,
        "nausea": 5
      },
      "demographics": {
        "age": 35,
        "gender": "M",
        "disease_duration_years": 8
      },
      "history": {
        "previous_flares": 4,
        "medications": ["adalimumab"],
        "last_flare_days_ago": 60
      }
    }
  ]
}
EOF
echo -e "\n"

echo -e "${BLUE}======================================${NC}"
echo -e "${GREEN}✅ Tests completados${NC}"
echo -e "${BLUE}======================================${NC}"

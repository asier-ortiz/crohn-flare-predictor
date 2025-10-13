# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Crohn Flare Predictor** is a machine learning system for predicting inflammatory bowel disease (IBD) flares based on daily symptom tracking data from the Flaredown dataset.

The project uses time-series data from 2,046 users with Crohn's disease/IBD, containing ~390k symptom, treatment, and condition records spanning 2015-2019.

## Development Environment

### Setup
```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Unix/macOS)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Python Version
Compatible with Python 3.8+, tested with Python 3.13.

## Common Commands

### Running Jupyter Notebooks
```bash
# Start Jupyter
jupyter notebook

# Or use Jupyter Lab
jupyter lab
```

### Testing
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_model.py

# Run with verbose output
pytest -v
```

### Code Quality
```bash
# Format code with black
black src/ api/ scripts/ tests/

# Lint with flake8
flake8 src/ api/ scripts/ tests/
```

### Data Preparation
```bash
# Create a sample dataset (useful for development/testing)
python scripts/create_sample.py
```

## Project Architecture

### Data Flow
1. **Raw data** (`data/raw/export.csv`) - Original Flaredown dataset (~600MB, 7.9M records)
2. **Filtered data** (`data/processed/crohn_filtered.csv`) - Users with Crohn/IBD only (~390k records)
3. **Sample data** (`data/processed/crohn_sample_10k.csv`) - 10k sample for quick testing
4. **Processed features** - Engineered time-series features for model training
5. **Trained models** - Saved in `models/` directory

### Module Structure

**`src/`** - Core ML pipeline modules:
- `preprocessing.py` - Data cleaning and transformation
- `feature_engineering.py` - Time-series feature creation
- `model.py` - Model training and evaluation logic
- `utils.py` - Shared utilities

**`api/`** - REST API for model deployment (FastAPI):
- `app.py` - API endpoints
- `schemas.py` - Pydantic request/response schemas

**`notebooks/`** - Analysis and experimentation:
- `01_exploratory_analysis.ipynb` - Dataset exploration, filtering Crohn users
- `02_feature_engineering.ipynb` - Feature creation (placeholder)
- `03_model_training.ipynb` - Model development (placeholder)

**`scripts/`** - Utility scripts:
- `create_sample.py` - Creates manageable dataset samples for development

**`tests/`** - Unit tests

### Data Schema

The Flaredown dataset uses a long-format structure:

| Column | Type | Description |
|--------|------|-------------|
| `user_id` | string | Anonymized user identifier |
| `age` | float | User age |
| `sex` | string | User sex |
| `country` | string | Country code |
| `checkin_date` | date | Date of check-in |
| `trackable_type` | string | Type: Symptom, Treatment, Condition, Weather, Food, Tag, HBI |
| `trackable_name` | string | Name of the tracked item |
| `trackable_value` | mixed | Severity (0-4 for symptoms) or value (treatment dose, weather condition, etc.) |

**Key Insight**: Multiple trackables per user per date creates a sparse, multi-variate time series.

### Workflow

1. **Data Filtering** (notebook 01):
   - Identify users with Crohn's disease/IBD/colitis conditions
   - Extract all their symptom/treatment/condition records
   - Result: 2,046 users, 513 with ≥30 days of data suitable for time-series modeling

2. **Feature Engineering** (notebook 02, in progress):
   - Pivot long-format data to wide format (one row per user-date)
   - Create rolling window features (3-day, 7-day symptom trends)
   - Handle missing values and irregular sampling
   - Define target variable (flare vs. non-flare)

3. **Model Training** (notebook 03, in progress):
   - Planned approach: LSTM for time-series prediction
   - Alternative: Random Forest/XGBoost for baseline
   - Evaluation: Precision/recall for flare prediction

## Key Technical Considerations

### Dataset Characteristics
- **Sparse data**: Not all users track daily; expect gaps
- **Heterogeneous trackables**: ~3600 unique symptoms, many user-created and not standardized
- **Mixed data types**: Numeric severity (0-4), categorical (yes/no), free-text doses
- **Imbalanced**: "Flare" events are minority class

### Memory Management
The full dataset is ~600MB. For development:
- Use `crohn_sample_10k.csv` (10k records) for quick iteration
- Use `crohn_filtered.csv` (390k records) for full training
- Consider chunked processing with `pandas.read_csv(chunksize=...)`

### Model Architecture Decision
- **LSTM preferred**: 513 users with ≥30 days of sequential data
- **Feature engineering critical**: Need to pivot and create temporal features
- **Target definition**: Must define "flare" (e.g., symptom severity spike, HBI score threshold)

## File Conventions

### Large Files (excluded from git)
- `data/raw/*.csv` - Raw datasets
- `models/*.pkl`, `models/*.joblib` - Trained model files
- `data/processed/cache/` - Intermediate processing artifacts

### Included Files
- `data/processed/sample_*.csv` - Small samples for testing

## Notes for Future Development

- **API deployment**: FastAPI files exist but are placeholder (1 line each). Uncomment FastAPI in requirements.txt when building API.
- **Deep learning**: TensorFlow/PyTorch dependencies commented in requirements.txt. Uncomment for LSTM development.
- **HBI scores**: Only 157 HBI (Harvey-Bradshaw Index) records exist. May not be sufficient for medical-grade validation.
- **Symptom standardization**: Consider mapping similar symptoms (e.g., "Diarrhea" vs "diarrhea" vs "loose stools") before modeling.

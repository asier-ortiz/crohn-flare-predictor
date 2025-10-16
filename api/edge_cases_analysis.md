# Edge Cases & Model Analysis

This document tracks interesting edge cases and potential model failures discovered during API testing.

## Purpose

Identify scenarios where the model might make incorrect predictions to guide future model improvements and deep analysis.

---

## Case 1: Post-Flare Recovery Misclassification

**Date**: 2025-10-16
**Case ID**: `edge_case_001`
**Status**: ⚠️ Potential False Positive

### Description

Patient showing clear improvement trend (-33% severity reduction) after recent flares, but model predicts critical flare risk with 99.88% confidence.

### Request Payload

```json
{
  "features": {
    "severity_mean": 1.2,
    "severity_max": 3.0,
    "severity_sum": 15.0,
    "symptom_count": 13,
    "abdominal_pain": 3.0,
    "blood_in_stool": 0.0,
    "diarrhea": 2.0,
    "fatigue": 3.0,
    "fever": 0.0,
    "joint_pain": 1.0,
    "nausea": 0.0,
    "other": 1.0,
    "weight_loss": 0.0,
    "severity_baseline_7d": 1.8,
    "severity_change_pct": -33.33,
    "severity_mean_3d": 1.4,
    "severity_max_3d": 3.0,
    "symptom_count_3d": 35.0,
    "severity_mean_7d": 1.7,
    "severity_max_7d": 4.0,
    "symptom_count_7d": 80.0,
    "severity_mean_14d": 1.8,
    "severity_max_14d": 4.0,
    "symptom_count_14d": 160.0,
    "severity_trend_3d": -0.3,
    "severity_std_3d": 0.4,
    "severity_trend_7d": -0.2,
    "severity_std_7d": 0.5,
    "diarrhea_7d": 2.5,
    "abdominal_pain_7d": 3.2,
    "blood_in_stool_7d": 0.0,
    "fatigue_7d": 2.8,
    "day_of_week": 5,
    "day_of_month": 20,
    "month": 3,
    "is_weekend": 1,
    "days_since_start": 200,
    "severity_mean_lag1": 1.4,
    "is_flare_lag1": 0.0,
    "severity_mean_lag2": 1.6,
    "is_flare_lag2": 1.0,
    "severity_mean_lag3": 1.9,
    "is_flare_lag3": 1.0
  },
  "user_id": "edge_case_001"
}
```

### Model Response

```json
{
  "is_flare": true,
  "flare_probability": 0.9987953901290894,
  "confidence": "High",
  "threshold_used": 0.1023726686835289,
  "risk_level": "Critical",
  "recommendation": "URGENT: High risk of flare detected. Contact your healthcare provider immediately. Monitor symptoms closely.",
  "timestamp": "2025-10-16T10:02:20.721854"
}
```

### Key Observations

#### Contradictory Signals

**Supporting FLARE prediction:**
- ✅ Recent flare history (2 flares in last 3 days: lag2=1, lag3=1)
- ✅ Elevated absolute symptoms:
  - Abdominal pain: 3/4
  - Diarrhea: 2/4
  - Fatigue: 3/4
- ✅ High historical averages:
  - 7-day severity: 1.7
  - 14-day severity: 1.8
  - 7-day symptom count: 80

**Against FLARE prediction:**
- ❌ **Strong improvement trend**: -33.33% severity change from baseline
- ❌ **Negative trends**:
  - 3-day trend: -0.3 (decreasing)
  - 7-day trend: -0.2 (decreasing)
- ❌ **Current severity below baseline**: 1.2 < 1.8
- ❌ **Yesterday no flare**: is_flare_lag1 = 0 (recovery started)

### Analysis

#### What the Model Got Wrong

The model appears to have:

1. **Over-weighted temporal persistence**: "If there were flares 2-3 days ago, there's still a flare"
2. **Ignored improvement trajectory**: -33% change should be a strong negative signal
3. **Prioritized absolute values over trends**: High symptoms but declining
4. **Over-confidence**: 99.88% probability seems extreme given conflicting signals

#### Clinical Context

This represents a **"post-flare recovery"** scenario:
- Patient had a flare episode 2-3 days ago
- Symptoms are still elevated but improving
- Clear downward trajectory
- Unlikely to be in active flare state

**True label**: Probably **NO FLARE** (recovery phase)
**Model prediction**: **FLARE** (99.88% confidence)
**Result**: Likely **False Positive**

### Model Biases Identified

1. **Temporal Persistence Bias**:
   - Model assumes flares persist longer than they do
   - Doesn't recognize recovery patterns quickly enough

2. **Absolute Value Dominance**:
   - Gives more weight to current symptom levels than change direction
   - Trend features (severity_trend_3d, severity_trend_7d) appear underutilized

3. **Conservative (High Recall) Bias**:
   - Model trained to maximize recall (don't miss flares)
   - Results in false alarms during recovery periods
   - Medically safer but reduces precision

### Implications for Model Improvement

#### Recommended Investigations (Notebook 05)

1. **Feature Importance Analysis**:
   - Which features dominated this decision?
   - Are lag features (is_flare_lag2, is_flare_lag3) too influential?
   - How much weight do trend features have?

2. **SHAP Value Analysis**:
   - What pushed probability from ~0% to 99.88%?
   - Visualize feature contributions for this specific case
   - Compare with correctly classified recovery cases

3. **Error Pattern Analysis**:
   - Find similar "recovery" cases in test set
   - Calculate false positive rate specifically for post-flare periods
   - Identify if this is a systematic issue

4. **Probability Calibration**:
   - Is 99.88% realistic for this scenario?
   - Check calibration curves
   - Model might be over-confident in general

5. **Threshold Optimization for Recovery Cases**:
   - Current threshold: 0.102 (optimized for overall F2)
   - Consider dynamic thresholds based on trend direction
   - Or add a "recovery detection" layer

#### Potential Model Enhancements

**New Features to Consider**:
```python
# Second-order trends
severity_acceleration_3d  # Rate of change of change
days_in_recovery         # Days since severity started declining

# Directional change features
trend_direction_change   # Did trend flip from positive to negative?
recovery_velocity        # Speed of improvement

# Contextual features
days_since_last_flare    # Explicit gap measurement
consecutive_improving_days  # Streak of improvement
```

**Model Architectures**:
- **LSTM/RNN**: Better at recognizing temporal patterns and reversals
- **Ensemble with recovery detector**: Separate model to identify recovery phases
- **Multi-task learning**: Jointly predict flare + recovery state

**Training Data Augmentation**:
- Ensure training set has enough "recovery" examples
- May be class imbalance within the "no flare" class
- Label engineering: "recovery", "stable", "worsening", "flare"

### Next Steps

- [ ] Add this case to test suite for regression testing
- [ ] Create notebook cell to analyze similar recovery patterns
- [ ] Run SHAP analysis on this specific prediction
- [ ] Check if model performs differently with custom threshold
- [ ] Collect more edge cases for comprehensive analysis
- [ ] Consider adding "recovery" as an intermediate class

---

## Case 2: [Future cases will be added here]

**Date**: TBD
**Case ID**: `edge_case_002`
**Status**: TBD

---

## Testing Edge Cases

To test these cases via API:

```bash
# Single prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d @api/edge_cases_analysis.md  # Extract JSON from this file

# With custom threshold
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": {...},
    "threshold": 0.5  # Try different thresholds
  }'
```

Or use the Python test script:

```python
import requests
import json

# Load this case
case = {
    "features": { ... },  # Copy from above
    "user_id": "edge_case_001"
}

# Test with different thresholds
for threshold in [0.1, 0.3, 0.5, 0.7]:
    case["threshold"] = threshold
    response = requests.post("http://localhost:8000/predict", json=case)
    result = response.json()
    print(f"Threshold {threshold}: is_flare={result['is_flare']}, "
          f"prob={result['flare_probability']:.4f}")
```

---

## References

- Model training notebook: `notebooks/04_model_training.ipynb`
- Model metadata: `models/model_metadata.json`
- API documentation: `api/README.md`

---

**Last Updated**: 2025-10-16
**Contributors**: Model Analysis Team

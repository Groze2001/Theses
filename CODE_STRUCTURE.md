# Code Pipeline Overview

## Script Organization

All code scripts follow a **numbered prefix** convention with **descriptive names**:

### Data Pipeline Sequence

```
01_data_exploration_seasonality.py
        ↓
02_model_training_bootstrap_validation.py
        ↓
03_feature_importance_analysis.py
        ├→ 04_model_validation_nbeats.py
        │
        ├→ 05_anomaly_rate_generalization_analysis.py
        │
        └→ 06_anomaly_detection_results_ensemble.py
```

---

## Script Details

| # | Script Name | Purpose | Input | Output |
|---|---|---|---|---|
| **01** | `data_exploration_seasonality.py` | STL decomposition, seasonal analysis by sector/period | Smart meter data (daily aggregate) | Seasonality figures, strength metrics |
| **02** | `model_training_bootstrap_validation.py` | Train top 4 models, bootstrap validation (100 seeds) | Training/validation/test splits | Bootstrap results, trimmed-mean accuracy |
| **03** | `feature_importance_analysis.py` | Permutation importance for RF/Ridge/N-BEATS | Test set predictions, features | Feature importance rankings |
| **04** | `model_validation_nbeats.py` | N-BEATS v2 diagnostic validation | Model predictions, residuals | Validation metrics, diagnostics |
| **05** | `anomaly_rate_generalization_analysis.py` | Compare train-vs-test anomaly rates, assess generalization | Training/test period anomalies | Generalization comparison table |
| **06** | `anomaly_detection_results_ensemble.py` | Compare 5 detectors, majority-vote ensemble, analysis | N-BEATS residuals, feature matrix | Anomaly results, flagged dates |

---

## Running the Pipeline

```bash
cd code/

# 1. Explore data and seasonality
python 01_data_exploration_seasonality.py

# 2. Train models with bootstrap validation
python 02_model_training_bootstrap_validation.py

# 3. Analyze feature importance
python 03_feature_importance_analysis.py

# 4. Validate N-BEATS model
python 04_model_validation_nbeats.py

# 5. Check generalization (train vs test)
python 05_anomaly_rate_generalization_analysis.py

# 6. Ensemble anomaly detection analysis
python 06_anomaly_detection_results_ensemble.py
```

---

## Key Files & Directories

```
theses_text/
├── Theses.tex                    ← Main thesis document (84 pages)
├── Theses.pdf                    ← Compiled PDF
├── bib/
│   └── refs.bib                  ← Bibliography (29 citations)
└── images/                       ← 17 figures and diagrams

code/
├── README.md                     ← Detailed documentation
├── 01_data_exploration_seasonality.py
├── 02_model_training_bootstrap_validation.py
├── 03_feature_importance_analysis.py
├── 04_model_validation_nbeats.py
├── 05_anomaly_rate_generalization_analysis.py
└── 06_anomaly_detection_results_ensemble.py

results/                          ← Generated outputs
├── seasonality/
├── forecasting/
├── feature_importance/
└── anomaly/
```

---

## Naming Convention

**Format:** `{NUMBER}_{VERB}_{OBJECT}.py`

- **NUMBER**: 01-06 (execution order)
- **VERB**: Action performed (data_exploration, model_training, feature_importance, etc.)
- **OBJECT**: What the script operates on (seasonality, bootstrap_validation, anomaly, etc.)

**Examples:**
- ✓ `01_data_exploration_seasonality.py` (clear: explores data seasonality)
- ✓ `02_model_training_bootstrap_validation.py` (clear: trains models with bootstrap)
- ✓ `03_feature_importance_analysis.py` (clear: analyzes feature importance)

---

## Quick Start

For the complete pipeline analysis:

```bash
# Run all scripts in sequence
for script in 0*.py; do
    echo "Running $script..."
    python "$script"
done

# Results will be saved to results/ directory
```

---

*Last Updated: August 2026*

# Code Pipeline: Pattern Detection in Energy Consumption

This directory contains the analysis scripts for the thesis on pattern detection and anomaly identification in smart meter energy consumption data.

## Script Execution Order

### 1. Data Exploration & Visualization
**`01_data_exploration_seasonality.py`**
- Performs STL decomposition (Seasonal and Trend decomposition using Loess)
- Generates seasonality distribution figures across COVID periods (pre-, in-, post-)
- Analyzes seasonal strength ($F_s$) by economic sector and tariff configuration
- Outputs: Seasonality and trend distribution visualizations

### 2. Model Training & Bootstrap Validation
**`02_model_training_bootstrap_validation.py`**
- Trains top 4 forecasting models (N-BEATS v2, Random Forest, Ridge, LSTM v2)
- Runs 100-seed bootstrap validation to assess model stability
- Computes trimmed-mean accuracy and standard deviation across seeds
- Excludes leakage-based features from Ridge and Random Forest
- Outputs: Bootstrap results table, cross-city average MAPE rankings

### 3. Feature Importance Analysis
**`03_feature_importance_analysis.py`**
- Computes permutation feature importance for top 3 models
- Measures impact of each feature on test-set MAPE
- Identifies feature dependencies: lag-based (RF, Ridge) vs. cyclical (N-BEATS v2)
- Outputs: Permutation importance rankings, ablation study results

### 4. Model Validation (N-BEATS)
**`04_model_validation_nbeats.py`**
- Validates N-BEATS v2 on validation and test periods
- Verifies model stability across random seeds
- Assesses hyperparameter sensitivity
- Outputs: Validation metrics, performance diagnostics

### 5. Anomaly Rate Generalization Analysis
**`05_anomaly_rate_generalization_analysis.py`**
- Compares anomaly rates between training period and test period
- Assesses whether ensemble generalizes to unseen data
- Analyzes distribution of flagged days (random vs. event-aligned)
- Outputs: Train-vs-test anomaly rate comparison table

### 6. Anomaly Detection Results & Ensemble Comparison
**`06_anomaly_detection_results_ensemble.py`**
- Compares 5 independent anomaly detectors (rolling z-score, Isolation Forest, LOF, One-Class SVM, K-Means)
- Evaluates majority-vote ensemble ($\geq 3/5$ threshold)
- Produces anomaly rate breakdown by municipality and detection method
- Identifies and analyzes flagged days (December 2021 cluster, etc.)
- Outputs: Anomaly detection results table, flagged dates list, feature profile analysis

---

## Dataset

- **Source:** GoiEner cooperative, Spain (2014–2022)
- **Scope:** 3 municipalities (Vitoria-Gasteiz, Donostia/San Sebastián, Pamplona/Iruña)
- **Resolution:** Daily municipal-aggregate consumption (kWh)
- **Preprocessing:** Merged imputed series with ≤5% missing observations
- **Retained Users:** 19,709 out of 25,559

---

## Key Parameters

### Data Split
- **Train:** 70% (2014-06-30 to 2020-08-31)
- **Validation:** 15% (2020-09-01 to 2021-08-31)
- **Test:** 15% (2021-09-01 to 2022-06-30)

### N-BEATS v2 Architecture
- Lookback window: 30 days
- Residual blocks: 3
- Hidden size: 64
- Dropout: 0.2
- Early stopping: Monitored on validation loss

### Anomaly Detection
- 5-detector ensemble with $\geq 3/5$ majority vote
- Rolling z-score threshold: $|z| > 3$ with adaptive 30-day window
- Contamination rate: Calibrated per-city from empirical residual distribution

---

## Output Structure

Results are saved to the `results/` directory with the following subdirectories:
- `seasonality/` - STL decomposition and seasonal analysis plots
- `forecasting/` - Model comparison and bootstrap validation results
- `feature_importance/` - Permutation importance rankings and ablations
- `anomaly/` - Anomaly detection results, flagged days, and feature profiles

---

## Dependencies

- Python 3.8+
- pandas, numpy, scikit-learn
- statsmodels (STL decomposition)
- tensorflow/keras (N-BEATS implementation)
- matplotlib, seaborn (visualization)

---

## Reproducibility

All scripts use fixed random seeds (0–99 for bootstrap validation) to ensure reproducible results.
Cross-city results are computed as bootstrap-trimmed means (dropping lowest/highest 5% of 100 runs).

---

*Last Updated: August 2026*

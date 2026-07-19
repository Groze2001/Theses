# Pattern Detection in Energy Consumption

Master's thesis — Gonçalo Rosa, ISCTE, MSc Data Science  
Supervisors: PhD Ana Maria de Almeida, PhD Diana Mendes

---

## What this project does

Identifies consumption patterns and anomalies in smart meter data from the GoiEner energy cooperative (Spain, 2014–2022). The pipeline covers data preparation, time-series decomposition, forecasting, and unsupervised anomaly detection on daily municipal aggregate consumption.

**Research questions:**
1. How are smart meter readings structured?
2. How do seasonality, trend, and residual components behave across COVID-19 periods?
3. What consumption archetypes exist across the user population?
4. Which anomaly detection algorithms perform best?

---

## Dataset

**Source:** [GoiEner smart meters dataset](https://doi.org/10.5281/zenodo.6648317)  
**Size:** 19,709 users retained after quality filtering (≤5% missing), hourly kWh  
**Period:** 2014–2022, segmented into pre-COVID, in-COVID (Mar 2020–May 2021), and post-COVID  
**Location:** Basque Country and Navarra, Spain

Raw per-user CSVs live in `Dataset/merged_imp_csv/` (hash-named files).  
Metadata (user, municipality, CNAE sector, tariff) is in `Dataset/filtered_metadata.csv`.

---

## Project structure

```
Theses/
├── Dataset/                        source data (do not modify)
│   ├── merged_imp_csv/             one CSV per user, imputed hourly series
│   └── filtered_metadata.csv       user metadata after quality filtering
│
├── code/                           analysis notebooks (run in order)
│   ├── 1_data_preparation.ipynb    ETL, quality filtering, metadata merge
│   ├── 2_decomposition_analysis.ipynb  STL decomposition, seasonality/trend/residuals
│   ├── 3_municipality_selection.ipynb  aggregate daily series, city selection
│   ├── 4_forecasting_models.ipynb  11 models benchmarked (with 19 engineered features)
│   ├── 4b_no_features_comparison.ipynb  ablation: calendar features only
│   ├── 5_anomaly_detection.ipynb   5-detector ensemble on N-BEATS residuals
│   ├── top4_bootstrap.py           100-seed bootstrap for top 4 models
│   ├── generate_seasonality_figures.py  seasonality plot generation
│   ├── model_descriptions.txt      plain-text model inventory
│   └── archive/                    old drafts and exploratory notebooks
│
├── results/
│   ├── decomposition/              per-period STL outputs (pre/in/post)
│   │   ├── seasonality_results_daily_{pre,in,post}.csv   daily Fs per user
│   │   ├── seasonality_results_weekly_{pre,in,post}.csv  weekly Fs per user
│   │   ├── trend_analysis_results_{pre,in,post}.csv      slope, category per user
│   │   ├── residual_analysis_results_{pre,in,post}.csv   kurtosis, extreme ratio
│   │   └── consumption_archetypes.csv                    k-means cluster assignments (k=3)
│   │
│   ├── forecasting/
│   │   ├── with_features/          main pipeline — 19 features, N-BEATS v2 selected
│   │   ├── no_features/            ablation — calendar features only
│   │   ├── per_user/               discarded per-user normalisation variants
│   │   └── bootstrap/              100-seed runs for top 4 models + summary_trimmed.csv
│   │
│   ├── anomaly/
│   │   ├── improved/               current run — per-city contamination calibration
│   │   └── original/               superseded run — fixed global contamination rate
│   │
│   ├── figures/                    output figures
│   ├── tables/                     output tables
│   ├── data/                       intermediate aggregated CSVs
│   └── models/                     saved model artefacts
│
├── raw_pub/                        raw publication CSVs (do not modify)
├── municipality_temp_batches/      batch processing intermediates
└── theses_text/                    LaTeX source for the written thesis
    ├── Theses.tex                  main document
    ├── bib/refs.bib                bibliography
    └── images/                     figures used in the thesis
```

---

## How to run

Run notebooks in numbered order. Each notebook reads from `Dataset/` and writes outputs to `results/`.

```
1_data_preparation       →  Dataset/filtered_metadata.csv
2_decomposition_analysis →  results/decomposition/
3_municipality_selection →  results/data/municipality_daily_consumption.csv
4_forecasting_models     →  results/forecasting/with_features/
4b_no_features           →  results/forecasting/no_features/
top4_bootstrap.py        →  results/forecasting/bootstrap/
5_anomaly_detection      →  results/anomaly/improved/
```

---

## Key results

| Component | Finding |
|-----------|---------|
| Decomposition | Daily Fs moderate for households (~0.47), near-perfect for public administration (~0.99). COVID in-period increased household regularity. |
| Archetypes | 3 clusters: Type A — high seasonality, growing (10.7%); Type B — mainstream residential (79.7%); Type C — irregular/sporadic (9.6%) |
| Best forecasting model | N-BEATS v2 (adapted, covariate-conditioned) — MAPE 3.7–5.2% across three cities, bootstrap trimmed mean |
| Anomaly detection | Ensemble ≥3/5 votes: 0.7–1.4% anomaly rate. December 2021 cluster consistent with Omicron wave and regional restrictions |
| Focus cities | Vitoria-Gasteiz (above cooperative mean), Donostia/San Sebastian (below mean), Pamplona/Iruna (near mean) |

---

## Dependencies

Python 3.x with: `pandas`, `numpy`, `scikit-learn`, `statsmodels`, `torch`, `matplotlib`, `neuralforecast`  
LaTeX: `pdflatex` or `lualatex` with `biblatex`/`biber`

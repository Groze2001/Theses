# Anomaly Detection: Improvements Reference

Compares `7_anomaly_detection.ipynb` (original) against `7_anomaly_detection_improved.ipynb`.
Results are saved to separate folders so both can be run independently:

| Version | Output folder |
|---------|--------------|
| Original | `anomaly_detection_results/` |
| Improved | `anomaly_detection_results_improved/` |

---

## Improvement 1 — Rolling z-score instead of global z-score

**File/cell:** `cell-load`

**Literature:** Zangrando et al. [12]

**What changed:**
```python
# ORIGINAL
df['z_residual'] = stats.zscore(df['residual'], nan_policy='omit')

# IMPROVED
def rolling_zscore(series, window=30, min_periods=10):
    roll = series.rolling(window=window, min_periods=min_periods, center=True)
    return (series - roll.mean()) / (roll.std() + 1e-8)

df['z_residual'] = rolling_zscore(df['residual'])
```

**Why it matters:**  
The global z-score divides by the standard deviation of the entire test set (~10 months of data).
Seasonal shifts (e.g. higher consumption in winter) inflate the global std, making winter spikes look
less extreme than they are. A 30-day rolling window measures each day against its local neighbourhood,
so a December holiday spike is compared against other December days, not against summer readings.

**Expected effect on results:**  
The z-score threshold (`|z| > 3`) may flag different days — particularly those that are extreme
*relative to their season* but not extreme globally. The rolling z-score feeds the feature matrix
of all ML detectors too (via `z_residual` column in `build_features`).

---

## Improvement 2 — Per-city contamination calibrated from data

**File/cell:** `cell-detectors` → `calibrate_contamination()`

**Literature:** Zangrando et al. [12]

**What changed:**
```python
# ORIGINAL
CONTAMINATION = 0.022   # hard-coded global constant from Ch. 3 EDA

# IMPROVED
def calibrate_contamination(df, fallback=0.022):
    rate = float(np.mean(np.abs(df['z_residual']) > 3))
    return float(np.clip(rate, 0.01, 0.10)) if rate > 0 else fallback
```

**Why it matters:**  
The 0.022 figure came from the Ch. 3 EDA over all 19,679 users for the full 2014–2022 period.
That is a population-level descriptive statistic. The contamination parameter for IForest, LOF,
and OC-SVM should reflect the anomaly rate in *this specific residual series*. Each city's model
residuals may naturally have a different outlier density. Calibrating per city makes the detector
parameters consistent with what is actually observed in the data each model produces.

**Note on clipping:** The function clamps to [0.01, 0.10] because IForest/LOF/OC-SVM become
unstable outside that range (too few or too many forced anomalies). The 0.022 fallback is only
used if the rolling z-score flags zero days above |z|>3.

---

## Improvement 3 — Extended feature matrix

**File/cell:** `cell-detectors` → `build_features()`

**Literature:** Jesmeen et al. [8]

**What changed:**
```python
# ORIGINAL — 5 features
X = pd.DataFrame({
    'residual':     df['residual'],
    'abs_residual': df['residual'].abs(),
    'z_residual':   df['z_residual'],
    'dow':          df['date'].dt.dayofweek,
    'month':        df['date'].dt.month,
})

# IMPROVED — 9 features (+4)
X = pd.DataFrame({
    'residual':        df['residual'],
    'abs_residual':    df['residual'].abs(),
    'z_residual':      df['z_residual'],
    'lag_residual_1d': df['residual'].shift(1).fillna(0),    # NEW
    'lag_residual_7d': df['residual'].shift(7).fillna(0),    # NEW
    'rolling_mean_7d': df['residual'].rolling(7, min_periods=1).mean(),  # NEW
    'rolling_std_7d':  df['residual'].rolling(7, min_periods=1).std().fillna(0),  # NEW
    'dow':             df['date'].dt.dayofweek,
    'month':           df['date'].dt.month,
})
```

**Why it matters:**  
Jesmeen et al. [8] show that LOF specifically benefits from feature extraction that captures
temporal structure. The original feature matrix treats each day as independent — a 0.05 kWh
residual on a day where the previous 7 days were all near zero looks the same as a 0.05 kWh
residual that follows a week of elevated residuals. The lag and rolling features give all
detectors the ability to distinguish these contexts:
- `lag_residual_1d/7d`: is the anomaly isolated or part of a run?
- `rolling_mean_7d`: is the recent local baseline shifted?
- `rolling_std_7d`: is there a volatile period that raises the noise floor locally?

**Expected effect:** IForest and LOF in particular may produce tighter agreement — days that
deviate from both the residual magnitude *and* the recent pattern will score higher.

---

## Improvement 4 — K-Means on full feature matrix X

**File/cell:** `cell-detectors` → K-Means block

**Literature:** Zhang et al. [14]

**What changed:**
```python
# ORIGINAL — 1-D clustering (broken: flags ~30-40% of days)
km = KMeans(n_clusters=2, random_state=42, n_init=10)
labels = km.fit_predict(df[['residual']].values)          # 1-D input
centroid_abs = np.abs(km.cluster_centers_[:, 0])
anomaly_cluster = int(np.argmax(centroid_abs))
res['kmeans_flag'] = (labels == anomaly_cluster).astype(int)

# IMPROVED — multi-dimensional clustering (matches Zhang et al. [14])
km = KMeans(n_clusters=2, random_state=42, n_init=10)
labels = km.fit_predict(X)                                # full 9-D feature matrix
c0_abs = np.abs(df['residual'].values[labels == 0]).mean()
c1_abs = np.abs(df['residual'].values[labels == 1]).mean()
anomaly_cluster = 0 if c0_abs > c1_abs else 1
res['kmeans_flag'] = (labels == anomaly_cluster).astype(int)
```

**Why the original was wrong:**  
With a 1-D input (just the raw residual), K-Means with k=2 simply finds the median of the
residual distribution and labels everything on one side of it as "cluster 1" and everything
on the other as "cluster 2". Since ~half the residuals are positive and ~half negative, the
resulting split flagged 27–41% of days depending on city — clearly not anomaly detection.

**Why the fix works:**  
Zhang et al. [14] use K-Means on a multi-dimensional feature space that includes temporal
and contextual information. With 9 features, K-Means must find a cluster that is jointly
extreme across magnitude, temporal context, and seasonal position. The anomaly cluster is
still identified post-hoc as the one with higher mean |residual|, but the cluster boundary
is now multi-dimensional and meaningful.

**Expected effect:** K-Means anomaly rate should drop from ~30–40% to a range comparable
with IForest and LOF. Its agreement with other methods (Jaccard score) should increase
substantially. This also makes the ensemble vote more balanced.

---

## How to compare results

Run both notebooks end-to-end, then compare the two output folders:

```
anomaly_detection_results/          ← original
anomaly_detection_results_improved/ ← improved
```

Key files to compare:
- `anomaly_rates_summary.csv` — method-by-method anomaly rates per city
- `anomaly_rates_pivot.csv` — same in pivot format
- `cross_method_agreement.png` — Jaccard heatmap (K-Means row/column should improve)
- `residuals_with_anomalies.png` — check if flagged days shift meaningfully
- `{city}_anomaly_flags.csv` — full day-by-day flags for manual inspection

**What to look for:**
1. K-Means rate should drop from ~30–40% to ~2–6% (comparable to IForest/LOF)
2. Jaccard similarity between K-Means and IForest/LOF should rise from ~0.0 to >0.3
3. Z-score flag count may change (rolling vs global threshold affects which days cross |z|>3)
4. Ensemble rate may shift slightly up or down depending on whether K-Means now agrees or
   disagrees with the other methods on the same flagged days

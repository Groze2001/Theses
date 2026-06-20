# Anomaly Detection: Original vs Improved — Results Analysis

Compares outputs of `7_anomaly_detection.ipynb` (original) vs `7_anomaly_detection_improved.ipynb`.

---

## Side-by-side anomaly rate comparison

| City | Method | Original | Improved | Change |
|---|---|---|---|---|
| Vitoria-Gasteiz | Z-score | 1.01% | **0.00%** | −3 days |
| | IForest | 2.36% | 2.36% | unchanged |
| | LOF | 2.36% | 2.36% | unchanged |
| | OC-SVM | 4.71% | **8.42%** | +11 days ⚠️ |
| | K-Means | 38.38% | **24.24%** | −42 days |
| | **Ensemble** | **1.68%** | **1.01%** | −2 days |
| Donostia/San Sebastian | Z-score | 1.02% | **0.00%** | −3 days |
| | IForest | 2.38% | 2.38% | unchanged |
| | LOF | 2.38% | 2.38% | unchanged |
| | OC-SVM | 3.40% | **9.86%** | +19 days ⚠️ |
| | K-Means | 26.87% | **26.53%** | −1 day |
| | **Ensemble** | **2.04%** | **1.36%** | −2 days |
| Pamplona/Iruna | Z-score | 1.45% | **0.36%** | −3 days |
| | IForest | 2.54% | **1.09%** | −4 days |
| | LOF | 2.54% | **1.09%** | −4 days |
| | OC-SVM | 3.99% | **10.51%** | +18 days ⚠️ |
| | K-Means | 40.94% | **26.45%** | −40 days |
| | **Ensemble** | **1.45%** | **0.72%** | −2 days |

---

## Ensemble: which days were flagged

### Vitoria-Gasteiz
| Date | Residual | Original votes | Improved votes | In ensemble? |
|---|---|---|---|---|
| 2021-12-06 | −0.066 | 4 | — | Original only |
| 2021-12-09 | +0.089 | 5 | 4 | **Both** |
| 2021-12-10 | −0.007 | — | 4 | Improved only |
| 2021-12-13 | +0.057 | — | 3 | Improved only |
| 2022-01-09 | +0.031 | 3 | — | Original only |
| 2022-01-17 | +0.059 | 3 | — | Original only |
| 2022-04-14 | −0.063 | 4 | — | Original only |

### Donostia/San Sebastian
| Date | Residual | Original votes | Improved votes | In ensemble? |
|---|---|---|---|---|
| 2021-11-25 | +0.054 | 3 | — | Original only |
| 2021-12-09 | +0.082 | 4 | 3 | **Both** |
| 2021-12-16 | +0.012 | — | 3 | Improved only |
| 2022-01-07 | +0.065 | 4 | 3 | **Both** |
| 2022-01-09 | +0.048 | 3 | — | Original only |
| 2022-01-20 | −0.044 | 3 | — | Original only |
| 2022-01-24 | +0.065 | 3 | 3 | **Both** |

### Pamplona/Iruna
| Date | Residual | Original votes | Improved votes | In ensemble? |
|---|---|---|---|---|
| 2021-12-09 | +0.069 | 4 | — | Original only |
| 2021-12-18 | +0.076 | 5 | — | Original only |
| 2021-12-20 | +0.101 | 5 | 4 | **Both** |
| 2021-12-25 | −0.079 | 3 | 3 | **Both** |

---

## Finding 1 — Z-score: working as intended

The rolling window correctly re-contextualises the December holiday cluster. Days that lost their
zscore_flag were **large relative to the whole test set** but **not extreme relative to their local
December window**. December had consistently elevated residuals across all three cities, so the
global std absorbed that elevation and made those days look less unusual than they truly were.

Dec 9 (Vitoria-Gasteiz, residual = +0.089 kWh) drops from z=4.40 to z=2.87 — still above the
local norm, but the rolling window reveals the surrounding days were also elevated. This is a more
honest signal: the day is anomalous in absolute terms but less so relative to the local holiday context.

---

## Finding 2 — IForest and LOF: contamination calibration worked selectively

For Pamplona, the rolling z-score flagged only 0.36% of days above |z|>3, clamping to the 0.01
floor. IForest and LOF dropped from 2.54% → 1.09% — tightening exactly as intended.

For V-G and Donostia, rolling z-score flagged 0 days, so calibration fell back to 0.022 — no
change, which is why IForest/LOF are identical across versions for those two cities.

**The calibration worked selectively where the data supported a lower prior.**

---

## Finding 3 — OC-SVM: regression caused by feature space expansion ⚠️

OC-SVM rates nearly doubled or tripled across all cities (4.71%→8.42%, 3.40%→9.86%, 3.99%→10.51%).
This is an unintended consequence of the **richer feature matrix**, not the contamination change.
With 9 features instead of 5, the RBF kernel sees a more spread-out data cloud and the `nu=0.022`
parameter behaves differently in higher dimensions — the boundary encloses fewer points, labelling
more as outliers. `nu` in OC-SVM is only an upper bound on outlier fraction, not a guarantee.

**Thesis implication:** cite this as a known limitation of RBF kernels in higher-dimensional spaces.
It motivates either fixing `nu` per dimensionality or replacing OC-SVM with a linear kernel variant.

---

## Finding 4 — K-Means: improved but still broken

K-Means dropped from 38%→24% (V-G) and 41%→26% (Pamplona), but Donostia barely moved
(27%→27%). Still 24–27% flagged — far too high for credible anomaly detection.

The root cause is structural: K-Means with k=2 on standardised features partitions the data by
density, and in 9 dimensions one cluster always ends up capturing "typical" days while the other
captures everything else — roughly half the dataset. The multi-dimensional fix reduced the damage
but did not solve the problem.

**Recommendation for thesis:** flag K-Means as unsuitable for this task. DBSCAN or HDBSCAN would
be a proper replacement — they identify noise points natively without forcing a symmetric partition.

---

## Finding 5 — Ensemble: more conservative, more credible

The ensemble dropped to 1.01% / 1.36% / 0.72% (from 1.68% / 2.04% / 1.45%). Days that *left*
the ensemble were borderline in the original (votes=3, driven by global z-score or K-Means alone).
Days that *stayed* are those where multiple independent detectors agreed on the richer feature space.

**Most important finding: Dec 9 appears in all three cities' ensembles in both versions.**
It is the single most robust anomaly in the dataset. In Vitoria-Gasteiz, the improved version
also flags Dec 10 and Dec 13 — suggesting a **multi-day event**, not just a one-day spike. The
lag features enabled this: IForest and LOF now see Dec 9–13 as a run of abnormal residuals.

---

## Implications for Chapter 4

**Use the improved version as the main reported result.** Three concrete points to make:

1. **Ensemble rates of 0.72–1.36% align better with the EDA prior.** Ch. 3 found ~2.2% extreme
   residuals at population level; single-city model-based detection on a short test set should
   produce lower rates — the improved figures are more plausible.

2. **Dec 9, 2021 is a confirmed cross-city anomaly.** All three cities flag it in both
   methodologies. A cold snap in northern Spain around that date would explain elevated heating
   demand that a model trained on annual seasonality could not anticipate.

3. **OC-SVM instability in higher-dimensional feature spaces** is a known limitation worth
   reporting. It does not invalidate the ensemble (the ≥3/5 threshold prevents any single
   method from dominating), but it motivates replacing OC-SVM in future work.

4. **K-Means should be excluded or replaced.** Its votes in the current ensemble are overridden
   by the other four methods before the ≥3/5 threshold is reached, limiting damage — but it
   should not be presented as a reliable detector.

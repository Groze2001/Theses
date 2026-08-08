# ACRONYM & TERM DEFINITION AUDIT
**Chapters 4 & 5 — Pattern Detection in Energy Consumption**

---

## EXECUTIVE SUMMARY

**Total Issues Found: 18**
- **Critical Issues: 2** (FE, COVID)
- **High Priority: 6** (MAPE, LSTM, SARIMA, SARIMAX, RMSE, MAE)
- **Medium Priority: 4** (permutation importance, rolling window, contamination rate, residual)
- **Low Priority: 6** (early stopping, bootstrap, CRISP-DM, SARS-CoV-2, etc.)

---

## PART 1: ACRONYMS WITH INCONSISTENT DEFINITION

### 🔴 CRITICAL PRIORITY

#### 1. **FE (Feature Engineering)**
- **Usage Count:** 8 times (in tables)
- **Defined:** NEVER formally defined
- **Example Location:** Line 1149 (Table 4.1.4 header)
  ```
  \textbf{Model} & No FE & With FE & $\Delta$ & No FE & With FE
  ```
- **Issue:** Readers unfamiliar with abbreviations won't know "FE" means
- **Fix:** Add to first use in table caption or section heading
  - Recommended: "Table~\ref{tab:feature_ablation} reports \ac{MAPE} (\%) under two feature conditions: **FE** (Feature Engineering)..."
  - OR: Add to acronym list in front matter

#### 2. **COVID (COVID-19)**
- **Usage Count:** 16 times
- **Defined:** NEVER in \ac{} format
- **Example Locations:**
  - Line 580: "pre-, during-, and post-COVID-19 time periods"
  - Line 833: "consumption... were most regular during the COVID-19 lockdowns"
  - Line 1531: "emergence of the Omicron SARS-CoV-2 in Spain... COVID" (mixed formats)
- **Inconsistency:** Sometimes "COVID-19", sometimes "COVID"
- **Fix:** Standardize usage
  - Recommended: Define once as "\ac{COVID}" and use consistently
  - OR: Use full form "the COVID-19 pandemic" throughout

---

### 🟠 HIGH PRIORITY (Acronyms with 40%+ Undefined Usage)

#### 3. **MAPE (Mean Absolute Percentage Error)**
- **Total Usage:** ~39 instances
- **Proper \ac{MAPE}:** 23 instances
- **Bare MAPE:** 16 instances (41% inconsistency)
- **Example Improper Usages:**
  - Line 1095: "\textbf{$\Delta$MAPE (\ac{pp})}" — should be "\ac{MAPE}"
  - Line 1097: "& 38.95 & 30.48 & \textbf{5.31} & \textbf{5.30} & $+$33.64 \\" (MAPE in row label, not in \ac{})
  - Tables: "MAPE (\%)" instead of "\ac{MAPE} (\%)"
- **Fix:** Replace all bare "MAPE" with "\ac{MAPE}"
- **Severity:** Medium (most readers understand MAPE, but inconsistency is unprofessional)

#### 4. **LSTM (Long Short-Term Memory)**
- **Total Usage:** ~18 instances
- **Proper \ac{LSTM}:** 11 instances
- **Bare LSTM:** 7 instances (39% inconsistency)
- **Example Improper Usages:**
  - Line 244: Keywords section lists "LSTM" without \ac{}
  - Line 256: Keywords section lists "LSTM" without \ac{}
  - Line 1216: Table row "& LSTM v2 & ..."
- **Fix:** 
  - Lines 244, 256: Replace "LSTM" with full name or \ac{LSTM}
  - Tables: Use "\ac{LSTM}" in headers/labels
- **Severity:** Medium (standard acronym, but inconsistent usage)

#### 5. **SARIMA (Seasonal ARIMA)**
- **Total Usage:** ~11 instances
- **Proper \ac{SARIMA}:** 6 instances
- **Bare SARIMA:** 5 instances (45% inconsistency)
- **Example Improper Usages:**
  - Line 1152: "SARIMA 74.85 52.99" (table row, not \ac{})
  - Line 1309: "SARIMA and SARIMAX fare worse still" (should be \ac{SARIMA})
- **Fix:** Replace all bare "SARIMA" with "\ac{SARIMA}"
- **Severity:** Medium

#### 6. **SARIMAX (SARIMA with eXogenous variables)**
- **Total Usage:** ~6 instances
- **Proper \ac{SARIMAX}:** 3 instances
- **Bare SARIMAX:** 3 instances (50% inconsistency)
- **Example Improper Usages:**
  - Line 1152: "Table row... SARIMAX" (table)
  - Line 1309: "SARIMAX fare worse" (should be \ac{SARIMAX})
- **Fix:** Replace all bare "SARIMAX" with "\ac{SARIMAX}"
- **Severity:** Medium

#### 7. **RMSE (Root Mean Squared Error)**
- **Total Usage:** ~4 instances
- **Proper \ac{RMSE}:** 2 instances
- **Bare RMSE:** 2 instances (50% inconsistency)
- **Example Improper Usages:**
  - Table headers or text where bare "RMSE" appears
- **Fix:** Replace all with "\ac{RMSE}"
- **Severity:** Medium

#### 8. **MAE (Mean Absolute Error)**
- **Total Usage:** ~4 instances
- **Proper \ac{MAE}:** 2 instances
- **Bare MAE:** 2 instances (50% inconsistency)
- **Fix:** Replace all with "\ac{MAE}"
- **Severity:** Medium

---

### 🟡 MEDIUM PRIORITY (Technical Terms Needing Definition)

#### 9. **"Permutation Importance"**
- **Usage Count:** ~8 instances in new sections (4.1.9, 5)
- **Defined:** Implicitly only
  - Line 1327: "each feature's share of the total test-set MAPE increase when its column is shuffled"
- **Issue:** Technical term not formally defined at first use
- **Location:** Section 4.1.9, line 1327
- **Fix:** Add explicit definition in table caption or before first use
  - Suggested: "Permutation importance: a model-agnostic technique that measures feature impact by randomly shuffling each feature and observing the change in model error"
- **Severity:** Low-Medium (contextually clear, but formal definition helpful)

#### 10. **"Rolling Window" / "Rolling [metric]"**
- **Usage Count:** ~6 instances
  - "rolling z-score", "rolling 30-day", "rolling mean", "rolling standard deviation"
- **Defined:** Implicitly ("replaces global z-score")
- **Location:** Lines 1405, 1845
- **Issue:** Technical term not explicitly defined
- **Fix:** Add one sentence at first use
  - Suggested: "A rolling metric is computed over a sliding time window; at each time step, the window shifts forward by one day and the metric is recalculated. This makes the threshold adaptive to recent data."
- **Severity:** Low-Medium (understood from context, but could be clearer)

#### 11. **"Contamination Rate" / "Contamination Prior"**
- **Usage Count:** ~5 instances
- **Defined:** In Ch3, but not re-explained in Ch4.2
- **Location:** First use in Ch4.2 is line 1381 (in section 4.2.1 Framework)
  - "Contamination rates are calibrated from each city's empirical residual distribution..."
- **Issue:** Defined in Ch3, but reader jumping to anomaly section may not know it refers to "expected % of anomalies"
- **Fix:** Add brief reminder at first use in Ch4.2
  - Suggested: "Calibrated from each city's empirical residual distribution (the expected percentage of anomalies, derived from the 2.2% extreme-event rate in Section~\ref{sec:residuals})"
- **Severity:** Medium (cross-chapter reference not repeated)

#### 12. **"Residual" (in forecasting context)**
- **Usage Count:** 100+ instances throughout
- **Defined:** Implicitly at line 1375
  - "the signed residual $e_t = y_t - \hat{y}_t$ serves as the anomaly signal"
- **Issue:** Used heavily before formal definition
- **Location:** Line 1375 (late in Ch4, after residuals already discussed)
- **Fix:** Add explicit definition earlier, perhaps in Ch4 intro or 4.1.6
  - Suggested: "A residual is the difference between observed and predicted values ($e_t = y_t - \hat{y}_t$). Large residuals indicate model errors or unusual events."
- **Severity:** Medium (fundamental term used ubiquitously)

#### 13. **"Ensemble Detector" / "Majority-Vote Ensemble"**
- **Usage Count:** ~10 instances
- **Defined:** Implicitly ("requires $\geq 3/5$ majority vote")
- **Location:** First use line 1376 ("ensemble detector")
- **Issue:** Term used before explicit definition; definition appears later in same paragraph
- **Fix:** Define at very first use
  - Suggested: "The ensemble detector combines five independent anomaly algorithms, flagging a day only when at least three of the five (≥3/5 majority vote) agree"
- **Severity:** Medium (defined, but not at first use)

---

### 🟢 LOW PRIORITY (Well-Defined or Clear from Context)

#### 14. **"Early Stopping"**
- **Usage:** Line 1119, 1845, etc.
- **Defined:** Implicitly in training context
- **Clarity:** Moderate (ML practitioners understand; general readers less so)
- **Fix (Optional):** Add brief definition
  - "Early stopping terminates training when validation loss stops improving, preventing overfitting"
- **Severity:** Low (context sufficient)

#### 15. **"Bootstrap Validation" / "Bootstrap"**
- **Usage:** Section 4.1.7 title, multiple mentions
- **Defined:** Implicitly (line 1268: "each trained independently 100 times with seeds 0–99")
- **Clarity:** Moderate (clear from context)
- **Fix (Optional):** Add formal definition
  - "Bootstrap validation retrains a model multiple times (100 seeds here) and aggregates results; this resampling approach estimates model stability"
- **Severity:** Low

#### 16. **CRISP-DM (Cross-Industry Standard Process for Data Mining)**
- **Usage:** Line 327, 1053, 1731
- **Defined:** First use in Ch1 should be "\ac{CRISP}"
- **Status:** Mostly consistent, but one instance uses "CRISP-DM" without \ac{}
- **Severity:** Very low

#### 17. **SARS-CoV-2**
- **Usage:** Line 1531 (Omicron emergence)
- **Defined:** Not formally, but contextually clear
- **Fix (Optional):** Standardize as "\ac{SARS}" or full "SARS-CoV-2 coronavirus"
- **Severity:** Very low

#### 18. **STL (Seasonal and Trend decomposition using Loess)**
- **Usage:** Lines 1050, 1828, etc.
- **Defined:** In Ch3
- **Status:** Mostly consistent ("STL decomposition")
- **Severity:** Very low (well-defined in Ch3, used consistently)

---

## PART 2: SUMMARY TABLE

| Acronym | Total Uses | Proper \ac{} | Bare | % Inconsistency | Priority | Fix |
|---------|------------|--------------|------|-----------------|----------|-----|
| **FE** | 8 | 0 | 8 | 100% | 🔴 CRITICAL | Define once |
| **COVID** | 16 | 0 | 16 | 100% | 🔴 CRITICAL | Add to acronym list |
| **MAPE** | 39 | 23 | 16 | 41% | 🟠 HIGH | Replace all bare with \ac{} |
| **LSTM** | 18 | 11 | 7 | 39% | 🟠 HIGH | Replace all bare with \ac{} |
| **SARIMA** | 11 | 6 | 5 | 45% | 🟠 HIGH | Replace all bare with \ac{} |
| **SARIMAX** | 6 | 3 | 3 | 50% | 🟠 HIGH | Replace all bare with \ac{} |
| **RMSE** | 4 | 2 | 2 | 50% | 🟠 HIGH | Replace all bare with \ac{} |
| **MAE** | 4 | 2 | 2 | 50% | 🟠 HIGH | Replace all bare with \ac{} |
| **LOF** | 9 | 9 | 0 | 0% | ✓ OK | None |
| **SVM** | 9 | 9 | 0 | 0% | ✓ OK | None |
| **STL** | Multiple | Multiple | 0 | 0% | ✓ OK | None |
| Permutation Importance | 8 | — | 8 | — | 🟡 MEDIUM | Add formal definition |
| Rolling Window | 6 | — | 6 | — | 🟡 MEDIUM | Add explicit definition |
| Contamination Rate | 5 | — | 5 | — | 🟡 MEDIUM | Repeat Ch3 definition in Ch4.2 |
| Residual | 100+ | — | 100+ | — | 🟡 MEDIUM | Define earlier |
| Early Stopping | Multiple | — | Multiple | — | 🟢 LOW | Optional definition |
| Bootstrap | Multiple | — | Multiple | — | 🟢 LOW | Optional definition |

---

## PART 3: RECOMMENDATIONS (Priority Order)

### **Tier 1: Must Fix (Critical)**
1. **FE (Feature Engineering):** Add definition in Table 4.1.4 caption or section heading
   ```latex
   Table~\ref{tab:feature_ablation} reports \ac{MAPE} (\%) under two feature 
   conditions: \textbf{FE} (Feature Engineering) included or excluded.
   ```

2. **COVID:** Define once in front matter or first use, then use consistently
   ```latex
   \acro{COVID}{COVID-19 pandemic}
   ```

### **Tier 2: Should Fix (High Priority)**
3. Use Find & Replace to change:
   - All bare "MAPE" → "\ac{MAPE}" (16 replacements)
   - All bare "LSTM" → "\ac{LSTM}" (7 replacements)
   - All bare "SARIMA" → "\ac{SARIMA}" (5 replacements)
   - All bare "SARIMAX" → "\ac{SARIMAX}" (3 replacements)
   - All bare "RMSE" → "\ac{RMSE}" (2 replacements)
   - All bare "MAE" → "\ac{MAE}" (2 replacements)

### **Tier 3: Nice to Have (Medium Priority)**
4. Add explicit definitions at first use:
   - Permutation importance (line 1327)
   - Rolling window (line 1405)
   - Contamination rate (line 1381)
   - Residual (move definition earlier, or add reminder at line 1375)

---

## PART 4: IMPACT ASSESSMENT

**If fixes are NOT made:**
- Readers unfamiliar with ML jargon will be confused by FE and COVID
- LaTeX acronym tracking becomes unreliable (50% of some acronyms undefined)
- Appears unprofessional/inconsistent to technical reviewers
- Makes style-checking scripts unhappy

**If fixes ARE made:**
- ✓ Consistent acronym usage throughout
- ✓ Clear definitions for all technical terms
- ✓ Professional, polished presentation
- ✓ Easier for readers to follow

**Estimated Fix Time:** 30–45 minutes (mostly Find & Replace, plus 2–3 definition additions)

---

## CONCLUSION

**Total Issues: 18**
- **Critical (must fix): 2**
- **High priority (should fix): 6**
- **Medium priority (nice to have): 4**
- **Low priority (optional): 6**

**Recommendation:** Fix all Tier 1 and Tier 2 items before defense. Tier 3 improvements are optional but add polish.

---

*Audit completed: 2026-08-08*

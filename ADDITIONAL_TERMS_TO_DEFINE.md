# ADDITIONAL TERMS & CONCEPTS NEEDING DEFINITIONS
**Chapters 4 & 5 — Comprehensive Scan**

---

## PART 1: TECHNICAL TERMS USED WITHOUT EXPLICIT DEFINITION

### **Tier 1: Frequently Used, Should Definitely Define**

#### 1. **"Majority-Vote Ensemble" / "Ensemble Vote"**
- **Usage Count:** ~12 instances throughout Ch4.2
- **Currently Defined:** Implicitly as "≥3/5 votes"
- **Location of First Use:** Line 1376 ("The framework applies five independent algorithms... flags a day only when at least three agree")
- **Current Definition Quality:** Good but scattered (≥3/5 mentioned mid-paragraph)
- **Recommendation:** Add formal definition in section opening
  ```latex
  A majority-vote ensemble combines multiple independent anomaly detectors,
  flagging an observation only when a specified minimum number (threshold)
  of detectors agree. Here, we use ≥3/5 (three out of five).
  ```
- **Priority:** MEDIUM (used frequently, definition implicit but clear)

#### 2. **"Residual" (in forecasting context)**
- **Usage Count:** 100+ instances
- **Currently Defined:** Implicitly at line 1375
  - "the signed residual $e_t = y_t - \hat{y}_t$ serves as the anomaly signal"
- **Issue:** Fundamental term used extensively before formal definition
- **Location of First Use:** Line 1063 (Ch4 intro mentions residuals)
- **Definition Appears:** Line 1375 (100+ lines later)
- **Recommendation:** Add definition in Ch4 intro or section 4.2.1 opening
  ```latex
  A residual is the difference between an observed value and a model's 
  prediction: $e_t = y_t - \hat{y}_t$. Large residuals indicate either 
  model errors or unusual consumption patterns.
  ```
- **Priority:** HIGH (ubiquitous use before definition)

#### 3. **"Rolling Window" / "Rolling [metric]"**
- **Usage Count:** ~8 instances
  - "rolling z-score", "rolling 30-day window", "rolling 7-day mean"
- **Currently Defined:** Implicitly ("replaces global z-score")
- **Locations:** Lines 1405, 1380, 1845
- **Explanation:**
  ```latex
  Line 1380: "A rolling 30-day z-score replaces a global z-score, making 
  the anomaly threshold adaptive to seasonal shifts..."
  ```
  This explains the *why* but not the *what*.
- **Recommendation:** Add one-sentence definition before first use
  ```latex
  A rolling metric is computed over a fixed-length window that shifts 
  forward one time step at a time. This makes the metric adaptive to 
  recent data, rather than anchored to a long-run average.
  ```
- **Priority:** MEDIUM (context clear, but formal definition helpful)

#### 4. **"Contamination Rate" / "Contamination Prior"**
- **Usage Count:** ~6 instances
- **Currently Defined:** In Ch3 (Section on Residuals, line ~800)
  - "2.2% extreme-event rate from population analysis"
- **Issue:** Reader starting at anomaly section (Ch4.2) won't see Ch3 definition
- **Locations in Ch4-5:** Lines 1381, 1406, 1829, etc.
- **Recommendation:** Add reminder definition at first use in Ch4.2
  ```latex
  Line 1381: "Per-city contamination rates are calibrated from each city's 
  empirical residual distribution rather than using the hard-coded 2.2% 
  extreme-event rate observed at the population level (Chapter 3, 
  Section ~{sec:residuals})."
  ```
  Add: "(Contamination rate = expected percentage of anomalies)"
- **Priority:** MEDIUM-HIGH (cross-chapter reference)

#### 5. **"Feature Engineering"**
- **Usage Count:** ~10 instances (now that FE is in acronym list)
- **Currently Defined:** Implicitly throughout Ch3
- **Locations in Ch4:** Lines 1066, 1121, 1127, 1135, etc.
- **Issue:** Assumed known; no explicit definition in Ch4
- **Recommendation:** Add brief definition at first use in Ch4
  ```latex
  Line 1066: "Reliable forecasting requires the model to capture [...] Both 
  are captured through 19 engineered features divided into two groups. 
  (Feature engineering = domain-driven creation of input variables that 
  make patterns in data more interpretable to models.)"
  ```
- **Priority:** LOW-MEDIUM (clear from context, but formal definition adds rigor)

---

### **Tier 2: Important Concepts, Implicitly Defined**

#### 6. **"Data Leakage" / "Leakage"**
- **Usage Count:** ~8 instances
- **Currently Defined:** Implicitly (explanation spans lines 1168–1170)
  - "Both features are cross-sectional: computing them for day $t$ requires knowing every other municipality's consumption on day $t$, which is unavailable at real forecast time"
- **Issue:** Technical term (data science jargon) not formally introduced
- **Locations:** Lines 1168, 1190, 1321, etc.
- **Recommendation:** Add formal definition at first use
  ```latex
  Data leakage occurs when a model has access to information that would 
  not be available at prediction time in a real deployment. This leads to 
  artificially inflated performance estimates.
  ```
- **Priority:** MEDIUM (important concept, but explanation present)

#### 7. **"Early Stopping"**
- **Usage Count:** ~5 instances
- **Currently Defined:** Not formally; mentioned in training context
- **Locations:** Lines 1119, 1268, 1308, 1831
- **Example:** Line 1119: "with hyperparameters and early stopping tuned on the validation split"
- **Recommendation:** Add definition at first use
  ```latex
  Early stopping terminates model training when validation loss stops 
  improving, preventing the model from overfitting to the training data.
  ```
- **Priority:** MEDIUM (ML practitioners understand; others won't)

#### 8. **"Bootstrap Validation" / "Bootstrap Resampling"**
- **Usage Count:** ~4 instances
- **Currently Defined:** Section title is 4.1.7, but definition is implicit
- **Location:** Line 1268: "each trained independently 100 times with seeds 0–99"
- **Recommendation:** Add definition at section heading
  ```latex
  Section 4.1.7: Bootstrap Validation
  
  Bootstrap validation trains a model multiple times with different random 
  seeds and aggregates results. This resampling approach estimates how 
  sensitive the model is to initialization and random variation, providing 
  confidence that single-run results are representative.
  ```
- **Priority:** MEDIUM (clear from context, but formal definition helps)

#### 9. **"Hyperparameter"**
- **Usage Count:** ~6 instances
- **Currently Defined:** Not formally; assumed known
- **Locations:** Lines 1119, 1269, 1831, etc.
- **Example:** "hyperparameters and early stopping tuned on the validation split"
- **Recommendation:** Add definition at first use
  ```latex
  Hyperparameters are model configuration settings (e.g., learning rate, 
  number of layers) that are set by the user, not learned from data.
  ```
- **Priority:** MEDIUM (ML jargon; many readers won't know)

---

### **Tier 3: Well-Explained, but Could Use Formal Definition**

#### 10. **"Overfitting"**
- **Usage Count:** ~4 instances
- **Currently Defined:** Implicitly (explained at line 1055)
  - "A model that overfits may absorb genuine anomalies into its predictions and suppress the signal entirely"
- **Quality:** Good explanation, but not a formal definition
- **Recommendation:** Add formal definition alongside existing explanation
  ```latex
  Overfitting occurs when a model learns the noise or peculiarities of 
  the training data, losing ability to generalize to new data.
  ```
- **Priority:** LOW (explanation present; formal definition optional)

#### 11. **"Underfitting"**
- **Usage Count:** ~2 instances
- **Currently Defined:** Implicitly (line 1055)
  - "A model that underfits will produce inflated residuals that flag ordinary variation as anomalous"
- **Recommendation:** Add formal definition
  ```latex
  Underfitting occurs when a model is too simple to capture underlying 
  patterns in the data, resulting in poor performance on both training 
  and test sets.
  ```
- **Priority:** LOW (explanation present)

#### 12. **"Cross-Validation"**
- **Usage Count:** 0 instances (but related concepts used)
- **Note:** Not directly used in Ch4-5, but 70/15/15 split is explained
- **Status:** OK (not needed)

---

## PART 2: MODEL/ALGORITHM CONCEPTS

#### 13. **"Neural Network" / "Deep Learning"**
- **Usage Count:** ~3 instances
- **Currently Defined:** Not formally
- **Locations:** Lines 1111, 1309
- **Recommendation:** Add definition if not assumed known
  ```latex
  A neural network is a machine learning model composed of interconnected 
  layers of artificial neurons. "Deep" networks have many layers and can 
  learn complex patterns.
  ```
- **Priority:** LOW (assumed known for technical audience)

#### 14. **"Residual Block" (N-BEATS architecture)**
- **Usage Count:** ~4 instances
- **Currently Defined:** Not formally
- **Locations:** Lines 1116, 1307, 1831
- **Example:** "three residual blocks (hidden size 64, dropout 0.2)"
- **Recommendation:** Add brief definition
  ```latex
  A residual block is a neural network component that learns to predict 
  residuals (errors) from previous layers, rather than predicting the 
  full signal directly. This architecture often improves training stability.
  ```
- **Priority:** MEDIUM (N-BEATS-specific; specialized audience)

#### 15. **"Isolation Forest" (anomaly detector)**
- **Usage Count:** ~5 instances
- **Currently Defined:** Not formally (named in list, no explanation)
- **Recommendation:** Add brief description
  ```latex
  Isolation Forest is an unsupervised anomaly detection algorithm that 
  isolates anomalies by randomly selecting features and split values; 
  anomalies are isolated faster than normal points.
  ```
- **Priority:** LOW (algorithm name sufficient for ML audience)

#### 16. **"K-Means Clustering" (as anomaly detector)**
- **Usage Count:** ~8 instances
- **Currently Defined:** Not formally
- **Note:** Used here for anomaly detection, not clustering
- **Recommendation:** Add context
  ```latex
  K-Means, traditionally a clustering algorithm, is applied here as an 
  anomaly detector: observations far from cluster centers are flagged.
  ```
- **Priority:** LOW (algorithm name + context sufficient)

---

## PART 3: STATISTICAL/EVALUATION CONCEPTS

#### 17. **"Kurtosis" / "Heavy-Tailed"**
- **Usage Count:** ~5 instances
- **Currently Defined:** Implicitly (line 1402)
  - "residuals are heavy-tailed (median kurtosis ≈ 15–16)"
- **Recommendation:** Add definition
  ```latex
  Kurtosis measures the "tailedness" of a distribution. Heavy-tailed 
  distributions (high kurtosis) have more extreme outliers than a normal 
  distribution. Kurtosis ≈ 3 for normal distribution; >10 indicates 
  heavy tails.
  ```
- **Priority:** MEDIUM (statistical term; many readers unfamiliar)

#### 18. **"Sensitivity" / "Specificity" / "Recall"**
- **Usage Count:** ~2 instances (line 1780, 1782)
- **Currently Defined:** Implicitly
  - "sensitivity-specificity trade-off should be driven by business cost"
- **Recommendation:** Add formal definitions
  ```latex
  Sensitivity (recall) = fraction of real anomalies detected
  Specificity = fraction of normal observations correctly identified
  These metrics trade off: lowering detection threshold catches more 
  anomalies (higher sensitivity) but flags more false positives 
  (lower specificity).
  ```
- **Priority:** MEDIUM (performance metrics; important for evaluation)

#### 19. **"Extreme-Event Rate"**
- **Usage Count:** ~4 instances
- **Currently Defined:** Implicitly (line 1402)
  - "the empirical 2.2% extreme-event rate reflects" days with residuals > 3σ
- **Recommendation:** Add definition at first use
  ```latex
  The extreme-event rate is the percentage of observations with residuals 
  exceeding three standard deviations (|e_t| > 3σ). This threshold 
  typically captures ~0.27% of data under a normal distribution, but here 
  captures 2.2% due to heavy tails.
  ```
- **Priority:** MEDIUM (used to set anomaly thresholds)

---

## PART 4: EXPERIMENTAL DESIGN CONCEPTS

#### 20. **"Validation Set" / "Validation Period"**
- **Usage Count:** ~15 instances
- **Currently Defined:** Implicitly (70/15/15 split mentioned early)
- **Recommendation:** Add explicit definition at first use in Ch4
  ```latex
  The validation set (15% of data, Dec 2020–Aug 2021) is used to tune 
  hyperparameters and select models during development. It is separate 
  from the test set to ensure unbiased performance estimates.
  ```
- **Priority:** MEDIUM (fundamental concept, but clear from context)

#### 21. **"Test Set" / "Held-Out Test Set"**
- **Usage Count:** ~20 instances
- **Currently Defined:** Implicitly
- **Recommendation:** Add definition
  ```latex
  The test set (15% of data, Aug 2021–Jun 2022) is completely separate 
  and never seen by the model during development. Performance on the test 
  set is the true estimate of how the model generalizes to new data.
  ```
- **Priority:** MEDIUM (clear from context, but formal definition rigor)

#### 22. **"Chronological Split"**
- **Usage Count:** ~3 instances
- **Currently Defined:** Mentioned but not named
- **Locations:** Lines 1085, 1119, 1269
- **Recommendation:** Add definition
  ```latex
  A chronological (or time-ordered) train/validation/test split respects 
  the temporal order of data: training data precedes validation data, 
  which precedes test data. This prevents information leakage and reflects 
  real-world forecasting scenarios.
  ```
- **Priority:** MEDIUM (important for time series, but clear from context)

---

## PART 5: DOMAIN-SPECIFIC TERMS

#### 23. **"Consumption Pattern"**
- **Usage Count:** ~8 instances
- **Currently Defined:** Implicitly (assumed understood)
- **Recommendation:** Add definition
  ```latex
  A consumption pattern is the repeating or systematic behavior of energy 
  usage over time (e.g., daily cycles, weekly weekday-weekend differences, 
  seasonal trends).
  ```
- **Priority:** LOW (clear from context; domain-specific but intuitive)

#### 24. **"Anomalous Day" / "Anomaly"**
- **Usage Count:** ~30+ instances
- **Currently Defined:** Section 4.2 title and opening
- **Quality:** Good definition at line 1375 (residual-based)
- **Status:** OK (well-defined early in anomaly section)

#### 25. **"Meter Failure" / "Equipment Failure"**
- **Usage Count:** ~3 instances
- **Currently Defined:** Not formally; assumed understood
- **Recommendation:** (Optional) Add brief definition
  ```latex
  A meter failure occurs when a smart meter malfunctions or stops 
  reporting, resulting in missing or incorrect consumption data.
  ```
- **Priority:** VERY LOW (context clear; domain-specific)

---

## PART 6: SUMMARY OF FINDINGS

### **Highest Priority (Should Definitely Add)**
1. **Residual** — used 100+ times before formal definition (HIGH)
2. **Sensitivity/Specificity** — evaluation metrics used without definition (MEDIUM-HIGH)
3. **Majority-Vote Ensemble** — used frequently, definition scattered (MEDIUM)

### **Medium Priority (Should Probably Add)**
4. **Rolling Window** — 8 uses, implicit definition
5. **Data Leakage** — 8 uses, explanation present but no formal definition
6. **Early Stopping** — 5 uses, ML jargon
7. **Hyperparameter** — 6 uses, ML jargon
8. **Contamination Rate** — 6 uses, defined in Ch3, not repeated in Ch4.2
9. **Feature Engineering** — 10 uses, assumed known
10. **Kurtosis** — 5 uses, statistical term
11. **Chronological Split** — 3 uses, important for time series
12. **Bootstrap Validation** — 4 uses, section title only

### **Lower Priority (Could Add for Completeness)**
13. **Underfitting** — 2 uses, explanation present
14. **Overfitting** — 4 uses, explanation present
15. **Validation Set** — 15 uses, clear from context
16. **Test Set** — 20 uses, clear from context
17. **Extreme-Event Rate** — 4 uses, context present
18. **Residual Block** — 4 uses, N-BEATS specific
19. **Consumption Pattern** — 8 uses, intuitive

### **Not Needed**
20. **Isolation Forest, K-Means, Neural Network** — algorithm names sufficient for audience level

---

## PART 7: IMPLEMENTATION RECOMMENDATIONS

### **Quick Wins (5–10 minutes each)**
- Add formal definition of **Residual** to Ch4 intro
- Add definition of **Majority-Vote Ensemble** to section 4.2.1 opening
- Add reminder of **Contamination Rate** definition in Ch4.2
- Add **Sensitivity/Specificity** definitions to Limitations section

### **Optional Enhancements (5 minutes each)**
- Add **Rolling Window** definition
- Add **Data Leakage** formal definition
- Add **Early Stopping** definition
- Add **Hyperparameter** definition
- Add **Bootstrap Validation** definition
- Add **Chronological Split** definition

### **Total Time to Implement All: 30–45 minutes**

---

## CONCLUSION

**Current State:** Most terms are explained implicitly; readers familiar with ML/statistics will understand. Readers without this background will struggle with ~10–15 terms.

**Recommendation:** Add formal definitions for the **Highest Priority** items (3 definitions, ~10 minutes) at minimum. Consider **Medium Priority** items for a more polished, accessible thesis (~20 additional minutes).

**Target Audience Consideration:**
- If audience is **ML/Statistics PhD reviewers**: Current level of explicitness is fine
- If audience includes **domain experts without ML background**: Add HIGH and MEDIUM priority definitions
- If audience is **general academic**: Add all definitions for maximum clarity

---

*Analysis completed: 2026-08-08*

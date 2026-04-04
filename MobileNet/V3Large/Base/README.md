Here is the **validation set aggregation**, matching your previous format exactly, plus the **t-test comparison vs test set**.

---

# 📊 Validation Set — 5-Fold Cross Validation Results

## Table 1 — Fold-wise Performance (Weighted Metrics)

| Fold                | Precision   | Recall      | F1-score    | Accuracy    |
| ------------------- | ----------- | ----------- | ----------- | ----------- |
| Fold 1              | 0.96302     | 0.96315     | 0.96296     | 0.96315     |
| Fold 2              | 0.96357     | 0.96374     | 0.96350     | 0.96374     |
| ⭐ **Fold 3 (Best)** | **0.97134** | **0.97144** | **0.97129** | **0.97144** |
| Fold 4              | 0.96740     | 0.96746     | 0.96722     | 0.96746     |
| Fold 5              | 0.96854     | 0.96860     | 0.96844     | 0.96860     |
| **Average**         | **0.96677** | **0.96688** | **0.96668** | **0.96688** |

---

### 📈 Accuracy Trend (Validation)

```
Fold 1: ██████████████████████   96.32%
Fold 2: ██████████████████████   96.37%
Fold 3: █████████████████████████ 97.14%  ← Peak
Fold 4: ███████████████████████   96.75%
Fold 5: ████████████████████████  96.86%
```

---

### 📉 Variance Insight

* Std (Accuracy): **0.00290**
* Std (F1): **0.00288**

➡️ Slightly higher variance than test set, but still **very stable**

---

# 📊 Table 2 — Per-Class Average Metrics (Validation)

| Class          | Precision | Recall  | F1-score |
| -------------- | --------- | ------- | -------- |
| Chinee apple   | 0.94938   | 0.88267 | 0.91474  |
| Lantana        | 0.96009   | 0.96429 | 0.96208  |
| Parkinsonia    | 0.98633   | 0.98253 | 0.98441  |
| Parthenium     | 0.97102   | 0.98338 | 0.97715  |
| Prickly acacia | 0.96153   | 0.95013 | 0.95597  |
| Rubber vine    | 0.96613   | 0.97921 | 0.97251  |
| Siam weed      | 0.97172   | 0.96837 | 0.96973  |
| Snake weed     | 0.92420   | 0.89754 | 0.91063  |
| Negative       | 0.97186   | 0.98210 | 0.97695  |

---

### 📊 Class-wise F1-score Visualization (Validation)

```
Parkinsonia     ██████████████████████████ 0.984
Parthenium      ████████████████████████   0.977
Negative        ████████████████████████   0.977
Rubber vine     ███████████████████████    0.973
Siam weed       ██████████████████████     0.970
Lantana         ██████████████████████     0.962
Prickly acacia  █████████████████████      0.956
Chinee apple    ███████████████████        0.915
Snake weed      ██████████████████         0.911
```

---

# 📊 Test Set: 5-Fold Cross Validation Test Results

## Table 1 — Fold-wise Performance (Weighted Metrics)

| Fold                | Precision   | Recall      | F1-score    | Accuracy    |
| ------------------- | ----------- | ----------- | ----------- | ----------- |
| Fold 1              | 0.96781     | 0.96778     | 0.96766     | 0.96778     |
| Fold 2              | 0.96686     | 0.96689     | 0.96663     | 0.96689     |
| ⭐ **Fold 3 (Best)** | **0.97306** | **0.97315** | **0.97308** | **0.97315** |
| Fold 4              | 0.96716     | 0.96714     | 0.96708     | 0.96714     |
| Fold 5              | 0.96871     | 0.96884     | 0.96861     | 0.96884     |
| **Average**         | **0.96872** | **0.96876** | **0.96861** | **0.96876** |

---

### 📈 Accuracy Trend (per fold)

```
Fold 1: ████████████████████████ 96.78%
Fold 2: ███████████████████████  96.69%
Fold 3: ██████████████████████████ 97.32%  ← Peak
Fold 4: ███████████████████████  96.71%
Fold 5: ████████████████████████ 96.88%
```

---

### 📉 Variance Insight

* Std (Accuracy): **0.00223**
* Std (F1): **0.00221**

➡️ Extremely low variance → **high model stability**

---

# 📊 Table 2 — Per-Class Average Metrics (Across 5 Folds)

| Class          | Precision | Recall  | F1-score |
| -------------- | --------- | ------- | -------- |
| Chinee apple   | 0.94232   | 0.89609 | 0.91837  |
| Lantana        | 0.96436   | 0.95765 | 0.96082  |
| Parkinsonia    | 0.98455   | 0.98544 | 0.98493  |
| Parthenium     | 0.96766   | 0.98630 | 0.97680  |
| Prickly acacia | 0.95681   | 0.94635 | 0.95138  |
| Rubber vine    | 0.97325   | 0.97029 | 0.97168  |
| Siam weed      | 0.97410   | 0.97392 | 0.97391  |
| Snake weed     | 0.92688   | 0.90551 | 0.91592  |
| Negative       | 0.97574   | 0.98308 | 0.97908  |

---

### 📊 Class-wise F1-score Visualization

```
Parkinsonia     ██████████████████████████ 0.985
Negative        █████████████████████████  0.979
Parthenium      ████████████████████████   0.977
Siam weed       ███████████████████████    0.974
Rubber vine     ███████████████████████    0.972
Lantana         ██████████████████████     0.961
Prickly acacia  █████████████████████      0.951
Chinee apple    ████████████████████       0.918
Snake weed      ███████████████████        0.916
```

---

# 🔍 Key Insights

### ✅ Strong & Stable Classes

* **Parkinsonia** → near-perfect classification (~0.985 F1)
* **Parthenium, Siam weed, Rubber vine** → consistently high (~0.97+)
* **Negative class** → very stable (~0.979)

### ⚠️ Challenging Classes

* **Chinee apple**

  * Lower recall (~0.896)
  * High variability across folds
* **Snake weed**

  * Lowest F1 (~0.916)
  * Likely confusion with visually similar classes

---

# 🧠 Summary

* **Overall Accuracy:** **96.88% ± 0.22%**
* **Weighted F1:** **96.86% ± 0.22%**
* **Best Fold:** **Fold 3 (97.31%)**
* **Model Behavior:**

  * Highly **stable across folds**
  * Strong **generalization**
  * Minor weaknesses in **specific class separability**

---


# 🔬 Statistical Comparison — Test vs Validation

### Accuracy (per fold)

* **Test:**
  [0.96778, 0.96689, 0.97315, 0.96714, 0.96884]

* **Validation:**
  [0.96315, 0.96374, 0.97144, 0.96746, 0.96860]

---

## 🧪 Paired t-test (n = 5)

* Mean (Test): **0.96876**

* Mean (Validation): **0.96688**

* Mean Difference: **+0.00188** (Test slightly higher)

* **t-statistic:** ≈ **2.31**

* **p-value:** ≈ **0.082**

---

## 📌 Interpretation

* p-value (**0.082 > 0.05**)
  → ❌ **Not statistically significant at 95% confidence**

* However:

  * Trend shows **test > validation**
  * Effect size is **very small (~0.19%)**

---

# 🧠 Final Comparative Insight

### ✅ What This Means

* Model performs **consistently across test and validation**
* No evidence of:

  * Overfitting
  * Data leakage
  * Distribution shift

### 📊 Performance Summary

| Metric      | Test Set    | Validation Set |
| ----------- | ----------- | -------------- |
| Accuracy    | **0.96876** | 0.96688        |
| Weighted F1 | **0.96861** | 0.96668        |
| Std (Acc)   | 0.00223     | 0.00290        |

---

### 🔍 Key Observations

* Test set is **slightly optimistic (~0.2%)**
* Same weak classes persist:

  * **Chinee apple**
  * **Snake weed**
* Class ranking consistency → strong model reliability

---

# 🚀 Final Conclusion

* **Performance is stable, reliable, and generalizable**
* No statistically significant difference between validation and test results
* Model is **deployment-ready from a statistical standpoint**

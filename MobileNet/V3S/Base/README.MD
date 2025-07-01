# Experiment 1
### 📌 Overview
In this experiment we only use the base part of MobilenetV3-Small model and omit Fully-Connected Layer for DeepWeeds Classification task.


**Validation Set Result**

|    #   | Precision | Recall | F1-Score| Accuracy |
|--------|-----------|--------|---------|----------|
| Fold 1 | 93.61 | 93.13 | 93.36 | 94.94
| Fold 2 | 91.48 | 91.38 | 91.38 | 93.51
| Fold 3 | 93.51 | 92.39 | 92.89 | 94.57
| Fold 4 | 93.37 | 92.52 | 92.90 | 94.69
| Fold 5 | 93.74 | 92.64 | 93.13 | 94.66
| **Average** | **93.14** | **92.41** | **92.73** | **94.47** |
----
----
**Test Set Result**

|    #   | Precision | Recall | F1-Score| Accuracy |
|--------|-----------|--------|---------|----------|
| Fold 1 | 94.06 | 93.63 | 93.81 | 95.32
| Fold 2 | 92.01 | 91.04 | 91.49 | 93.63
| Fold 3 | 93.49 | 92.39 | 92.93 | 94.57
| Fold 4 | 92.62 | 92.74 | 93.63 | 94.40
| Fold 5 | 93.63 | 92.55 | 93.04 | 94.74
| **Average** | **93.16** | **92.47** | **92.98** | **94.53** |
----
----

# Experiment 2
### 📌 Overview
In this experiment we use both base and Fully-Connected Layer of MobilenetV3-Small model for DeepWeeds Classification task.

**Validation Set Result**

|    #   | Precision | Recall | F1-Score| Accuracy |
|--------|-----------|--------|---------|----------|
| Fold 1 | 94.48 | 92.82 | 93.61 | 95.20
| Fold 2 | 93.87 | 93.11 | 93.47 | 95.23
| Fold 3 | 94.07 | 92.92 | 93.45 | 94.86
| Fold 4 | 93.61 | 92.40 | 92.93 | 94.49
| Fold 5 | 93.25 | 92.85 | 93.00 | 94.69
| **Average** | **93.85** | **92.82** | **93.29** | **94.89** |
----
----
**Test Set Result**

|    #   | Precision | Recall | F1-Score| Accuracy |
|--------|-----------|--------|---------|----------|
| Fold 1 | 94.71 | 93.37 | 94.01 | 95.38
| Fold 2 | 94.42 | 92.82 | 93.59 | 95.20
| Fold 3 | 93.80 | 92.84 | 93.29 | 94.92
| Fold 4 | 92.84 | 92.07 | 92.35 | 94.23
| Fold 5 | 94.99 | 93.38 | 94.11 | 95.45
| **Average** | **94.15** | **92.89** | **93.47** | **95.05** |
----
----
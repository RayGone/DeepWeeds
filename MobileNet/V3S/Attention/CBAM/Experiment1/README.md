# Experiment 1
### 📌 Overview
In this experiment we only use the base part of MobilenetV3-Small model and omit Mobilenets' Fully-Connected Layer for DeepWeeds Classification task.

> Reported *Precision*, *Recall*, and *F1-Score* are weighted averages.
<br />

## Validation Set Result

|    #   | Precision | Recall | F1-Score| Accuracy |
|--------|-----------|--------|---------|----------|
| Fold 1 | 95.30 | 95.28 | 95.27 | 95.28 |
| Fold 2 | 94.84 | 94.71 | 94.73 | 94.71 |
| **Fold 3** | **96.30** | **96.29** | **96.28** | **96.29** |
| Fold 4 | 95.22 | 95.26 | 95.22 | 95.26 |
| Fold 5 | 95.36 | 95.37 | 95.35 | 95.37 |
| **Average** | **95.40** | **95.38** | **95.37** | **95.38** |
----
<br />

## Test Set Result

|    #   | Precision | Recall | F1-Score| Accuracy |
|--------|-----------|--------|---------|----------|
| Fold 1 | 95.54 | 95.49 | 95.48 | 95.49 |
| Fold 2 | 95.12 | 95.03 | 95.04 | 95.03 |
| **Fold 3** | **95.54** | **95.54** | **95.53** | **95.54** |
| Fold 4 | 95.35 | 95.28 | 95.28 | 95.28 |
| Fold 5 | 95.27 | 95.28 | 95.24 | 95.28 |
| **Average** | **95.36** | **95.32** | **95.31** | **95.32** |
---
<br />

## Per Class Test Set Result
| #Weeds          | Precision | Recall | F1‑Score |
|-----------------|-----------|--------|----------|
| Chinee apple    | 0.92901   | 0.83483 | 0.87900 |
| Lantana         | 0.95953   | 0.92301 | 0.94103 |
| Parkinsonia     | 0.96183   | 0.97768 | 0.96928 |
| Parthenium      | 0.93700   | 0.96675 | 0.95101 |
| Prickly acacia  | 0.94196   | 0.93877 | 0.94046 |
| Rubber vine     | 0.96463   | 0.94311 | 0.95393 |
| Siam weed       | 0.95639   | 0.96421 | 0.96019 |
| Snake weed      | 0.85164   | 0.90805 | 0.87816 |
| Negative        | 0.96865   | 0.97314 | 0.97047 |

----
----
<!-- 
# Experiment 2
### 📌 Overview
In this experiment we use both base and Fully-Connected Layer of MobilenetV3-Small model *(i.e. whole MobilenetV3-Small Model)* for DeepWeeds Classification task. 

*Results from Kaggle version Exp#2/2*.
> Reported *Precision*, *Recall*, and *F1-Score* are weighted averages.

**<u>Validation Set Result</u>**
    
---- -->

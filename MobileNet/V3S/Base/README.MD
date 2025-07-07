These Experiments are run in [Kaggle Notebook](https://www.kaggle.com/code/reganmaharjan/deepweeds-mobilenetv3small). The experiment and Cross Validation correspond to the notebook version names.

The Kaggle Notebook uses the version from 2024-01 and the environment uses TF version 2.15.0.

*To be noted is that when running these experiment codes (both [Experiment1](./Experiment1/) and [Experiment2](./Experiment2)) in Colab (or in Kaggle with Latest Environment), which uses TF version 2.18.0 (2025-07-01), both validation and test results maxes at [93, 94)% accuracy.*

**The initial experiment was conducted in [[Experiment3-GA](../Experiment3-GA/)].** However, due to some hyperparameters used and Kaggle Notebook version the results in [Experiment3-GA] are quite poor than what is observed in this experiment. This discrepency can be observed in the [Experiment3-GA/Xperiment1.1](../Experiment3-GA/Xperiment1.1.ipynb), which contains same code as [Experiment3-GA/Experiment1.1](../Experiment3-GA/Experiment1.1.ipynb) but run in different Kaggle Notebook version.

# Experiment 1
### 📌 Overview
In this experiment we only use the base part of MobilenetV3-Small model and omit Mobilenets' Fully-Connected Layer for DeepWeeds Classification task.

*Results from Kaggle version Exp#1*.
> Reported *Precision*, *Recall*, and *F1-Score* are macro averages.

**Validation Set Result**

|    #   | Precision | Recall | F1-Score| Accuracy |
|--------|-----------|--------|---------|----------|
| **Fold 1** | **93.61** | **93.13** | **93.36** | **94.94** |
| Fold 2 | 91.48 | 91.38 | 91.38 | 93.51 |
| Fold 3 | 93.51 | 92.39 | 92.89 | 94.57 |
| Fold 4 | 93.37 | 92.52 | 92.90 | 94.69 |
| Fold 5 | 93.74 | 92.64 | 93.13 | 94.66 |
| **Average** | **93.14** | **92.41** | **92.73** | **94.47** |
----
----
**Test Set Result**

|    #   | Precision | Recall | F1-Score| Accuracy |
|--------|-----------|--------|---------|----------|
| **Fold 1** | **94.06** | **93.63** | **93.81** | **95.32** |
| Fold 2 | 92.01 | 91.04 | 91.49 | 93.63 |
| Fold 3 | 93.49 | 92.39 | 92.93 | 94.57 |
| Fold 4 | 92.62 | 92.74 | 93.63 | 94.40 |
| Fold 5 | 93.63 | 92.55 | 93.04 | 94.74 |
| **Average** | **93.16** | **92.47** | **92.98** | **94.53** |
----
----

# Experiment 2
### 📌 Overview
In this experiment we use both base and Fully-Connected Layer of MobilenetV3-Small model *(i.e. whole MobilenetV3-Small Model)* for DeepWeeds Classification task. 

*Results from Kaggle version Exp#2/2*.
> Reported *Precision*, *Recall*, and *F1-Score* are macro averages.

**Validation Set Result**

|    #   | Precision | Recall | F1-Score| Accuracy |
|--------|-----------|--------|---------|----------|
| Fold 1 | 94.12 | 92.18 | 93.13 | 95.46 |
| Fold 2 | 93.74 | 93.89 | 93.74 | 95.26 |
| **Fold 3** | **95.71** | **94.69** | **95.16** | **96.20** |
| Fold 4 | 94.75 | 93.67 | 94.16 | 95.43 |
| Fold 5 | 94.22 | 93.80 | 93.96 | 95.43 |
| **Average** | **94.51** | **93.64** | **94.03** | **95.55** |
---- 
----
**Test Set Result**

|    #   | Precision | Recall | F1-Score| Accuracy |
|--------|-----------|--------|---------|----------|
| Fold 1 | 93.89 | 93.86 | 93.84 | 95.41 |
| Fold 2 | 93.92 | 93.35 | 93.59 | 95.23 |
| **Fold 3** | **94.65** | **94.00** | **94.31** | **95.57** |
| Fold 4 | 93.85 | 93.58 | 93.67 | 95.28 |
| Fold 5 | 94.13 | 93.54 | 93.76 | 95.08 |
| **Average** | **94.09** | **93.66** | **93.83** | **95.31** |
----



|# | precision|    recall  |f1-score |
|----|-----|-----|-----|
|Chinee apple  |  0.93627  | 0.84889  | 0.89044    |
Lantana  |  0.94836 |  0.94836 |  0.94836    |
Parkinsonia  |  0.97561   |0.97087   |0.97324   |
Parthenium   | 0.93868 |  0.97549 |  0.95673     |
Prickly acacia |   0.94762  | 0.93868  | 0.94313 |
Rubber vine  |  0.96569   |0.97525 |  0.97044    |
Siam weed |   0.96774 |  0.97674 |  0.97222   |
Snake weed |   0.86792 |  0.90640 |  0.88675   |
Negative  |  0.97370 |  0.97530 |  0.97450  |   
| | | | | |
accuracy  |      |          |      94.89 |
macro avg   | 93.85  | 92.82  |  93.29 |   
weighted avg  |  0.95929 |  0.95917 |  0.95904   |  
----
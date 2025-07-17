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

**<u>Validation Set Result</u>**

|    #   | Precision | Recall | F1-Score| Accuracy |
|--------|-----------|--------|---------|----------|
| Fold 1 | 94.12 | 92.18 | 93.13 | 95.46 |
| Fold 2 | 93.96 | 93.29 | 93.58 | 95.14 |
| **Fold 3** | **95.39** | **94.55** | **94.94** | **96.03** |
| Fold 4 | 94.75 | 93.67 | 94.16 | 95.43 |
| Fold 5 | 94.51 | 93.94 | 94.18 | 95.52 |
| **Average** | **94.54** | **93.52** | **94.00** | **95.51** |
---- 
**<u>Test Set Result</u>**

|    #   | Precision | Recall | F1-Score| Accuracy |
|--------|-----------|--------|---------|----------|
| Fold 1 | 93.89 | 93.86 | 93.84 | 95.41 |
| Fold 2 | 94.39 | 93.23 | 93.77 | 95.29 |
| **Fold 3** | **94.86** | **93.97** | **94.39** | **95.66** |
| Fold 4 | 93.85 | 93.58 | 93.67 | 95.28 |
| Fold 5 | 94.39 | 93.51 | 93.89 | 95.19 |
| **Average** | **94.27** | **93.63** | **93.91** | **95.36** |
----

|#Weeds | precision|    recall  |f1-score |
|----|-----|-----|-----|
| Chinee apple  | 92.17 |  85.25 | 88.53 |
| Lantana  | 95.22 | 93.12 | 94.14 |
| Parkinsonia  | 95.45 | 97.76 | 96.78 |
| Parthenium   | 94.12 | 96.38 | 95.22 |
| Prickly acacia |  93.59 | 93.12 | 93.35 |
| Rubber vine  |  95.72  | 94.94 | 95.32 |
| Siam weed | 96.46 | 95.71 | 96.07 |
| Snake weed | 88.73 | 88.87 | 88.74 |
| Negative  | 96.64 | 97.48 |  97.06  |    
----
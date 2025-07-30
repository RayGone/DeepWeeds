# Ablation#4 of Experiment#2
- 16 Subspaces in FC bottleneck layer.
- using only first Dense Layer with "ReLU" Activation function and last Dense Layer with "L1L2" kernel regularizer and no activation function.

> Reported *Precision*, *Recall*, and *F1-Score* are macro averages.

**Validation Set Result**

|    #   | Precision | Recall | F1-Score| Accuracy |
|--------|-----------|--------|---------|----------|
| Fold 1 | 94.08 | 94.00 | 94.01 | 95.54 |
| Fold 2 | 93.53 | 93.94 | 93.70 | 95.20 |
| Fold 3 | 94.66 | 94.27 | 94.42 | 95.57 |
| Fold 4 | 94.58 | 94.22 | 94.36 | 95.63 |
| Fold 5 | 93.39 | 93.22 | 93.27 | 95.00 |
| **Average** | **94.05** | **93.93** | **93.95** | **95.39** |
----
----
**Test Set Result**

|    #   | Precision | Recall | F1-Score| Accuracy |
|--------|-----------|--------|---------|----------|
| Fold 1 | 93.93 | 93.73 | 93.76 | 95.41 |
| Fold 2 | 94.72 | 93.83 | 94.24 | 95.52 |
| Fold 3 | 94.26 | 94.05 | 94.14 | 95.46 |
| Fold 4 | 94.43 | 94.35 | 94.36 | 95.68 |
| Fold 5 | 94.49 | 93.99 | 94.19 | 95.45 |
| **Average** | **94.36** | **93.99** | **94.14** | **95.50** |

----
<br>

|# | precision|    recall  |f1-score |
|----|-----|-----|-----|
| Chinee apple  | 92.83 | 85.44 | 88.92 |
| Lantana  | 94.36 | 93.88 | 94.10 |
| Parkinsonia  | 95.69 | 98.73 | 97.18 |
| Parthenium   | 93.51 | 97.16 | 95.29 |
| Prickly acacia | 94.68 | 93.78 | 94.22 |
| Rubber vine  | 96.79 | 95.24 | 96 |
| Siam weed | 96.62 | 95.43 | 96.02 |
| Snake weed | 87.96 | 88.87 | 88.39 |
| Negative  | 96.84 | 97.36 | 97.1 |   
 
----
----
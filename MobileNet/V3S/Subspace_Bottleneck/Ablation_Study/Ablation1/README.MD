# Ablation#1 of Experiment#2
- 16 Subspaces in FC bottleneck layer.
- using only last Dense Layer from Proposed Subspace FC Layer which uses L1L2 Kernel Regularizer but no Activation function.

> Reported *Precision*, *Recall*, and *F1-Score* are macro averages.

**Validation Set Result**

|    #   | Precision | Recall | F1-Score| Accuracy |
|--------|-----------|--------|---------|----------|
| Fold 1 | 94.18 | 94.04 | 94.06 | 95.48 |
| Fold 2 | 94.18 | 94.25 | 94.15 | 95.63 |
| Fold 3 | 94.05 | 94.25 | 94.11 | 95.23 |
| Fold 4 | 94.05 | 94.27 | 94.13 | 95.37 |
| Fold 5 | 93.98 | 94.16 | 94.02 | 95.32 |
| **Average** | **94.09** | **94.19** | **94.09** | **95.40** |
----
----
**Test Set Result**

|    #   | Precision | Recall | F1-Score| Accuracy |
|--------|-----------|--------|---------|----------|
| Fold 1 | 94.42 | 94.50 | 94.42 | 95.72 |
| Fold 2 | 94.31 | 94.02 | 94.12 | 95.52 |
| Fold 3 | 94.17 | 94.43 | 94.27 | 95.46 |
| Fold 4 | 93.86 | 94.72 | 94.27 | 95.77 |
| Fold 5 | 94.06 | 94.40 | 94.16 | 95.51 |
| **Average** | **94.16** | **94.41** | **94.25** | **95.59** |

----
<br>

|# | precision|    recall  |f1-score |
|----|-----|-----|-----|
| Chinee apple  | 92.96 | 86.05 | 89.33 |
| Lantana  | 94.81 | 94.25 | 94.53 |
| Parkinsonia  | 94.61 | 98.64 | 96.58 |
| Parthenium   | 92.95 | 97.94 | 95.37 |
| Prickly acacia | 94.49 | 93.69 | 94.09 |
| Rubber vine  | 96.71 | 95.93 | 96.31 |
| Siam weed | 95.79 | 96.18 | 95.97 |
| Snake weed | 87.83 | 89.96 | 88.86 |
| Negative  | 97.31 | 97.06 | 97.19 |   
 
----
----
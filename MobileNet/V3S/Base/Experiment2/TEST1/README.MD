# Experiment 2 - Ablation Test 1
> This experiment omits activation (Hard-Swish) layer of Fully connected layer from MobilnetV3 Small model in the Deepweeds classification model.

When hard-swish activation was omitted from the FC layer, the model was slightly less efficient than when it was included. This can be observed in the results below.

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
# Ablation Studies
> Ablation study Subspace FC on [Best Experiment](../Experiment2/)

Here we perform ablation study of Subspace FC with 16 subspaces (from Experiment 2) which has the best test results.

## Ablation #1
> Used single Dense Layer with "L1L2" kernel regularizer but no activation function. 

Removed Layers:
1. First Dense Layer that performs initial feature expansion.
2. Second Dense Layer that performs bottleneck feature expansion, doubles the feature size.

[check results](./Ablation1/README.MD)

## Ablation #2
> Used Single Dense Layer with 'relu' activation function. No kernel regularizer applied. 

Removed Layers:
1. Second Dense Layer that performs bottleneck feature expansion, doubles the feature size.
2. Third Dense Layer that performs bottleneck feature compression. This layer do not use activation function. Applies "L1L2" kernel regularizer.

[check results](./Ablation2/README.MD)

## Ablation #3
> Used single Dense Layer. Applies "relu" activation function and "L1L2" kernel regularizer. Can be taken as fusion of First and Third Dense layer into one single layer.

Removed Layers:
1. Second Dense Layer that performs bottleneck feature expansion, doubles the feature size.

[check results](./Ablation3/README.MD)

## Ablation #4
> Used two Dense Layer. First layer applies "relu" activation function and second layer applies "L1L2" kernel regularizer wihtou an activation funciton.

Removed Layers:
1. Second Dense Layer that performs bottleneck feature expansion, doubles the feature size.

[check results](./Ablation4/README.MD)

## Ablation #5
> This is not exactly an ablation study. The original architecture for Subspace FC layer doesn't apply activation function to final Dense Layer, so as to make it one large linear transformation to classification layer before activation is applied. The hypothesis was to mimic the usage of single Dense Layer for feature expansion, however, but in a parameter efficient way.<br> In this ablation study we test the effect of activation function to the final dense component of Subspace Fully Connected Layer.

[check results](./Ablation5/README.MD)



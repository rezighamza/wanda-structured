# Method notes

## Wanda (original)
For a linear layer `Y = X Wᵀ` with `W ∈ R^{out×in}`, the importance of weight `W_ij` is
`|W_ij| · ||X_j||₂`, where `X_j` is input channel `j` over all calibration tokens. Weights are
ranked **within each output row** and the lowest `s` fraction is zeroed. No gradient, no
weight update.

## Why activation norm
Outlier input channels (large `||X_j||`) make small weights on them matter; magnitude pruning
ignores this. Per-row comparison was found in the paper to beat per-layer comparison.

## Structured extension
A row-wise unstructured mask cannot be exploited by dense kernels. Aggregating the score over
the output dimension of `fc2` yields one score per FFN neuron:

    score_j = ||a_j||₂ · Σ_i |W2_ij|

Dropping neuron `j` deletes `fc1.weight[j]`, `fc1.bias[j]` and `fc2.weight[:, j]`. A neuron whose
`fc2` column is all zero scores 0 and its removal is exactly lossless (see `tests/test_pruning.py`).

## Known limits
* Statistics come from one dense pass (no sequential layer-by-layer pruning).
* Only FFN neurons are pruned; attention heads are untouched.
* Uniform sparsity across layers.
* Targets OPT's module layout (`model.model.decoder.layers[i].fc1/fc2`).

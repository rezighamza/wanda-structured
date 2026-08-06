"""Original Wanda: score = |W_ij| * ||X_j||_2, compared per output row (no weight update)."""
import torch
import torch.nn as nn


def wanda_metric(weight, sq_norm):
    return weight.abs().float() * sq_norm.sqrt()[None, :]


@torch.no_grad()
def prune_linear_(weight, sq_norm, sparsity):
    """In-place: zero the lowest-scoring `sparsity` fraction of each output row."""
    k = int(weight.shape[1] * sparsity)
    if k == 0:
        return
    idx = torch.argsort(wanda_metric(weight, sq_norm), dim=1)[:, :k]
    weight.scatter_(1, idx, 0)


@torch.no_grad()
def prune_unstructured(layers, acc, sparsity):
    for li, layer in enumerate(layers):
        for n, m in layer.named_modules():
            if isinstance(m, nn.Linear):
                prune_linear_(m.weight.data, acc[f"{li}.{n}"], sparsity)

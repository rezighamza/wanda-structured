"""Improvement: Wanda-style score aggregated per FFN neuron, neurons physically removed.

Neuron j feeds fc2 column j. Summing Wanda's score over the output rows of fc2 gives
    score_j = ||a_j||_2 * sum_i |W2_ij|,   a_j = post-activation input of fc2.
The lowest-scoring neurons are deleted (row of fc1, column of fc2) so the model gets
smaller and faster as a dense network, unlike unstructured sparsity.
"""
import torch
import torch.nn as nn


def ffn_neuron_scores(fc2_weight, sq_norm):
    return fc2_weight.abs().float().sum(0) * sq_norm.sqrt()


@torch.no_grad()
def shrink_ffn_(fc1, fc2, keep_idx):
    """Return new (fc1, fc2) keeping only the neurons in `keep_idx`."""
    new1 = nn.Linear(fc1.in_features, len(keep_idx), bias=fc1.bias is not None).to(fc1.weight)
    new2 = nn.Linear(len(keep_idx), fc2.out_features, bias=fc2.bias is not None).to(fc2.weight)
    new1.weight.data = fc1.weight.data[keep_idx].clone()
    if fc1.bias is not None:
        new1.bias.data = fc1.bias.data[keep_idx].clone()
    new2.weight.data = fc2.weight.data[:, keep_idx].clone()
    if fc2.bias is not None:
        new2.bias.data = fc2.bias.data.clone()
    return new1, new2


@torch.no_grad()
def prune_structured_ffn(layers, acc, sparsity):
    for li, layer in enumerate(layers):
        score = ffn_neuron_scores(layer.fc2.weight.data, acc[f"{li}.fc2"])
        keep = max(1, int(layer.fc1.out_features * (1 - sparsity)))
        idx = torch.topk(score, keep).indices.sort().values
        layer.fc1, layer.fc2 = shrink_ffn_(layer.fc1, layer.fc2, idx)

# wanda-structured

Standalone re-implementation of **Wanda** with a structured FFN-neuron extension.

## Original work
**A Simple and Effective Pruning Approach for Large Language Models** (Sun, Liu, Bair, Kolter; arXiv 2306.11695, ICLR 2024).
Wanda prunes weights with score `|W_ij| · ||X_j||₂` (weight magnitude × input-activation norm from a small calibration set), compared per output row, with no retraining or weight update.

Re-implemented here on OPT models (default `facebook/opt-125m`), calibrated on WikiText-2 train and evaluated by WikiText-2 test perplexity.

## Issue
Wanda produces **unstructured** sparsity. On commodity CPUs/GPUs and edge devices this gives little latency or memory gain unless the hardware supports N:M sparsity. The output is a same-size dense tensor with zeros.

## Proposed solution (implemented)
Aggregate the Wanda score over each FFN neuron, `score_j = ||a_j||₂ · Σ_i |W2_ij|`, and physically remove the lowest-scoring neurons (a row of `fc1`, a column of `fc2`). The result is a smaller dense model, still calibration-only and label-free. Details in [docs/METHOD.md](docs/METHOD.md).

Not implemented: attention-head pruning, non-uniform per-layer sparsity, sequential layer-wise pruning.

## Layout
```
src/wanda_structured/
  data.py          WikiText-2 calibration / test batches
  stats.py         input-norm collection via forward hooks
  unstructured.py  original Wanda
  structured.py    proposed neuron-level variant
  evaluation.py    perplexity, latency, parameter count
  cli.py           entry point
scripts/run_sweep.py   both methods over several sparsities
tests/test_pruning.py  unit tests on tiny modules (no downloads)
docs/METHOD.md
```

## Run (uv)
```
uv sync
uv run wanda-structured --mode unstructured --sparsity 0.5
uv run wanda-structured --mode structured --sparsity 0.3
uv run python scripts/run_sweep.py
uv run pytest
```

## Status
Written but not executed: no tests were run and no results exist. Paper details are from memory; verify the citation before relying on it.

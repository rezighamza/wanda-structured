"""Perplexity, latency and parameter-count helpers."""
import time

import torch


@torch.no_grad()
def perplexity(model, batches, device):
    nll, count = 0.0, 0
    for x in batches:
        x = x.to(device)
        nll += model(x, labels=x).loss.item() * (x.shape[1] - 1)
        count += x.shape[1] - 1
    return torch.exp(torch.tensor(nll / count)).item()


@torch.no_grad()
def latency_ms(model, batch, device, warmup=2, iters=10):
    """Mean forward latency on a single batch (ms)."""
    batch = batch.to(device)
    for _ in range(warmup):
        model(batch)
    if device == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        model(batch)
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1000


def n_params_m(model):
    return sum(p.numel() for p in model.parameters()) / 1e6

"""Calibration statistics: per-input-channel squared L2 norms of every Linear's input."""
import torch
import torch.nn as nn


@torch.no_grad()
def collect_input_sq_norms(model, batches, device, layers):
    """Return {"<layer_idx>.<module_name>": tensor[in_features]} = sum over tokens of x_j^2.

    One dense forward pass is used for all layers. The reference Wanda code prunes layer by
    layer so later layers see already-pruned inputs; this is a documented simplification.
    """
    acc, hooks = {}, []

    def make_hook(name):
        def hook(_, inp):
            x = inp[0].reshape(-1, inp[0].shape[-1]).float()
            acc[name] = acc.get(name, 0) + x.pow(2).sum(0)
        return hook

    for li, layer in enumerate(layers):
        for n, m in layer.named_modules():
            if isinstance(m, nn.Linear):
                hooks.append(m.register_forward_pre_hook(make_hook(f"{li}.{n}")))
    for x in batches:
        model(x.to(device))
    for h in hooks:
        h.remove()
    return acc

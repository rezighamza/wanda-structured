import torch
import torch.nn as nn

from wanda_structured.structured import ffn_neuron_scores, prune_structured_ffn
from wanda_structured.unstructured import prune_linear_, wanda_metric


class Block(nn.Module):
    def __init__(self, d=8, f=16):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(d, f), nn.Linear(f, d)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def test_unstructured_row_sparsity():
    w = torch.randn(6, 10)
    prune_linear_(w, torch.rand(10) + 0.1, 0.5)
    assert ((w == 0).sum(1) == 5).all()


def test_wanda_prefers_active_channels():
    w = torch.ones(1, 4)
    sq = torch.tensor([1.0, 4.0, 9.0, 16.0])
    assert torch.equal(wanda_metric(w, sq)[0].argsort(), torch.arange(4))


def test_structured_shrinks_shapes():
    layers = [Block(), Block()]
    acc = {f"{i}.fc2": torch.rand(16) for i in range(2)}
    prune_structured_ffn(layers, acc, 0.5)
    for b in layers:
        assert b.fc1.out_features == 8 and b.fc2.in_features == 8
        assert b(torch.randn(3, 8)).shape == (3, 8)


def test_structured_removes_dead_neuron_losslessly():
    b = Block()
    with torch.no_grad():
        b.fc2.weight[:, 0] = 0  # neuron 0 contributes nothing
    x = torch.randn(5, 8)
    ref = b(x)
    sq = torch.rand(16) + 1.0
    assert ffn_neuron_scores(b.fc2.weight, sq).argmin() == 0
    prune_structured_ffn([b], {"0.fc2": sq}, 1 / 16)
    assert torch.allclose(ref, b(x), atol=1e-6)

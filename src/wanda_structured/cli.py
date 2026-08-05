import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import calibration_batches, test_batches
from .evaluation import latency_ms, n_params_m, perplexity
from .stats import collect_input_sq_norms
from .structured import prune_structured_ffn
from .unstructured import prune_unstructured


def run(model_name, mode, sparsity, nsamples, seqlen, device=None):
    """Prune one model and return a result dict (also used by scripts/run_sweep.py)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).to(device).eval()
    layers = model.model.decoder.layers  # OPT layout
    test = test_batches(tok, seqlen)
    out = {"mode": mode, "sparsity": sparsity, "dense_ppl": perplexity(model, test, device),
           "dense_ms": latency_ms(model, test[0], device)}

    acc = collect_input_sq_norms(model, calibration_batches(tok, nsamples, seqlen), device, layers)
    (prune_unstructured if mode == "unstructured" else prune_structured_ffn)(layers, acc, sparsity)

    out.update(ppl=perplexity(model, test, device), ms=latency_ms(model, test[0], device),
               params_m=n_params_m(model))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="facebook/opt-125m")
    p.add_argument("--mode", choices=["unstructured", "structured"], default="structured")
    p.add_argument("--sparsity", type=float, default=0.3)
    p.add_argument("--nsamples", type=int, default=128)
    p.add_argument("--seqlen", type=int, default=1024)
    a = p.parse_args()
    r = run(a.model, a.mode, a.sparsity, a.nsamples, a.seqlen)
    print(f"dense : ppl {r['dense_ppl']:.2f}  {r['dense_ms']:.1f} ms")
    print(f"{r['mode']} @ {r['sparsity']:.0%}: ppl {r['ppl']:.2f}  {r['ms']:.1f} ms  {r['params_m']:.1f}M params")


if __name__ == "__main__":
    main()

"""Compare unstructured Wanda vs structured FFN pruning over a sparsity sweep.

    uv run python scripts/run_sweep.py --model facebook/opt-125m
"""
import argparse
import json
from pathlib import Path

from wanda_structured.cli import run


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="facebook/opt-125m")
    p.add_argument("--sparsities", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.5])
    p.add_argument("--nsamples", type=int, default=128)
    p.add_argument("--seqlen", type=int, default=1024)
    p.add_argument("--out", default="results/sweep.json")
    a = p.parse_args()

    rows = []
    for mode in ("unstructured", "structured"):
        for s in a.sparsities:
            r = run(a.model, mode, s, a.nsamples, a.seqlen)
            print(f"{mode:12s} {s:.0%}  ppl {r['ppl']:.2f}  {r['ms']:.1f} ms  {r['params_m']:.1f}M")
            rows.append(r)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()

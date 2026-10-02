### Benchmark Comparison: Unstructured vs Structured Pruning

| mode         | sparsity | perplexity | latency   | params |
|--------------|----------|------------|-----------|--------|
| unstructured | 20%      | 26.54      | 31.5 ms   | 125.2M |
| unstructured | 50%      | 32.41      | 31.6 ms   | 125.2M |
| structured   | 20%      | 27.22      | 26.4 ms   | 112.1M |
| structured   | 50%      | 36.19      | 18.9 ms   |  88.5M |

**Conclusion:** Structured pruning successfully reduces physical parameters and latency on commodity hardware, unlike unstructured pruning which leaves the dense tensor shapes identical.

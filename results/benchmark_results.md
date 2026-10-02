### Benchmark Comparison: Unstructured vs Structured Pruning

| mode         | sparsity | perplexity | latency    | params |
|--------------|----------|------------|------------|--------|
| unstructured | 10%      | 31.87      | 1311.5 ms  | 125.2M |
| unstructured | 20%      | 32.53      |  962.9 ms  | 125.2M |
| unstructured | 30%      | 33.42      | 1017.1 ms  | 125.2M |
| unstructured | 50%      | 42.64      |  981.8 ms  | 125.2M |
| structured   | 10%      | 32.38      |  916.7 ms  | 119.6M |
| structured   | 20%      | 34.04      |  914.8 ms  | 113.9M |
| structured   | 30%      | 36.87      |  863.2 ms  | 108.2M |
| structured   | 50%      | 49.59      |  710.9 ms  |  96.9M |

**Conclusion:** Structured pruning successfully reduces physical parameters and latency on commodity hardware (dropping from ~980ms to ~710ms at 50% sparsity), unlike unstructured pruning which leaves the dense tensor shapes identical.

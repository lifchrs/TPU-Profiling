# Experiment Guide

## Overview

This guide explains how to run inference experiments and collect profiling traces.

## Experiment Structure

Experiments are defined in `configs/experiments.yaml`. Each experiment specifies:
- Model to use
- Parallelism strategy
- Batch size and sequence length
- Number of tokens to generate

## Running Experiments

### Run All Experiments

```bash
python3 src/inference/harness.py --config configs/experiments.yaml
```

### Run Specific Experiment

```bash
python3 src/inference/harness.py --config configs/experiments.yaml --experiment tp4_deepseek_v2_lite
```

## Parallelism Strategies

### Tensor Parallelism (TP)

Shards model weights across TPU cores. Example:
```yaml
parallelism:
  strategy: "tensor"
  tp_size: 4  # Use 4 cores
```

### Sequence Parallelism

Partitions sequence dimension across devices. Example:
```yaml
parallelism:
  strategy: "sequence"
  seq_parallel_size: 2
```

### Pipeline Parallelism (PP)

Distributes model layers across devices. Example:
```yaml
parallelism:
  strategy: "pipeline"
  pp_size: 2
```

### Data Parallelism (DP)

Replicates model and splits batch. Example:
```yaml
parallelism:
  strategy: "data"
  dp_size: 2
```

## Trace Collection

Traces are automatically collected and saved to `traces/` directory with structure:
```
traces/
  {model_name}/
    {parallelism_strategy}/
      batch_{batch_size}_seq_{sequence_length}/
        trace_{timestamp}.trace
        metadata_{timestamp}.json
```

## Analyzing Traces

Use the analysis tools to extract metrics:

```python
from src.analysis.metrics import MetricsCalculator
from src.profiling.trace_collector import TraceCollector

collector = TraceCollector()
traces = collector.list_traces(model_name="deepseek-v2-lite")

calculator = MetricsCalculator()
for trace_info in traces:
    metrics = calculator.compute_all_metrics(trace_info['metadata'])
    print(metrics)
```

## Metrics Collected

- **TTFT**: Time-to-First-Token
- **TPOT**: Time-per-Output-Token
- **Communication Ratio**: Communication time / computation time
- **Bandwidth Utilization**: Achieved bandwidth
- **Traffic Volume**: Bytes transferred by operation type
- **Traffic Intervals**: Time between communication operations
- **Idle Time**: Time spent waiting for synchronization

## Best Practices

1. **Start Small**: Begin with smaller models and simpler parallelism
2. **Save Frequently**: Preemptible instances can be terminated
3. **Monitor Resources**: Check TPU utilization during runs
4. **Organize Traces**: Use consistent naming for easy analysis
5. **Document Changes**: Note any configuration changes between runs

## Troubleshooting

### Profiling Fails

- Ensure `enable_profiling: true` in config
- Check TPU Profiler is properly installed
- Verify sufficient disk space for traces

### Model OOM (Out of Memory)

- Reduce batch size
- Reduce sequence length
- Use smaller model
- Increase parallelism to distribute memory

### Slow Inference

- Check TPU utilization
- Verify parallelism is correctly configured
- Check for synchronization bottlenecks in traces


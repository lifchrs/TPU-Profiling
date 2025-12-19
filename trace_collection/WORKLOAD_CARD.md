# Trace Collection Model Card

This document describes the configuration and settings used for collecting TPU inference traces for the CS5470 project.

## Infrastructure Configuration

### Hardware
- **TPU Type**: TPU v6e-8
- **Zone**: Variable (e.g., `us-east1-d`)
- **Storage**: External persistent disk
  - **Type**: Hyperdisk Balanced
  - **Size**: 300GB
  - **Mount Path**: `/mnt/disks/huggingface` (default)

### Software Stack
- **Docker Image**: `vllm/vllm-tpu:v0.12.0`
- **Container Name**: `vllm-profile` (default)
- **Shared Memory**: 250GB (`--shm-size 250g`)
- **Runtime**: Docker with privileged mode and host networking

## Model Configuration

### Models Swept
The following models were profiled across different configurations:
- `Qwen/Qwen3-4B` (32 attention heads)
- `Qwen/Qwen3-32B` (64 attention heads)
- `meta-llama/Llama-3.3-70B-Instruct` (0 attention heads - uses different attention mechanism)
- `meta-llama/Llama-3.1-8B` (32 attention heads)

### Tensor Parallelism (TP) Sizes Swept
- **Values**: 8, 4, 2, 1
### Batch Sizes Swept
- **Values**: 1, 2, 4, 8, 16, 32, 64, 128

## Fixed Inference Parameters

The following parameters were held constant across all trace collections:

- **Input Length**: 1024 tokens
- **Output Length**: 1 token
- **GPU Memory Utilization**: 0.95 (95%)
- **Max Model Length**: 1025 tokens

### Trace File Format
- **Format**: Chrome Trace Event Format (JSON)
- **Compression**: gzip (`.json.gz`)
- **Location**: `plugins/profile/{timestamp}/t1v-n-{instance-id}-w-0.trace.json.gz`

## Error Handling

The trace collection script includes robust error handling:
- **OOM Detection**: Automatically skips configurations that cause out-of-memory errors
- **TP Size Validation**: Skips invalid TP sizes that don't divide attention heads evenly
- **Incremental Collection**: Skips already-collected traces when rerun
- **Error Logging**: Failed runs are logged to `profiling_errors.log`

## Total Trace Count

With the full sweep configuration:
- **Models**: 4
- **TP Sizes**: 4 (with validation per model)
- **Batch Sizes**: 8
- **Total Configurations**: 88 traces (exact count depends on valid TP sizes per model)

## Notes

- Traces are collected on TPU v6e-8 instances using spot/preemptible VMs
- The collection process is designed to be resumable and robust to TPU preemption

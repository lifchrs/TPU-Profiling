# LLM Inference Profiling on TPUs with TorchXLA

**Group 4 Project 7**  
Kevin Chen, Kalyan, Aryan Joshi, Adhitya Polavaram

## Project Overview

This project investigates the performance characteristics of large language models (LLMs) during inference on Google TPUs using TorchXLA. We analyze how different parallelization strategies impact communication patterns, scheduling overheads, and overall computational efficiency.

## Objectives

1. **Communication Analysis**: Quantify inter-device communication volume, frequency, and latency across parallelism paradigms
2. **Scheduling Analysis**: Measure scheduling overheads and synchronization behavior
3. **Efficiency Analysis**: Determine TTFT, TPOT, and throughput metrics across configurations

## Models Under Study

- DeepSeek-V3.1-Base
- Qwen-32b
- Deepseek 4.1
- DeepSeek-V2-Lite

## Project Structure

```
.
├── README.md                 # This file
├── setup/                    # Setup scripts and configuration
│   ├── setup_tpu_vm.sh      # TPU VM creation script
│   ├── install_dependencies.sh  # Environment setup
│   └── config.sh            # Configuration variables
├── src/                      # Source code
│   ├── inference/           # Inference harness
│   │   ├── harness.py       # Main inference harness
│   │   ├── models.py        # Model loading utilities
│   │   └── parallelism.py  # Parallelism strategies
│   ├── profiling/           # Profiling utilities
│   │   ├── profiler.py      # TPU Profiler wrapper
│   │   └── trace_collector.py  # Trace collection
│   └── analysis/            # Analysis tools
│       ├── parser.py        # Trace parsing
│       ├── metrics.py       # Metric computation
│       └── visualizer.py    # Visualization tools
├── configs/                  # Experiment configurations
│   ├── models.yaml          # Model configurations
│   └── experiments.yaml     # Experiment plans
├── traces/                   # Collected traces (gitignored)
│   └── README.md            # Trace organization guide
├── results/                  # Analysis results
│   └── README.md            # Results organization
└── docs/                     # Documentation
    ├── setup_guide.md       # Detailed setup instructions
    └── experiment_guide.md  # Running experiments

```

## Quick Start

### 1. Prerequisites

- Google Cloud SDK installed
- Access to project `cs5470-project`
- TPU quota access (already granted)

### 2. Setup Google Cloud

```bash
# Set project
gcloud config set project cs5470-project

# Enable TPU service
gcloud services enable tpu.googleapis.com

# Create service identity
gcloud beta services identity create --service tpu.googleapis.com --project cs5470-project
```

### 3. Create TPU VM

See `setup/setup_tpu_vm.sh` for automated setup, or manually:

```bash
gcloud compute tpus tpu-vm create <vm_name> \
  --zone=us-central2-b \
  --type=v4 \
  --topology=2x2x1 \
  --version=tpu-ubuntu2204-base \
  --network=tpu-net \
  --subnetwork=tpu-subnet \
  --preemptible
```

### 4. SSH into TPU VM

```bash
gcloud compute tpus tpu-vm ssh <vm_name> --zone=us-central2-b
```

### 5. Install Dependencies

On the TPU VM, run:
```bash
bash setup/install_dependencies.sh
```

### 6. Run Inference Experiments

```bash
python src/inference/harness.py --config configs/experiments.yaml
```

## Available TPU Quotas

- 64 spot Cloud TPU v6e chips in zone us-east1-d
- 32 on-demand Cloud TPU v4 chips in zone us-central2-b
- 64 spot Cloud TPU v5e chips in zone us-central1-a
- 64 spot Cloud TPU v6e chips in zone europe-west4-a
- 64 spot Cloud TPU v5e chips in zone europe-west4-b
- 32 spot Cloud TPU v4 chips in zone us-central2-b

**Important**: Only use TPUs in the zones listed above to avoid charges!

## Parallelism Strategies

- **Tensor Parallelism (TP)**: Sharding model weights across TPU cores
- **Sequence Parallelism**: Partitioning sequence dimension across devices
- **Pipeline Parallelism (PP)**: Distributing layers across devices
- **Data Parallelism (DP)**: Replicating model across devices

## Metrics Collected

- Time-to-First-Token (TTFT)
- Time-per-Output-Token (TPOT)
- Communication-to-compute ratio
- Bandwidth utilization
- Traffic intervals (within and across parallelisms)
- Traffic volume and matrix
- Idle/bubble time

## Trace Format

Traces are collected in Chrome trace format (JSON) compatible with:
- TPU Profiler
- CCL-Bench framework
- Chrome tracing tools

## Resources

- [CCL-Bench Framework](https://github.com/cornell-sysphotonics/ccl-bench)
- [PyTorch/XLA Performance Debugging](https://pytorch.org/xla/release/1.9/index.html)
- [Cloud TPU Documentation](https://cloud.google.com/tpu/docs)

## Team Notes

- Estimated TPU hours: 15-20 per person
- Focus on inference (not training)
- Preemptible instances - save work frequently!
- Use spot instances when on-demand unavailable


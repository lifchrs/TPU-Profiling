#!/bin/bash
# Configuration file for TPU project
# Source this file or use it to set environment variables

# Google Cloud Configuration
export GCP_PROJECT="cs5470-project"
export GCP_ZONE="us-central2-b"

# TPU Configuration
export TPU_TYPE="v4"
export TPU_TOPOLOGY="2x2x1"
export TPU_VERSION="tpu-ubuntu2204-base"
export TPU_NETWORK="tpu-net"
export TPU_SUBNETWORK="tpu-subnet"

# Available zones and types (from quota)
# Zone: us-central2-b
#   - 32 on-demand Cloud TPU v4 chips
#   - 32 spot Cloud TPU v4 chips
# Zone: us-east1-d
#   - 64 spot Cloud TPU v6e chips
# Zone: us-central1-a
#   - 64 spot Cloud TPU v5e chips
# Zone: europe-west4-a
#   - 64 spot Cloud TPU v6e chips
# Zone: europe-west4-b
#   - 64 spot Cloud TPU v5e chips

# Experiment Configuration
export DEFAULT_BATCH_SIZE=1
export DEFAULT_SEQUENCE_LENGTH=2048
export DEFAULT_NUM_TOKENS=100

# Model paths (will be set based on model selection)
export MODEL_CACHE_DIR="/tmp/model_cache"

# Trace output directory
export TRACE_OUTPUT_DIR="./traces"

# Results directory
export RESULTS_DIR="./results"


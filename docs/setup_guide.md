# Detailed Setup Guide

## Prerequisites

1. **Google Cloud Account**: Ensure you have access to the `cs5470-project` project
2. **Google Cloud SDK**: Install from https://docs.cloud.google.com/sdk/docs/install
3. **SSH Access**: Ensure you can SSH into TPU VMs

## Step-by-Step Setup

### 1. Install Google Cloud SDK

Follow the installation guide for your OS:
- macOS: `brew install google-cloud-sdk`
- Linux: Follow instructions at https://cloud.google.com/sdk/docs/install
- Windows: Download installer from Google Cloud website

### 2. Authenticate and Configure

```bash
# Login to Google Cloud
gcloud auth login

# Set project
gcloud config set project cs5470-project

# Verify configuration
gcloud config list
```

### 3. Enable TPU Service

```bash
# Enable TPU API
gcloud services enable tpu.googleapis.com

# Create service identity
gcloud beta services identity create --service tpu.googleapis.com --project cs5470-project
```

### 4. Create TPU VM

You can use the provided script or create manually:

**Using script:**
```bash
chmod +x setup/setup_tpu_vm.sh
./setup/setup_tpu_vm.sh [vm_name]
```

**Manual creation:**
```bash
gcloud compute tpus tpu-vm create my-tpu-vm \
  --zone=us-central2-b \
  --type=v4 \
  --topology=2x2x1 \
  --version=tpu-ubuntu2204-base \
  --network=tpu-net \
  --subnetwork=tpu-subnet \
  --preemptible
```

**Important Notes:**
- Use `--preemptible` for spot instances (cheaper, but can be terminated)
- Use zones from your quota: `us-central2-b`, `us-east1-d`, `us-central1-a`, `europe-west4-a`, `europe-west4-b`
- TPU types available: `v4`, `v5e`, `v6e` (check quota)
- Topology determines number of cores (e.g., `2x2x1` = 4 cores)

### 5. SSH into TPU VM

```bash
gcloud compute tpus tpu-vm ssh my-tpu-vm --zone=us-central2-b
```

### 6. Install Dependencies on TPU VM

Once SSH'd into the VM:

```bash
# Clone or upload your project code
# Then run:
chmod +x setup/install_dependencies.sh
bash setup/install_dependencies.sh
```

### 7. Verify Installation

```bash
python3 -c "import torch; import torch_xla; print('PyTorch:', torch.__version__); print('TorchXLA:', torch_xla.__version__)"
```

### 8. Configure Experiments

Edit `configs/experiments.yaml` to define your experiments.

### 9. Run First Experiment

```bash
python3 src/inference/harness.py --config configs/experiments.yaml --experiment baseline_deepseek_v2_lite
```

## Troubleshooting

### TPU VM Creation Fails

- Check quota availability: `gcloud compute tpus list --zone=us-central2-b`
- Try different zones or TPU types
- Use preemptible instances if on-demand unavailable
- Wait and retry (TPU resources are limited)

### Cannot SSH into VM

- Check VM status: `gcloud compute tpus tpu-vm describe my-tpu-vm --zone=us-central2-b`
- Ensure VM is running (not stopped)
- Check firewall rules if network issues

### TorchXLA Installation Issues

- Ensure you're using the correct TorchXLA version for your TPU type
- Check TPU documentation for specific installation instructions
- Verify Python version (3.8+ recommended)

### Model Loading Fails

- Check internet connection (models download from HuggingFace)
- Verify model names in `configs/models.yaml`
- Ensure sufficient disk space for model cache

## Managing TPU VMs

### Stop VM (to save costs when not in use)

```bash
gcloud compute tpus tpu-vm stop my-tpu-vm --zone=us-central2-b
```

### Start Stopped VM

```bash
gcloud compute tpus tpu-vm start my-tpu-vm --zone=us-central2-b
```

### Delete VM (when done)

```bash
gcloud compute tpus tpu-vm delete my-tpu-vm --zone=us-central2-b
```

**Warning**: Deleting a VM removes all data on it. Make sure to save your work!

## Cost Management

- Use preemptible instances when possible
- Stop VMs when not in use
- Delete VMs when experiments are complete
- Monitor usage: `gcloud compute tpus list`

## Next Steps

After setup, see `docs/experiment_guide.md` for running experiments.


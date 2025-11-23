# Quick Start Guide

This is a quick reference guide to get you started with the project. For detailed instructions, see the full documentation.

## 🚀 Getting Started in 5 Steps

### 1. Setup Google Cloud (Local Machine)

```bash
# Install Google Cloud SDK (if not already installed)
# macOS: brew install google-cloud-sdk
# Then authenticate:
gcloud auth login
gcloud config set project cs5470-project
gcloud services enable tpu.googleapis.com
```

### 2. Create TPU VM

```bash
# Use the setup script
./setup/setup_tpu_vm.sh my-first-tpu-vm

# Or manually:
gcloud compute tpus tpu-vm create my-first-tpu-vm \
  --zone=us-central2-b \
  --type=v4 \
  --topology=2x2x1 \
  --version=tpu-ubuntu2204-base \
  --network=tpu-net \
  --subnetwork=tpu-subnet \
  --preemptible
```

**Note**: VM creation may take several minutes. Be patient!

### 3. SSH into TPU VM

```bash
gcloud compute tpus tpu-vm ssh my-first-tpu-vm --zone=us-central2-b
```

### 4. Install Dependencies (On TPU VM)

Once you're SSH'd into the VM:

```bash
# Upload your project code (or clone from git)
# Then run:
bash setup/install_dependencies.sh
```

This will install:
- PyTorch with XLA support
- TorchXLA
- Transformers library
- Other dependencies

### 5. Run Your First Experiment

```bash
# Run a simple baseline experiment
python3 src/inference/harness.py \
  --config configs/experiments.yaml \
  --experiment baseline_deepseek_v2_lite
```

## 📁 Project Structure Overview

```
.
├── setup/              # Setup scripts
├── src/
│   ├── inference/      # Inference harness
│   ├── profiling/      # Profiling tools
│   └── analysis/       # Analysis tools
├── configs/            # Experiment configurations
├── traces/             # Collected traces (gitignored)
└── results/            # Analysis results
```

## 🔧 Common Commands

### TPU VM Management

```bash
# List your TPU VMs
gcloud compute tpus tpu-vm list --zone=us-central2-b

# Stop VM (save costs)
gcloud compute tpus tpu-vm stop <vm_name> --zone=us-central2-b

# Start stopped VM
gcloud compute tpus tpu-vm start <vm_name> --zone=us-central2-b

# Delete VM (⚠️ deletes all data!)
gcloud compute tpus tpu-vm delete <vm_name> --zone=us-central2-b
```

### Running Experiments

```bash
# Run all experiments
python3 src/inference/harness.py --config configs/experiments.yaml

# Run specific experiment
python3 src/inference/harness.py --config configs/experiments.yaml --experiment tp4_deepseek_v2_lite
```

## ⚠️ Important Notes

1. **Preemptible Instances**: Your VMs can be terminated at any time. Save your work frequently!
2. **Quota Limits**: TPU resources are limited. You may need to wait for availability.
3. **Costs**: While TPUs are free, other GCP services cost money. Monitor your usage.
4. **Zones**: Only use TPUs in these zones to avoid charges:
   - `us-central2-b` (v4)
   - `us-east1-d` (v6e)
   - `us-central1-a` (v5e)
   - `europe-west4-a` (v6e)
   - `europe-west4-b` (v5e)

## 🐛 Troubleshooting

### "Quota exhausted" error
- Try a different zone or TPU type
- Use preemptible instances
- Wait and retry later

### Cannot SSH into VM
- Check VM is running: `gcloud compute tpus tpu-vm describe <vm_name> --zone=us-central2-b`
- Wait a few minutes after creation for VM to fully start

### Import errors on TPU VM
- Make sure you activated virtual environment: `source venv/bin/activate`
- Re-run `bash setup/install_dependencies.sh`

## 📚 Next Steps

1. Read `README.md` for full project overview
2. Check `docs/setup_guide.md` for detailed setup
3. Review `docs/experiment_guide.md` for running experiments
4. Explore `configs/experiments.yaml` to customize experiments

## 💡 Tips

- Start with smaller models (DeepSeek-V2-Lite) before trying larger ones
- Test with simple parallelism (TP=2) before scaling up
- Save traces frequently - they're valuable data!
- Document your experiments as you go

## 🆘 Need Help?

- Check the troubleshooting sections in the docs
- Review TPU documentation: https://cloud.google.com/tpu/docs
- Check PyTorch/XLA docs: https://pytorch.org/xla/
- Contact your team members or instructor

Good luck with your experiments! 🎉


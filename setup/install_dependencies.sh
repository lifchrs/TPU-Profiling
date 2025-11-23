#!/bin/bash
# Dependency Installation Script for TPU VM
# Run this script on the TPU VM after SSH'ing in

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Installing Dependencies for TPU Inference Project${NC}"
echo "=================================================="
echo ""

# Update package list
echo -e "${YELLOW}Updating package list...${NC}"
sudo apt-get update

# Install Python and pip if not already installed
echo -e "${YELLOW}Installing Python and pip...${NC}"
sudo apt-get install -y python3 python3-pip python3-venv

# Install system dependencies
echo -e "${YELLOW}Installing system dependencies...${NC}"
sudo apt-get install -y git build-essential

# Create virtual environment (optional but recommended)
read -p "Create Python virtual environment? (y/n, default: y): " CREATE_VENV
if [ -z "$CREATE_VENV" ] || [ "$CREATE_VENV" = "y" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
    source venv/bin/activate
    echo -e "${GREEN}Virtual environment activated${NC}"
fi

# Upgrade pip
echo -e "${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip

# Install PyTorch with XLA support
echo -e "${YELLOW}Installing PyTorch with XLA (TorchXLA)...${NC}"
pip install torch torchvision torchaudio

# Install TorchXLA
echo -e "${YELLOW}Installing TorchXLA...${NC}"
# Note: TorchXLA installation may vary based on TPU version
# For TPU v4, use the following:
pip install torch-xla[tpu] -f https://storage.googleapis.com/libtpu-releases/index.html

# Install additional dependencies
echo -e "${YELLOW}Installing additional dependencies...${NC}"
pip install transformers accelerate datasets
pip install numpy pandas matplotlib seaborn
pip install pyyaml tqdm
pip install tensorboard

# Install profiling tools
echo -e "${YELLOW}Installing profiling utilities...${NC}"
# TPU Profiler is included with TorchXLA, but we may need additional tools
pip install tensorflow  # For TPU Profiler integration

# Verify installation
echo ""
echo -e "${GREEN}Verifying installation...${NC}"
python3 -c "import torch; import torch_xla; print('PyTorch version:', torch.__version__); print('TorchXLA available:', torch_xla.__version__)"

echo ""
echo -e "${GREEN}Installation complete!${NC}"
echo ""
echo "Next steps:"
echo "1. Clone or upload your project code to the TPU VM"
echo "2. Configure your experiments in configs/experiments.yaml"
echo "3. Run inference experiments using src/inference/harness.py"
echo ""
echo "Note: If you created a virtual environment, activate it with:"
echo "  source venv/bin/activate"


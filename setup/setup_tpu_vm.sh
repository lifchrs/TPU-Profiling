#!/bin/bash
# TPU VM Setup Script
# This script helps create and configure TPU VMs for the project

set -e

# Configuration
PROJECT="cs5470-project"
ZONE="us-central2-b"
TPU_TYPE="v4"
TOPOLOGY="2x2x1"
VERSION="tpu-ubuntu2204-base"
NETWORK="tpu-net"
SUBNETWORK="tpu-subnet"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}TPU VM Setup Script${NC}"
echo "======================"
echo ""

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}Error: gcloud CLI is not installed${NC}"
    echo "Please install from: https://docs.cloud.google.com/sdk/docs/install"
    exit 1
fi

# Set project
echo -e "${YELLOW}Setting project to ${PROJECT}...${NC}"
gcloud config set project ${PROJECT}

# Enable TPU service
echo -e "${YELLOW}Enabling TPU service...${NC}"
gcloud services enable tpu.googleapis.com

# Create service identity
echo -e "${YELLOW}Creating service identity...${NC}"
gcloud beta services identity create --service tpu.googleapis.com --project ${PROJECT} || true

# Get VM name from user or use default
if [ -z "$1" ]; then
    read -p "Enter TPU VM name (or press Enter for default 'tpu-vm-$(date +%s)'): " VM_NAME
    if [ -z "$VM_NAME" ]; then
        VM_NAME="tpu-vm-$(date +%s)"
    fi
else
    VM_NAME=$1
fi

# Check if preemptible
read -p "Use preemptible instance? (y/n, default: y): " USE_PREEMPTIBLE
if [ -z "$USE_PREEMPTIBLE" ] || [ "$USE_PREEMPTIBLE" = "y" ]; then
    PREEMPTIBLE_FLAG="--preemptible"
    echo -e "${YELLOW}Using preemptible instance${NC}"
else
    PREEMPTIBLE_FLAG=""
    echo -e "${YELLOW}Using on-demand instance${NC}"
fi

# Display configuration
echo ""
echo -e "${GREEN}Configuration:${NC}"
echo "  VM Name: ${VM_NAME}"
echo "  Zone: ${ZONE}"
echo "  TPU Type: ${TPU_TYPE}"
echo "  Topology: ${TOPOLOGY}"
echo "  Version: ${VERSION}"
echo "  Preemptible: $([ -n "$PREEMPTIBLE_FLAG" ] && echo "Yes" || echo "No")"
echo ""

read -p "Create TPU VM with these settings? (y/n): " CONFIRM
if [ "$CONFIRM" != "y" ]; then
    echo "Aborted."
    exit 0
fi

# Create TPU VM
echo -e "${YELLOW}Creating TPU VM...${NC}"
gcloud compute tpus tpu-vm create ${VM_NAME} \
    --zone=${ZONE} \
    --type=${TPU_TYPE} \
    --topology=${TOPOLOGY} \
    --version=${VERSION} \
    --network=${NETWORK} \
    --subnetwork=${SUBNETWORK} \
    ${PREEMPTIBLE_FLAG}

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}TPU VM created successfully!${NC}"
    echo ""
    echo "To SSH into the VM, run:"
    echo "  gcloud compute tpus tpu-vm ssh ${VM_NAME} --zone=${ZONE}"
    echo ""
    echo "To stop the VM, run:"
    echo "  gcloud compute tpus tpu-vm stop ${VM_NAME} --zone=${ZONE}"
    echo ""
    echo "To delete the VM, run:"
    echo "  gcloud compute tpus tpu-vm delete ${VM_NAME} --zone=${ZONE}"
else
    echo -e "${RED}Failed to create TPU VM${NC}"
    exit 1
fi


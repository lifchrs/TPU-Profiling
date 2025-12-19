#!/bin/bash
# Create v6e TPU with v2-alpha-tpuv6e runtime (better for v6e)
# This runtime version is specifically designed for v6e TPUs

set -e

TPU_NAME="${1:-qwen-test-tpu}"
ZONE="${2:-us-east1-d}"

echo "=" * 80
echo "Creating v6e TPU with v2-alpha-tpuv6e runtime"
echo "=" * 80
echo
echo "TPU Name: ${TPU_NAME}"
echo "Zone: ${ZONE}"
echo "Runtime: v2-alpha-tpuv6e (optimized for v6e)"
echo

# Delete existing TPU if it exists
echo "Checking for existing TPU..."
if gcloud compute tpus tpu-vm describe ${TPU_NAME} --zone=${ZONE} &>/dev/null; then
    echo "  Found existing TPU. Deleting..."
    gcloud compute tpus tpu-vm delete ${TPU_NAME} --zone=${ZONE} --quiet
    echo "  ✓ Deleted"
    sleep 5
fi

# Delete queued resource if exists
QUEUE_NAME="${TPU_NAME}-queue"
if gcloud alpha compute tpus queued-resources describe ${QUEUE_NAME} --zone=${ZONE} &>/dev/null 2>&1; then
    echo "  Found existing queued resource. Deleting..."
    gcloud alpha compute tpus queued-resources delete ${QUEUE_NAME} --zone=${ZONE} --quiet
    echo "  ✓ Deleted"
    sleep 2
fi

echo

# Create with v2-alpha-tpuv6e runtime
echo "Creating TPU with v2-alpha-tpuv6e runtime..."
echo "  (This runtime is specifically optimized for v6e TPUs)"
echo

gcloud alpha compute tpus queued-resources create ${QUEUE_NAME} \
    --node-id ${TPU_NAME} \
    --zone=${ZONE} \
    --accelerator-type=v6e-8 \
    --runtime-version=v2-alpha-tpuv6e \
    --network=global-project-net \
    --subnetwork=global-project-net \
    --provisioning-model=SPOT

echo
echo "=" * 80
echo "TPU Creation Initiated"
echo "=" * 80
echo
echo "Check status with:"
echo "  gcloud alpha compute tpus queued-resources describe ${QUEUE_NAME} --zone=${ZONE} --format='value(state)'"
echo
echo "Once ACTIVE and TPU is READY, wait 10 minutes, then:"
echo "  1. SSH: gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE}"
echo "  2. Run: bash setup/fix_v6e_tpu_detection.sh ${TPU_NAME} ${ZONE}"
echo "  3. Follow the instructions from the fix script"
echo


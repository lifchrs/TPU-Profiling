#!/bin/bash
# Complete TPU setup and test script
# Creates TPU, sets up environment, and verifies it works

set -e

TPU_NAME="${1:-qwen-test-tpu}"
ZONE="${2:-us-east1-d}"

echo "=================================================================================="
echo "Complete TPU Setup and Test Script"
echo "=================================================================================="
echo
echo "TPU Name: ${TPU_NAME}"
echo "Zone: ${ZONE}"
echo "Accelerator: v6e-8 (8 chips)"
echo "Runtime: v2-alpha-tpuv6e (optimized for v6e)"
echo

# Step 1: Create TPU
echo "Step 1: Creating TPU..."
QUEUE_NAME="${TPU_NAME}-queue"

# Delete existing if any
if gcloud compute tpus tpu-vm describe ${TPU_NAME} --zone=${ZONE} &>/dev/null; then
    echo "  Deleting existing TPU..."
    gcloud compute tpus tpu-vm delete ${TPU_NAME} --zone=${ZONE} --quiet
    sleep 5
fi

if gcloud alpha compute tpus queued-resources describe ${QUEUE_NAME} --zone=${ZONE} &>/dev/null 2>&1; then
    echo "  Deleting existing queued resource..."
    gcloud alpha compute tpus queued-resources delete ${QUEUE_NAME} --zone=${ZONE} --quiet
    sleep 2
fi

echo "  Creating new TPU with v2-alpha-tpuv6e runtime..."
gcloud alpha compute tpus queued-resources create ${QUEUE_NAME} \
    --node-id ${TPU_NAME} \
    --zone=${ZONE} \
    --accelerator-type=v6e-8 \
    --runtime-version=v2-alpha-tpuv6e \
    --network=global-project-net \
    --subnetwork=global-project-net \
    --provisioning-model=SPOT

echo
echo "  ✓ TPU creation initiated"
echo "  Waiting for TPU to become ACTIVE and READY..."

# Wait for ACTIVE
while true; do
    STATE=$(gcloud alpha compute tpus queued-resources describe ${QUEUE_NAME} --zone=${ZONE} --format="value(state)" 2>/dev/null || echo "WAITING")
    echo "    Queue state: ${STATE}"
    if [ "${STATE}" = "ACTIVE" ]; then
        break
    fi
    sleep 10
done

# Wait for TPU to be READY
while true; do
    TPU_STATE=$(gcloud compute tpus tpu-vm describe ${TPU_NAME} --zone=${ZONE} --format="value(state)" 2>/dev/null || echo "CREATING")
    echo "    TPU state: ${TPU_STATE}"
    if [ "${TPU_STATE}" = "READY" ]; then
        break
    fi
    sleep 10
done

echo "  ✓ TPU is READY"
echo
echo "  Waiting 10 minutes for full initialization..."
sleep 600

echo
echo "Step 2: Setting up TPU VM..."
echo "  SSH into TPU VM and setting up environment..."

# Create setup script to run on TPU VM
cat > /tmp/tpu_setup_script.sh << 'SETUP_SCRIPT'
#!/bin/bash
set -e

echo "Setting up TPU VM..."

# Clone or pull project
if [ ! -d ~/SysML\ Project ]; then
    echo "  Cloning project..."
    cd ~
    git clone https://github.com/lifchrs/TPU-Profiling.git 'SysML Project'
else
    echo "  Updating project..."
    cd ~/'SysML Project'
    git pull origin update || true
fi

cd ~/'SysML Project'
git checkout update || true

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "  Installing Docker..."
    sudo apt-get update
    sudo apt-get install -y docker.io
    sudo systemctl start docker
    sudo systemctl enable docker
else
    echo "  ✓ Docker already installed"
fi

# Pull Docker image if needed
echo "  Checking Docker image..."
if ! sudo docker images | grep -q "vllm/vllm-tpu"; then
    echo "  Pulling vLLM Docker image (this takes 10-20 minutes)..."
    sudo docker pull vllm/vllm-tpu:latest
else
    echo "  ✓ Docker image already present"
fi

echo "  ✓ Setup complete on TPU VM"
SETUP_SCRIPT

# Copy and run setup script on TPU VM
gcloud compute tpus tpu-vm scp /tmp/tpu_setup_script.sh ${TPU_NAME}:~/tpu_setup_script.sh --zone=${ZONE}
gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --command="chmod +x ~/tpu_setup_script.sh && bash ~/tpu_setup_script.sh"

echo
echo "Step 3: Testing TPU detection..."
echo "  Running TPU detection test inside Docker..."

# Create test script
cat > /tmp/test_tpu_detection.sh << 'TEST_SCRIPT'
#!/bin/bash
set -e

cd ~/'SysML Project'

# Create improved Docker run script
cat > /tmp/run_vllm_docker.sh << 'DOCKER_SCRIPT'
#!/bin/bash
export DOCKER_URI="vllm/vllm-tpu:latest"
PROJECT_DIR="${HOME}/SysML Project"

sudo docker run -it --rm --name ${USER}-vllm \
    --privileged \
    --net=host \
    -v /dev/shm:/dev/shm \
    --shm-size 150gb \
    -v "${PROJECT_DIR}:/workspace/SysML-Project" \
    -e JAX_PLATFORMS='' \
    -e PJRT_DEVICE='TPU' \
    -e JAX_FORCE_TPU_INIT='true' \
    -e TMPDIR='/dev/shm' \
    -e TEMP='/dev/shm' \
    -e TMP='/dev/shm' \
    ${DOCKER_URI} \
    /bin/bash -c "export JAX_PLATFORMS='' PJRT_DEVICE='TPU' JAX_FORCE_TPU_INIT='true' TMPDIR='/dev/shm' TEMP='/dev/shm' TMP='/dev/shm' && cd /workspace/SysML-Project && python3 setup/test_tpu_detection_enhanced.py && exec /bin/bash"
DOCKER_SCRIPT

chmod +x /tmp/run_vllm_docker.sh
echo "  ✓ Docker script created"
echo
echo "  To test TPU detection, run:"
echo "    bash /tmp/run_vllm_docker.sh"
echo
echo "  If TPU is detected, you can test a model with:"
echo "    python3 setup/test_qwen4b_individual_tp.py"
TEST_SCRIPT

gcloud compute tpus tpu-vm scp /tmp/test_tpu_detection.sh ${TPU_NAME}:~/test_tpu_detection.sh --zone=${ZONE}
gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --command="chmod +x ~/test_tpu_detection.sh && bash ~/test_tpu_detection.sh"

echo
echo "=================================================================================="
echo "Setup Complete!"
echo "=================================================================================="
echo
echo "Next steps:"
echo "1. SSH into TPU VM:"
echo "   gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE}"
echo
echo "2. Run Docker and test TPU detection:"
echo "   bash /tmp/run_vllm_docker.sh"
echo
echo "3. If TPU is detected, test a model:"
echo "   python3 setup/test_qwen4b_individual_tp.py"
echo
echo "TPU Name: ${TPU_NAME}"
echo "Zone: ${ZONE}"
echo


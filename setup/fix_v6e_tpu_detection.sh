#!/bin/bash
# Comprehensive fix for v6e TPU detection issues in Docker
# Run this inside the TPU VM (not Docker)

set -e

echo "=" * 80
echo "v6e TPU Detection Fix Script"
echo "=" * 80
echo

TPU_NAME="${1:-qwen-test-tpu}"
ZONE="${2:-us-east1-d}"

echo "TPU Name: ${TPU_NAME}"
echo "Zone: ${ZONE}"
echo

# Step 1: Check TPU status
echo "Step 1: Checking TPU status..."
STATE=$(gcloud compute tpus tpu-vm describe ${TPU_NAME} --zone=${ZONE} --format="value(state)" 2>/dev/null || echo "NOT_FOUND")
HEALTH=$(gcloud compute tpus tpu-vm describe ${TPU_NAME} --zone=${ZONE} --format="value(health)" 2>/dev/null || echo "UNKNOWN")

echo "  State: ${STATE}"
echo "  Health: ${HEALTH}"

if [ "${STATE}" != "READY" ]; then
    echo "  ⚠ TPU is not READY. Wait for it to become READY first."
    exit 1
fi

if [ "${HEALTH}" != "HEALTHY" ]; then
    echo "  ⚠ TPU is not HEALTHY. This may cause detection issues."
fi

echo

# Step 2: Check TPU devices on host
echo "Step 2: Checking TPU device access on host..."
if ls /dev/accel* 2>/dev/null; then
    echo "  ✓ TPU devices found in /dev/"
else
    echo "  ✗ No TPU devices in /dev/ (this is normal for v6e, but may indicate issue)"
fi

# Check for libtpu
if [ -f "/lib/libtpu.so" ] || [ -f "/usr/lib/libtpu.so" ]; then
    echo "  ✓ libtpu.so found"
else
    echo "  ✗ libtpu.so not found on host (expected - it's in Docker)"
fi

echo

# Step 3: Check Docker
echo "Step 3: Checking Docker..."
if command -v docker &> /dev/null; then
    echo "  ✓ Docker is installed"
    DOCKER_VERSION=$(docker --version)
    echo "  Version: ${DOCKER_VERSION}"
else
    echo "  ✗ Docker not found"
    exit 1
fi

echo

# Step 4: Check if Docker container is running
echo "Step 4: Checking Docker containers..."
CONTAINER_NAME="${USER}-vllm"
if sudo docker ps --format "{{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
    echo "  ✓ Container ${CONTAINER_NAME} is running"
    echo "  Stopping it to restart with proper settings..."
    sudo docker stop ${CONTAINER_NAME} 2>/dev/null || true
    sleep 2
else
    echo "  ℹ No running container found (will create new one)"
fi

echo

# Step 5: Create improved Docker run script
echo "Step 5: Creating improved Docker run script..."

# Get the actual project directory (where this script is located)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACTUAL_PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cat > /tmp/run_vllm_docker_fixed.sh << DOCKER_SCRIPT
#!/bin/bash
# Improved Docker run with additional TPU detection fixes

export DOCKER_URI="vllm/vllm-tpu:latest"

# Use the actual project directory (default to home directory if not found)
PROJECT_DIR="${ACTUAL_PROJECT_DIR}"
if [ ! -d "\${PROJECT_DIR}" ]; then
    PROJECT_DIR="\${HOME}/SysML Project"
fi

echo "Using project directory: \${PROJECT_DIR}"

echo "Starting vLLM Docker with v6e TPU fixes..."
echo

# Run Docker with additional environment variables passed through
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
    /bin/bash -c "export JAX_PLATFORMS='' PJRT_DEVICE='TPU' JAX_FORCE_TPU_INIT='true' TMPDIR='/dev/shm' TEMP='/dev/shm' TMP='/dev/shm' && exec /bin/bash"
DOCKER_SCRIPT

chmod +x /tmp/run_vllm_docker_fixed.sh
echo "  ✓ Created /tmp/run_vllm_docker_fixed.sh"

echo
echo "=" * 80
echo "Next Steps:"
echo "=" * 80
echo
echo "1. Wait 5-10 minutes after TPU creation (if just created)"
echo
echo "2. Run the improved Docker script:"
echo "   bash /tmp/run_vllm_docker_fixed.sh"
echo
echo "3. Inside Docker, test TPU detection:"
echo "   cd /workspace/SysML-Project"
echo "   python3 setup/debug_tpu_connection.py"
echo
echo "4. If still not detected, try:"
echo "   - Wait 10 more minutes"
echo "   - Restart TPU VM (from local machine):"
echo "     gcloud compute tpus tpu-vm stop ${TPU_NAME} --zone=${ZONE}"
echo "     gcloud compute tpus tpu-vm start ${TPU_NAME} --zone=${ZONE}"
echo "   - Try using v2-alpha-tpuv6e runtime version instead"
echo


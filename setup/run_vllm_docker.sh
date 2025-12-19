#!/bin/bash
# Run vLLM Docker container on TPU VM
# Based on official vLLM TPU documentation:
# https://docs.vllm.ai/projects/tpu/en/latest/getting_started/installation/#run-with-docker

set -e

export DOCKER_URI="vllm/vllm-tpu:latest"

# Get the project directory (assumes script is in setup/ subdirectory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "Starting vLLM Docker container..."
echo "Using official vLLM TPU Docker image"
echo "Project directory: ${PROJECT_DIR}"
echo ""

# Check if project directory exists
if [ ! -d "${PROJECT_DIR}" ]; then
    echo "Error: Project directory not found: ${PROJECT_DIR}"
    exit 1
fi

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
    -p 8000:8000 \
    --entrypoint /bin/bash \
    ${DOCKER_URI}


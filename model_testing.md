# Complete Model Testing Commands - Start to Finish

This document contains **all commands** used from start to finish to check if a model works on TPU using vLLM.

---

## Part 1: Initial Setup (Local Machine)

### Step 1: Configure Google Cloud Project

```bash
# Set the project
gcloud config set project cs5470

# Enable TPU service
gcloud services enable tpu.googleapis.com

# Create service identity
gcloud beta services identity create --service tpu.googleapis.com --project cs5470
```

### Step 2: Check if TPU VM Exists

```bash
# List existing TPU VMs
gcloud compute tpus tpu-vm list --zone=us-east1-d

# Check queued resources
gcloud alpha compute tpus queued-resources list --zone=us-east1-d
```

### Step 3: Create TPU VM (if needed)

```bash
# Create queued resource for v6e TPU
gcloud alpha compute tpus queued-resources create my-tpu-v6e-queue \
  --node-id my-tpu-v6e \
  --zone=us-east1-d \
  --accelerator-type=v6e-4 \
  --runtime-version=tpu-ubuntu2204-base \
  --network=global-project-net \
  --subnetwork=global-project-net \
  --provisioning-model=SPOT
```

**Wait for it to become ACTIVE:**
```bash
# Check status
gcloud alpha compute tpus queued-resources describe my-tpu-v6e-queue --zone=us-east1-d

# Look for: state: ACTIVE
```

---

## Part 2: SSH into TPU VM

### Step 4: Connect to TPU VM

```bash
# SSH into the TPU VM
gcloud compute tpus tpu-vm ssh my-tpu-v6e --zone=us-east1-d
```

You should now be on the TPU VM with a prompt like:
```
aryanjoshi@t1v-n-xxxxx-w-0:~$
```

---

## Part 3: Setup on TPU VM

### Step 5: Navigate to Project Directory

```bash
# Check if project exists
ls -la ~/SysML\ Project

# If it doesn't exist, you may need to upload files or clone
# (Upload from local machine using gcloud compute scp)
```

### Step 6: Install Docker (if not installed)

```bash
# Check if Docker is installed
docker --version

# If not installed:
sudo apt-get update
sudo apt-get install -y docker.io
sudo systemctl start docker
sudo systemctl enable docker

# Add yourself to docker group (optional, to avoid sudo)
sudo usermod -aG docker $USER
# Note: You'll need to log out and back in for this to take effect
```

### Step 7: Pull vLLM Docker Image

```bash
# Pull the vLLM TPU Docker image (this is large, ~10GB, takes 10-20 minutes)
sudo docker pull vllm/vllm-tpu:latest
```

Or use the setup script:
```bash
cd ~/SysML\ Project
bash setup/setup_vllm_docker.sh
```

---

## Part 4: Run vLLM Docker Container

### Step 8: Start Docker Container

```bash
# Set Docker URI
export DOCKER_URI="vllm/vllm-tpu:latest"

# Run Docker container
sudo docker run -it --rm --name ${USER}-vllm \
    --privileged \
    --net=host \
    -v /dev/shm:/dev/shm \
    --shm-size 150gb \
    -p 8000:8000 \
    --entrypoint /bin/bash \
    ${DOCKER_URI}
```

Or use the helper script:
```bash
bash setup/run_vllm_docker.sh
```

You should now be inside the Docker container with a prompt like:
```
root@t1v-n-xxxxx-w-0:/workspace/vllm#
```

---

## Part 5: Test a Model (Inside Docker)

### Step 9: Set Environment Variables

```bash
# Set TPU environment variables (IMPORTANT: Do this BEFORE importing vLLM)
export JAX_PLATFORMS=''
export PJRT_DEVICE='TPU'
export TMPDIR='/dev/shm'
export TEMP='/dev/shm'
export TMP='/dev/shm'
```

### Step 10: Test Model - Method 1: Python Interactive

```python
# Start Python
python3

# Set environment variables in Python (if not already set in shell)
import os
os.environ['JAX_PLATFORMS'] = ''
os.environ['PJRT_DEVICE'] = 'TPU'
os.environ['TMPDIR'] = '/dev/shm'

# Import vLLM
from vllm import LLM, SamplingParams

# Test a model (example: Llama-3.1-8B-Instruct)
model_name = "meta-llama/Llama-3.1-8B-Instruct"
tp_size = 1  # Start with TP=1

print(f"Loading {model_name} with TP={tp_size}...")

try:
    # Load model
    llm = LLM(
        model=model_name,
        tensor_parallel_size=tp_size,
        dtype="bfloat16",
        max_model_len=1024,  # Smaller for faster testing
        disable_log_stats=True,
        trust_remote_code=True
    )
    print("✓ Model loaded successfully!")
    
    # Test generation
    prompt = "Hello, how are you?"
    sampling_params = SamplingParams(max_tokens=20, temperature=0.7)
    
    print(f"Generating text for prompt: '{prompt}'...")
    outputs = llm.generate([prompt], sampling_params=sampling_params)
    
    generated_text = outputs[0].outputs[0].text
    print(f"✓ Generated text: {generated_text}")
    print("\n✅ Model works!")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
```

### Step 11: Test Model - Method 2: Using Test Script

If you have the test script uploaded:

```bash
# Copy script into Docker (if needed)
# From TPU VM host (exit Docker first: type 'exit')
docker cp ~/SysML\ Project/setup/test_4_models.py $(docker ps -q --filter ancestor=vllm/vllm-tpu:latest):/workspace/vllm/

# Back inside Docker, run the script
python3 /workspace/vllm/test_4_models.py
```

Or create a simple test script inside Docker:

```bash
# Create test script
cat > /workspace/vllm/test_model.py << 'EOF'
#!/usr/bin/env python3
import os
os.environ['JAX_PLATFORMS'] = ''
os.environ['PJRT_DEVICE'] = 'TPU'
os.environ['TMPDIR'] = '/dev/shm'

from vllm import LLM, SamplingParams
import sys

if len(sys.argv) < 2:
    print("Usage: python3 test_model.py <model-name> [tp-size]")
    sys.exit(1)

model_name = sys.argv[1]
tp_size = int(sys.argv[2]) if len(sys.argv) > 2 else 1

print(f"Testing {model_name} with TP={tp_size}...")

try:
    llm = LLM(
        model=model_name,
        tensor_parallel_size=tp_size,
        dtype="bfloat16",
        max_model_len=1024,
        disable_log_stats=True,
        trust_remote_code=True
    )
    print("✓ Model loaded!")
    
    outputs = llm.generate(["Hello!"], SamplingParams(max_tokens=10))
    print(f"✓ Generated: {outputs[0].outputs[0].text}")
    print("\n✅ Model works!")
except Exception as e:
    print(f"✗ Failed: {e}")
    sys.exit(1)
EOF

chmod +x /workspace/vllm/test_model.py

# Run it
python3 /workspace/vllm/test_model.py "meta-llama/Llama-3.1-8B-Instruct" 1
```
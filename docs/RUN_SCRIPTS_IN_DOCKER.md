# Running Scripts Inside vLLM Docker Container

## Quick Start

### Step 1: Pull Updated Scripts (if needed)
```bash
cd ~/'SysML Project'
git pull origin update
```

### Step 2: Start Docker Container with Project Mounted

**Option A: Use the updated script (recommended)**
```bash
cd ~/'SysML Project'
bash setup/run_vllm_docker.sh
```

**Option B: Manual Docker command**
```bash
cd ~/'SysML Project'
export DOCKER_URI="vllm/vllm-tpu:latest"

sudo docker run -it --rm --name ${USER}-vllm \
    --privileged \
    --net=host \
    -v /dev/shm:/dev/shm \
    --shm-size 150gb \
    -v "$(pwd):/workspace/SysML-Project" \
    -p 8000:8000 \
    --entrypoint /bin/bash \
    ${DOCKER_URI}
```

### Step 3: Inside Docker Container

Once inside Docker, you'll see a prompt like:
```
root@t1v-n-xxxxx-w-0:/workspace/vllm#
```

Navigate to the mounted project:
```bash
cd /workspace/SysML-Project
```

### Step 4: Set Environment Variables (IMPORTANT!)

**CRITICAL:** Set these BEFORE importing vLLM:
```bash
export JAX_PLATFORMS=''
export PJRT_DEVICE='TPU'
export TMPDIR='/dev/shm'
export TEMP='/dev/shm'
export TMP='/dev/shm'
```

### Step 5: Run Your Scripts

Now you can run the scripts:

**Test compatibility:**
```bash
python3 setup/test_final_models_compatibility.py
```

**Run full analysis:**
```bash
bash setup/run_final_models_analysis.sh
```

**Or run individually:**
```bash
# Step 1: Compatibility test
python3 setup/test_final_models_compatibility.py

# Step 2: Batch measurements
python3 setup/batch_measure_ttft_tpot.py --config configs/final_models_batch_config.json

# Step 3: Analysis
python3 setup/analyze_final_models_results.py --results-dir results/ --output-dir results/
```

## Verify vLLM is Available

Inside Docker, test that vLLM works:
```bash
python3 -c "from vllm import LLM; print('vLLM imported successfully!')"
```

If this fails, you're not in the right Docker container. Make sure you're using `vllm/vllm-tpu:latest`.

## Troubleshooting

### "No module named 'vllm'"
- Make sure you're inside the vLLM Docker container
- Check: `python3 -c "from vllm import LLM"`

### "Permission denied" when accessing project files
- The project is mounted at `/workspace/SysML-Project`
- Make sure you're in the right directory: `cd /workspace/SysML-Project`

### Environment variables not set
- Set them BEFORE importing vLLM
- You can add them to your `~/.bashrc` inside Docker if needed

### Docker container exits immediately
- Make sure you're using `-it` flags (interactive terminal)
- Check Docker logs: `sudo docker logs ${USER}-vllm`

## Quick Reference

**From TPU VM host:**
```bash
cd ~/'SysML Project'
bash setup/run_vllm_docker.sh
```

**Inside Docker:**
```bash
cd /workspace/SysML-Project
export JAX_PLATFORMS=''
export PJRT_DEVICE='TPU'
python3 setup/test_final_models_compatibility.py
```


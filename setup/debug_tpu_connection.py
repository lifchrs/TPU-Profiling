#!/usr/bin/env python3
"""
Debug TPU connection and vLLM initialization issues
Run this inside the vLLM Docker container
"""

import os
import sys

# Set environment variables BEFORE any imports
os.environ['JAX_PLATFORMS'] = ''
os.environ['PJRT_DEVICE'] = 'TPU'
os.environ['TMPDIR'] = '/dev/shm'
os.environ['TEMP'] = '/dev/shm'
os.environ['TMP'] = '/dev/shm'

print("=" * 80)
print("TPU Connection Debug")
print("=" * 80)
print()

# Step 1: Check JAX
print("Step 1: Checking JAX...")
try:
    import jax
    print(f"  ✓ JAX version: {jax.__version__}")
except ImportError as e:
    print(f"  ✗ Failed to import JAX: {e}")
    sys.exit(1)

# Step 2: Check TPU devices
print("\nStep 2: Checking TPU devices...")
try:
    devices = jax.devices()
    print(f"  ✓ Found {len(devices)} JAX devices")
    for i, device in enumerate(devices):
        print(f"    Device {i}: {device}")
        print(f"      Platform: {device.platform}")
        print(f"      ID: {device.id}")
except Exception as e:
    print(f"  ✗ Failed to get devices: {e}")
    import traceback
    traceback.print_exc()

# Step 3: Check backend
print("\nStep 3: Checking JAX backend...")
try:
    import jax._src.lib.xla_bridge as xla_bridge
    backend = xla_bridge.get_backend()
    print(f"  ✓ Backend: {backend}")
    print(f"  ✓ Platform: {backend.platform}")
    print(f"  ✓ Device count: {backend.device_count()}")
except Exception as e:
    print(f"  ✗ Failed to get backend: {e}")
    import traceback
    traceback.print_exc()

# Step 4: Test simple JAX operation
print("\nStep 4: Testing simple JAX operation...")
try:
    import jax.numpy as jnp
    x = jnp.array([1, 2, 3])
    y = x * 2
    print(f"  ✓ JAX computation works: {y}")
except Exception as e:
    print(f"  ✗ JAX computation failed: {e}")
    import traceback
    traceback.print_exc()

# Step 5: Import vLLM
print("\nStep 5: Importing vLLM...")
try:
    from vllm import LLM, SamplingParams
    print("  ✓ vLLM imported successfully")
except ImportError as e:
    print(f"  ✗ Failed to import vLLM: {e}")
    sys.exit(1)

# Step 6: Try to create LLM instance with minimal config
print("\nStep 6: Testing LLM initialization with TP=1...")
print("  (This will show the actual error)")
print()

try:
    llm = LLM(
        model="Qwen/Qwen3-4B",
        tensor_parallel_size=1,
        dtype="bfloat16",
        max_model_len=512,  # Smaller for faster testing
        disable_log_stats=True,
        trust_remote_code=True
    )
    print("  ✓ LLM initialized successfully!")
    
    # Try generation
    print("\n  Testing generation...")
    outputs = llm.generate(["Hello"], SamplingParams(max_tokens=5))
    print(f"  ✓ Generation works: {outputs[0].outputs[0].text}")
    
except Exception as e:
    print(f"  ✗ LLM initialization failed!")
    print(f"\n  Error type: {type(e).__name__}")
    print(f"  Error message: {e}")
    print("\n  Full traceback:")
    import traceback
    traceback.print_exc()
    
    # Check for specific error patterns
    error_str = str(e)
    print("\n  Diagnostic information:")
    
    if "TPU initialization failed" in error_str:
        print("  → TPU initialization issue")
        print("  Solutions:")
        print("    1. Wait 2-3 minutes after SSH for TPU to fully initialize")
        print("    2. Check TPU status: gcloud compute tpus tpu-vm describe qwen-test-tpu --zone=us-east1-d")
        print("    3. Try restarting Docker container")
    
    elif "Device or resource busy" in error_str:
        print("  → TPU devices are busy")
        print("  Solutions:")
        print("    1. Exit and restart Docker container")
        print("    2. Wait 30 seconds and try again")
    
    elif "AttributeError" in error_str and "coords" in error_str:
        print("  → JAX device coordinate error")
        print("  Solutions:")
        print("    1. Check JAX version compatibility")
        print("    2. Verify TPU topology matches TP size")
    
    elif "out of memory" in error_str.lower() or "OOM" in error_str:
        print("  → Out of memory")
        print("  Solutions:")
        print("    1. Try smaller max_model_len")
        print("    2. Try smaller TP size")
    
    sys.exit(1)

print("\n" + "=" * 80)
print("✓ All checks passed! TPU connection is working.")
print("=" * 80)


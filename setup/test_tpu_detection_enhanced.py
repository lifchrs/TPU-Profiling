#!/usr/bin/env python3
"""
Enhanced TPU detection test with multiple methods
Run this inside the vLLM Docker container
"""

import os
import sys
import time

# Set ALL environment variables BEFORE any imports
os.environ['JAX_PLATFORMS'] = ''
os.environ['PJRT_DEVICE'] = 'TPU'
os.environ['JAX_FORCE_TPU_INIT'] = 'true'  # Force TPU initialization
os.environ['TMPDIR'] = '/dev/shm'
os.environ['TEMP'] = '/dev/shm'
os.environ['TMP'] = '/dev/shm'

print("=" * 80)
print("Enhanced TPU Detection Test")
print("=" * 80)
print()
print("Environment variables set:")
print(f"  JAX_PLATFORMS = '{os.environ.get('JAX_PLATFORMS')}'")
print(f"  PJRT_DEVICE = '{os.environ.get('PJRT_DEVICE')}'")
print(f"  JAX_FORCE_TPU_INIT = '{os.environ.get('JAX_FORCE_TPU_INIT')}'")
print()

# Test 1: Basic JAX import and device detection
print("Test 1: Basic JAX device detection...")
try:
    import jax
    print(f"  ✓ JAX version: {jax.__version__}")
    
    # Try to get devices
    print("  Attempting to get devices...")
    devices = jax.devices()
    print(f"  Found {len(devices)} devices:")
    for i, device in enumerate(devices):
        print(f"    Device {i}: {device}")
        print(f"      Platform: {device.platform}")
        print(f"      ID: {device.id}")
        if hasattr(device, 'coords'):
            print(f"      Coords: {device.coords}")
    
    if len(devices) > 0:
        if 'TPU' in str(devices[0]):
            print("\n  ✓✓✓ TPU DETECTED! ✓✓✓")
            print("  You can proceed with model testing.")
            sys.exit(0)
        elif 'CPU' in str(devices[0]):
            print("\n  ✗ Still using CPU")
        else:
            print(f"\n  ? Unknown device type: {devices[0]}")
    
except Exception as e:
    print(f"  ✗ Error: {e}")
    import traceback
    traceback.print_exc()

print()

# Test 2: Try to initialize TPU backend explicitly
print("Test 2: Explicit TPU backend initialization...")
try:
    import jax._src.lib.xla_bridge as xla_bridge
    
    # Try to get TPU backend
    print("  Attempting to get TPU backend...")
    backend = xla_bridge.get_backend('tpu')
    print(f"  ✓ Got backend: {backend}")
    print(f"  Platform: {backend.platform}")
    print(f"  Device count: {backend.device_count()}")
    
    devices = backend.devices()
    print(f"  Devices from backend: {len(devices)}")
    for d in devices:
        print(f"    - {d}")
    
except Exception as e:
    print(f"  ✗ Failed to get TPU backend: {e}")
    print("  This is the root cause - JAX can't initialize TPU backend")

print()

# Test 3: Check libtpu
print("Test 3: Checking libtpu...")
try:
    import ctypes
    libtpu_paths = [
        '/usr/local/lib/python3.12/site-packages/libtpu/libtpu.so',
        '/lib/libtpu.so',
        '/usr/lib/libtpu.so'
    ]
    
    found = False
    for path in libtpu_paths:
        if os.path.exists(path):
            print(f"  ✓ Found libtpu.so at: {path}")
            found = True
            break
    
    if not found:
        print("  ✗ libtpu.so not found in common locations")
        print("  This may be why TPU detection fails")
    
except Exception as e:
    print(f"  ✗ Error checking libtpu: {e}")

print()

# Test 4: Try PJRT directly
print("Test 4: Testing PJRT device access...")
try:
    # Try to use PJRT to get devices
    import subprocess
    result = subprocess.run(
        ['python3', '-c', 
         'import os; os.environ["JAX_PLATFORMS"]=""; os.environ["PJRT_DEVICE"]="TPU"; '
         'import jax; print("Devices:", jax.devices())'],
        capture_output=True,
        text=True,
        timeout=30
    )
    print(f"  Output: {result.stdout}")
    if result.stderr:
        print(f"  Errors: {result.stderr}")
except Exception as e:
    print(f"  ✗ Error: {e}")

print()
print("=" * 80)
print("Diagnostic Summary")
print("=" * 80)
print()
print("If TPU is still not detected:")
print("  1. Wait 10-15 minutes after TPU creation")
print("  2. Try using v2-alpha-tpuv6e runtime version")
print("  3. Restart TPU VM")
print("  4. Check TPU health: gcloud compute tpus tpu-vm describe <name> --zone=<zone>")
print("  5. Consider using v4 TPU instead (more reliable)")
print()


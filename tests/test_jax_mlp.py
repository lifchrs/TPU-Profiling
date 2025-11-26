#!/usr/bin/env python3
"""
Simple MLP test on TPU using JAX
This is a minimal test to verify TPU access and basic functionality
"""

import jax
import jax.numpy as jnp
from jax import random
import time

def create_mlp(key, layer_sizes):
    """Create a simple MLP with given layer sizes"""
    keys = random.split(key, len(layer_sizes) - 1)
    params = []
    
    for i in range(len(layer_sizes) - 1):
        w_key, b_key = random.split(keys[i])
        w = random.normal(w_key, (layer_sizes[i], layer_sizes[i+1])) * 0.1
        b = random.normal(b_key, (layer_sizes[i+1],)) * 0.1
        params.append({'weights': w, 'bias': b})
    
    return params

def forward(params, x):
    """Forward pass through MLP"""
    for i, layer in enumerate(params):
        x = jnp.dot(x, layer['weights']) + layer['bias']
        if i < len(params) - 1:  # No activation on last layer
            x = jax.nn.relu(x)
    return x

def loss_fn(params, x, y):
    """Mean squared error loss"""
    pred = forward(params, x)
    return jnp.mean((pred - y) ** 2)

@jax.jit
def train_step(params, x, y, learning_rate=0.01):
    """Single training step"""
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    # Simple gradient descent update
    new_params = []
    for param, grad in zip(params, grads):
        new_params.append({
            'weights': param['weights'] - learning_rate * grad['weights'],
            'bias': param['bias'] - learning_rate * grad['bias']
        })
    return new_params, loss

def main():
    print("=" * 60)
    print("JAX MLP Test on TPU")
    print("=" * 60)
    
    # Check available devices
    print("\n1. Checking available devices...")
    devices = jax.devices()
    print(f"   Found {len(devices)} device(s):")
    for i, device in enumerate(devices):
        print(f"   Device {i}: {device}")
        print(f"      Platform: {device.platform}")
        print(f"      Device ID: {device.id}")
    
    # Check if TPU is available
    tpu_devices = [d for d in devices if d.platform == 'tpu']
    if tpu_devices:
        print(f"\n   ✅ TPU devices found: {len(tpu_devices)}")
    else:
        print(f"\n   ⚠️  No TPU devices found. Using: {devices[0].platform}")
    
    # Create model
    print("\n2. Creating MLP model...")
    key = random.PRNGKey(42)
    layer_sizes = [784, 256, 128, 10]  # Simple MNIST-like architecture
    params = create_mlp(key, layer_sizes)
    print(f"   Model architecture: {' -> '.join(map(str, layer_sizes))}")
    
    # Create dummy data
    print("\n3. Creating dummy data...")
    batch_size = 32
    x = random.normal(key, (batch_size, layer_sizes[0]))
    y = random.normal(key, (batch_size, layer_sizes[-1]))
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {y.shape}")
    
    # Test forward pass
    print("\n4. Testing forward pass...")
    start_time = time.time()
    output = forward(params, x)
    forward_time = time.time() - start_time
    print(f"   Output shape: {output.shape}")
    print(f"   Forward pass time: {forward_time:.4f}s")
    
    # Test training step
    print("\n5. Testing training step...")
    start_time = time.time()
    new_params, loss = train_step(params, x, y)
    train_time = time.time() - start_time
    print(f"   Initial loss: {loss:.6f}")
    print(f"   Training step time: {train_time:.4f}s")
    
    # Run a few training iterations
    print("\n6. Running 10 training iterations...")
    current_params = params
    start_time = time.time()
    for i in range(10):
        current_params, loss = train_step(current_params, x, y)
        if (i + 1) % 5 == 0:
            print(f"   Iteration {i+1}: loss = {loss:.6f}")
    total_time = time.time() - start_time
    print(f"   Total time for 10 iterations: {total_time:.4f}s")
    print(f"   Average time per iteration: {total_time/10:.4f}s")
    
    print("\n" + "=" * 60)
    print("✅ MLP test completed successfully!")
    print("=" * 60)
    print("\nNext steps:")
    print("  - If this works, you can proceed to more complex models")
    print("  - Try increasing batch size and model size")
    print("  - Test with actual data (e.g., MNIST)")

if __name__ == '__main__':
    main()




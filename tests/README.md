# Test Scripts

This directory contains test scripts to verify TPU setup and functionality.

## test_jax_mlp.py

A simple Multi-Layer Perceptron (MLP) test using JAX to verify:
- TPU device access
- Basic JAX operations on TPU
- Forward pass and training step
- Performance benchmarking

### Usage

On your TPU VM:

```bash
# Make sure JAX is installed
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Or for TPU specifically:
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/jax_tpu_releases.html

# Run the test
python3 tests/test_jax_mlp.py
```

### Expected Output

You should see:
- List of available TPU devices
- Model creation confirmation
- Forward pass timing
- Training step timing
- Success message

### Troubleshooting

- **No TPU devices found**: Make sure you're running on a TPU VM, not a regular VM
- **JAX installation fails**: Check TPU VM has internet access
- **Import errors**: Verify JAX is installed for TPU (not CPU/GPU version)




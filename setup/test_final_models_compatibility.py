#!/usr/bin/env python3
"""
Quick compatibility test for the 3 final models with TPU

Tests:
- meta-llama/Llama-3.1-8B-Instruct
- Qwen/Qwen3-4B
- Qwen/Qwen3-32B

Tests with TP=1 first, then tries higher TP sizes if successful.
Run this inside the vLLM Docker container.
"""

import os
import sys
import time
from datetime import datetime

# Set environment variables BEFORE any imports
os.environ['JAX_PLATFORMS'] = ''
os.environ['PJRT_DEVICE'] = 'TPU'
os.environ['TMPDIR'] = '/dev/shm'
os.environ['TEMP'] = '/dev/shm'
os.environ['TMP'] = '/dev/shm'

try:
    from vllm import LLM, SamplingParams
except ImportError as e:
    print(f"✗ Failed to import vLLM: {e}")
    print("  Make sure you're running this inside the vLLM Docker container")
    sys.exit(1)


# Models to test
MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-32B"
]

# TP sizes to test (in order)
TP_SIZES = [1, 2, 4, 8]


def test_model(model_name: str, tp_size: int, max_model_len: int = 2048) -> dict:
    """Test a single model with a specific TP size"""
    print(f"\n  Testing TP={tp_size}...", end=" ", flush=True)
    
    try:
        # Load model
        start_time = time.time()
        llm = LLM(
            model=model_name,
            tensor_parallel_size=tp_size,
            dtype="bfloat16",
            max_model_len=max_model_len,
            disable_log_stats=True,
            trust_remote_code=True
        )
        load_time = time.time() - start_time
        
        # Test generation
        sampling_params = SamplingParams(max_tokens=10, temperature=0.7)
        outputs = llm.generate(["Hello, how are you?"], sampling_params=sampling_params)
        generated_text = outputs[0].outputs[0].text
        
        print(f"✓ SUCCESS (loaded in {load_time:.2f}s)")
        return {
            'status': 'SUCCESS',
            'tp_size': tp_size,
            'load_time': load_time,
            'generated_text': generated_text[:50]  # First 50 chars
        }
        
    except Exception as e:
        print(f"✗ FAILED: {str(e)[:100]}")
        return {
            'status': 'FAILED',
            'tp_size': tp_size,
            'error': str(e)
        }


def main():
    print("=" * 80)
    print("Final Models TPU Compatibility Test")
    print("=" * 80)
    print(f"\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nModels to test: {len(MODELS)}")
    for i, model in enumerate(MODELS, 1):
        print(f"  {i}. {model}")
    print(f"\nTP sizes to test: {TP_SIZES}")
    print()
    
    results = {}
    
    for model_idx, model_name in enumerate(MODELS, 1):
        print("\n" + "=" * 80)
        print(f"Model {model_idx}/{len(MODELS)}: {model_name}")
        print("=" * 80)
        
        model_results = {
            'model': model_name,
            'tests': []
        }
        
        # Test each TP size
        for tp_size in TP_SIZES:
            result = test_model(model_name, tp_size)
            model_results['tests'].append(result)
            
            # If failed, stop testing higher TP sizes for this model
            if result['status'] == 'FAILED':
                print(f"    Stopping TP tests for {model_name} after TP={tp_size} failure")
                break
        
        results[model_name] = model_results
        
        # Print summary for this model
        successful_tps = [t['tp_size'] for t in model_results['tests'] if t['status'] == 'SUCCESS']
        if successful_tps:
            print(f"\n  ✓ {model_name} works with TP sizes: {successful_tps}")
        else:
            print(f"\n  ✗ {model_name} failed all TP size tests")
    
    # Print final summary
    print("\n\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    
    for model_name, model_results in results.items():
        successful_tps = [t['tp_size'] for t in model_results['tests'] if t['status'] == 'SUCCESS']
        failed_tps = [t['tp_size'] for t in model_results['tests'] if t['status'] == 'FAILED']
        
        print(f"\n{model_name}:")
        if successful_tps:
            print(f"  ✓ Working TP sizes: {successful_tps}")
        if failed_tps:
            print(f"  ✗ Failed TP sizes: {failed_tps}")
    
    print("\n" + "=" * 80)
    print("Compatibility test complete!")
    print("=" * 80)
    
    # Return exit code based on results
    all_failed = all(
        all(t['status'] == 'FAILED' for t in model_results['tests'])
        for model_results in results.values()
    )
    
    if all_failed:
        print("\n⚠ WARNING: All models failed! Check TPU connectivity and setup.")
        sys.exit(1)
    else:
        print("\n✓ At least one model works. Proceed with batch measurements.")
        sys.exit(0)


if __name__ == '__main__':
    main()


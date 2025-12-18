#!/usr/bin/env python3
"""
Test Qwen/Qwen3-4B individually with different TP sizes
Tests TP=1, TP=2, TP=4, TP=8 one at a time with proper cleanup

Run this inside the vLLM Docker container
"""

import os
import sys
import time
import gc
import json
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


MODEL_NAME = "Qwen/Qwen3-4B"
TP_SIZES = [1, 2, 4, 8]
RESULTS_DIR = "results"
TEST_PROMPT = "Hello, how are you?"
NUM_TEST_TOKENS = 10


def test_tp_size(tp_size: int) -> dict:
    """Test model with a specific TP size"""
    print("\n" + "=" * 80)
    print(f"Testing {MODEL_NAME} with TP={tp_size}")
    print("=" * 80)
    print()
    
    result = {
        'model': MODEL_NAME,
        'tp_size': tp_size,
        'status': 'UNKNOWN',
        'timestamp': datetime.now().isoformat()
    }
    
    llm = None
    
    try:
        # Load model
        print(f"Loading model with TP={tp_size}...")
        print("(This may take 2-3 minutes)")
        print()
        
        load_start = time.time()
        llm = LLM(
            model=MODEL_NAME,
            tensor_parallel_size=tp_size,
            dtype="bfloat16",
            max_model_len=2048,
            disable_log_stats=True,
            trust_remote_code=True
        )
        load_time = time.time() - load_start
        
        print(f"✓ Model loaded successfully in {load_time:.2f}s")
        result['load_time_seconds'] = load_time
        
        # Test generation
        print()
        print("Testing generation...")
        sampling_params = SamplingParams(max_tokens=NUM_TEST_TOKENS, temperature=0.7)
        
        gen_start = time.time()
        outputs = llm.generate([TEST_PROMPT], sampling_params=sampling_params)
        gen_time = time.time() - gen_start
        
        generated_text = outputs[0].outputs[0].text
        num_tokens = len(outputs[0].outputs[0].token_ids)
        
        print(f"✓ Generation successful in {gen_time:.2f}s")
        print(f"  Generated {num_tokens} tokens")
        print(f"  Output: {generated_text[:100]}")
        
        result['status'] = 'SUCCESS'
        result['generation_time_seconds'] = gen_time
        result['num_tokens_generated'] = num_tokens
        result['generated_text'] = generated_text[:200]  # First 200 chars
        
        print()
        print("=" * 80)
        print(f"✓ SUCCESS: {MODEL_NAME} works with TP={tp_size}!")
        print("=" * 80)
        
    except Exception as e:
        print()
        print("=" * 80)
        print(f"✗ FAILED: {MODEL_NAME} with TP={tp_size}")
        print("=" * 80)
        print(f"\nError: {e}")
        print()
        
        result['status'] = 'FAILED'
        result['error'] = str(e)
        
        # Print diagnostic info
        error_str = str(e)
        if "Device or resource busy" in error_str:
            print("→ Device busy error - TPU devices may still be in use")
        elif "AttributeError" in error_str and "coords" in error_str:
            print("→ Device coordinate error - JAX can't access TPU device info")
        elif "TPU initialization failed" in error_str:
            print("→ TPU initialization failed - devices may not be available")
        
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
    
    finally:
        # Cleanup
        if llm is not None:
            print()
            print("Cleaning up model instance...")
            del llm
            gc.collect()
            print("✓ Cleanup complete")
            
            # Wait a bit to ensure resources are released
            print("Waiting 5 seconds for resource cleanup...")
            time.sleep(5)
    
    return result


def main():
    print("=" * 80)
    print("Individual TP Size Test for Qwen/Qwen3-4B")
    print("=" * 80)
    print(f"\nModel: {MODEL_NAME}")
    print(f"TP sizes to test: {TP_SIZES}")
    print(f"Test prompt: {TEST_PROMPT}")
    print(f"Number of tokens: {NUM_TEST_TOKENS}")
    print(f"\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("Note: Each test will be run individually with cleanup between tests")
    print()
    
    # Create results directory
    import pathlib
    results_path = pathlib.Path(RESULTS_DIR)
    results_path.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    # Test each TP size individually
    for tp_size in TP_SIZES:
        result = test_tp_size(tp_size)
        all_results.append(result)
        
        # Save individual result
        result_file = results_path / f"qwen3_4b_tp{tp_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n✓ Result saved to: {result_file}")
        
        # If failed, ask if we should continue
        if result['status'] == 'FAILED':
            print()
            print("⚠ Test failed. Options:")
            print("  1. Continue to next TP size (recommended)")
            print("  2. Exit and investigate")
            print()
            print("Continuing to next TP size in 3 seconds...")
            time.sleep(3)
    
    # Print final summary
    print("\n\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print()
    
    successful = [r for r in all_results if r['status'] == 'SUCCESS']
    failed = [r for r in all_results if r['status'] == 'FAILED']
    
    print(f"Total tests: {len(all_results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print()
    
    if successful:
        print("✓ Successful TP sizes:")
        for r in successful:
            load_time = r.get('load_time_seconds', 0)
            print(f"  • TP={r['tp_size']}: Loaded in {load_time:.2f}s")
    
    if failed:
        print("\n✗ Failed TP sizes:")
        for r in failed:
            error = r.get('error', 'Unknown error')
            print(f"  • TP={r['tp_size']}: {error[:80]}")
    
    # Save summary
    summary_file = results_path / f"qwen3_4b_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    summary = {
        'model': MODEL_NAME,
        'timestamp': datetime.now().isoformat(),
        'results': all_results,
        'summary': {
            'total': len(all_results),
            'successful': len(successful),
            'failed': len(failed)
        }
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print()
    print(f"✓ Summary saved to: {summary_file}")
    print()
    print("=" * 80)
    print("Testing complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()


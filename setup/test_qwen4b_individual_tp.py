#!/usr/bin/env python3
"""
Test Qwen/Qwen3-4B with individual TP sizes (1, 2, 4, 8)
Each test runs independently with proper cleanup

Usage:
    python3 test_qwen4b_individual_tp.py 1    # Test TP=1 only
    python3 test_qwen4b_individual_tp.py 2    # Test TP=2 only
    python3 test_qwen4b_individual_tp.py 4    # Test TP=4 only
    python3 test_qwen4b_individual_tp.py 8    # Test TP=8 only
    python3 test_qwen4b_individual_tp.py all   # Test all TP sizes sequentially

Run this inside the vLLM Docker container
"""

import os
import sys
import time
import gc
import argparse
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


def test_tp_size(tp_size: int, max_model_len: int = 2048) -> dict:
    """Test model with a specific TP size"""
    print("=" * 80)
    print(f"Testing {MODEL_NAME} with TP={tp_size}")
    print("=" * 80)
    print(f"\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # Check JAX devices before loading
        print("Checking JAX devices...")
        import jax
        try:
            devices = jax.devices()
            print(f"  ✓ Found {len(devices)} JAX devices")
            if len(devices) < tp_size:
                print(f"  ⚠ Warning: Only {len(devices)} devices available, but TP={tp_size} requested")
        except Exception as e:
            print(f"  ⚠ Could not enumerate devices: {e}")
        print()
        
        # Load model
        print(f"Loading model with TP={tp_size}...")
        print("(This may take 2-3 minutes)")
        print()
        
        start_time = time.time()
        llm = LLM(
            model=MODEL_NAME,
            tensor_parallel_size=tp_size,
            dtype="bfloat16",
            max_model_len=max_model_len,
            disable_log_stats=True,
            trust_remote_code=True
        )
        load_time = time.time() - start_time
        
        print(f"✓ Model loaded in {load_time:.2f}s")
        print()
        
        # Test generation
        print("Testing generation...")
        sampling_params = SamplingParams(max_tokens=20, temperature=0.7)
        gen_start = time.time()
        outputs = llm.generate(["Hello, how are you? Please explain machine learning briefly."], sampling_params=sampling_params)
        gen_time = time.time() - gen_start
        
        generated_text = outputs[0].outputs[0].text
        num_tokens = len(outputs[0].outputs[0].token_ids)
        
        print(f"✓ Generation successful!")
        print(f"  Generated {num_tokens} tokens in {gen_time:.2f}s")
        print(f"  Output: {generated_text[:150]}...")
        print()
        
        result = {
            'status': 'SUCCESS',
            'tp_size': tp_size,
            'load_time': load_time,
            'gen_time': gen_time,
            'num_tokens': num_tokens,
            'generated_text': generated_text[:200]
        }
        
        # Cleanup
        print("Cleaning up...")
        del llm
        gc.collect()
        print("✓ Cleanup complete")
        print()
        
        # Wait a bit to ensure resources are released
        print("Waiting 5 seconds for resource cleanup...")
        time.sleep(5)
        print()
        
        return result
        
    except Exception as e:
        print()
        print("=" * 80)
        print("✗ FAILED")
        print("=" * 80)
        print(f"\nError: {e}")
        print()
        print("Full traceback:")
        import traceback
        traceback.print_exc()
        print()
        
        return {
            'status': 'FAILED',
            'tp_size': tp_size,
            'error': str(e)
        }


def main():
    parser = argparse.ArgumentParser(
        description='Test Qwen/Qwen3-4B with individual TP sizes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test TP=1 only
  python3 test_qwen4b_individual_tp.py 1
  
  # Test TP=4 only
  python3 test_qwen4b_individual_tp.py 4
  
  # Test all TP sizes sequentially
  python3 test_qwen4b_individual_tp.py all
        """
    )
    
    parser.add_argument('tp_size', type=str, nargs='?', default='all',
                       help='TP size to test (1, 2, 4, 8, or "all" for sequential testing)')
    
    args = parser.parse_args()
    
    # Determine which TP sizes to test
    if args.tp_size.lower() == 'all':
        tp_sizes_to_test = TP_SIZES
        print("=" * 80)
        print("Testing Qwen/Qwen3-4B with ALL TP sizes sequentially")
        print("=" * 80)
        print(f"\nWill test: {TP_SIZES}")
        print("Each test will run independently with cleanup between tests.")
        print()
    else:
        try:
            tp_size = int(args.tp_size)
            if tp_size not in TP_SIZES:
                print(f"✗ Invalid TP size: {tp_size}")
                print(f"  Valid options: {TP_SIZES} or 'all'")
                sys.exit(1)
            tp_sizes_to_test = [tp_size]
        except ValueError:
            print(f"✗ Invalid argument: {args.tp_size}")
            print(f"  Must be one of: {TP_SIZES} or 'all'")
            sys.exit(1)
    
    # Run tests
    results = {}
    
    for tp_size in tp_sizes_to_test:
        result = test_tp_size(tp_size)
        results[tp_size] = result
        
        if result['status'] == 'SUCCESS':
            print("=" * 80)
            print(f"✓ SUCCESS: TP={tp_size} works!")
            print("=" * 80)
        else:
            print("=" * 80)
            print(f"✗ FAILED: TP={tp_size}")
            print("=" * 80)
            
            # If testing all and one fails, ask if user wants to continue
            if len(tp_sizes_to_test) > 1:
                print()
                print("Continue with next TP size? (This test failed, but you can continue)")
                print("Press Ctrl+C to stop, or wait 10 seconds to continue...")
                try:
                    time.sleep(10)
                except KeyboardInterrupt:
                    print("\nStopped by user")
                    break
        
        print()
        print()
    
    # Print summary
    print("=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print()
    
    successful = [tp for tp, r in results.items() if r['status'] == 'SUCCESS']
    failed = [tp for tp, r in results.items() if r['status'] == 'FAILED']
    
    if successful:
        print(f"✓ Successful TP sizes ({len(successful)}):")
        for tp in successful:
            result = results[tp]
            print(f"  • TP={tp}: Loaded in {result['load_time']:.2f}s, Generated {result['num_tokens']} tokens in {result['gen_time']:.2f}s")
        print()
    
    if failed:
        print(f"✗ Failed TP sizes ({len(failed)}):")
        for tp in failed:
            result = results[tp]
            error = result.get('error', 'Unknown error')
            print(f"  • TP={tp}: {error[:100]}")
        print()
    
    # Exit code
    if failed:
        sys.exit(1)
    else:
        print("✓ All tests passed!")
        sys.exit(0)


if __name__ == '__main__':
    main()


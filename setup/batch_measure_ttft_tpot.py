#!/usr/bin/env python3
"""
Batch measurement of TTFT and TPOT across multiple models and configurations

This script tests multiple models with different tensor parallelism sizes
and collects TTFT/TPOT metrics for comparison.

Usage:
    python3 batch_measure_ttft_tpot.py --config configs/batch_test_config.json
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime
from pathlib import Path

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
    sys.exit(1)

# Import the measurement function
# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from measure_ttft_tpot import measure_ttft_tpot


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        return json.load(f)


def run_batch_measurements(config: dict) -> list:
    """
    Run batch measurements based on configuration
    
    Config format:
    {
        "models": [
            {
                "name": "meta-llama/Llama-3.1-8B-Instruct",
                "tp_sizes": [1, 2, 4],
                "max_model_len": 2048
            }
        ],
        "test_config": {
            "prompt": "Hello, how are you?",
            "num_tokens": 50,
            "temperature": 0.7,
            "top_p": 0.95,
            "num_runs": 1,
            "warmup_runs": 0
        },
        "output_dir": "results"
    }
    """
    results = []
    output_dir = Path(config.get('output_dir', 'results'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    test_config = config.get('test_config', {})
    models = config.get('models', [])
    
    print("=" * 80)
    print("Batch TTFT/TPOT Measurement")
    print("=" * 80)
    print(f"\nModels to test: {len(models)}")
    print(f"Output directory: {output_dir}")
    print(f"\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    for model_idx, model_config in enumerate(models):
        model_name = model_config['name']
        tp_sizes = model_config.get('tp_sizes', [1])
        max_model_len = model_config.get('max_model_len', 2048)
        
        print(f"\n{'#'*80}")
        print(f"# Model {model_idx + 1}/{len(models)}: {model_name}")
        print(f"{'#'*80}\n")
        
        for tp_size in tp_sizes:
            print(f"\n{'='*80}")
            print(f"Testing: {model_name} with TP={tp_size}")
            print(f"{'='*80}\n")
            
            try:
                # Load model
                print(f"Loading model with TP={tp_size}...")
                load_start = time.time()
                llm = LLM(
                    model=model_name,
                    tensor_parallel_size=tp_size,
                    dtype="bfloat16",
                    max_model_len=max_model_len,
                    disable_log_stats=True,
                    trust_remote_code=True
                )
                load_time = time.time() - load_start
                print(f"✓ Model loaded in {load_time:.2f}s\n")
                
                # Measure
                result = measure_ttft_tpot(
                    llm=llm,
                    prompt=test_config.get('prompt', "Hello, how are you?"),
                    num_tokens=test_config.get('num_tokens', 50),
                    temperature=test_config.get('temperature', 0.7),
                    top_p=test_config.get('top_p', 0.95),
                    num_runs=test_config.get('num_runs', 1),
                    warmup_runs=test_config.get('warmup_runs', 0)
                )
                
                # Add metadata
                result['model'] = model_name
                result['tp_size'] = tp_size
                result['max_model_len'] = max_model_len
                result['load_time_seconds'] = load_time
                result['status'] = 'SUCCESS'
                
                results.append(result)
                
                # Print summary
                print(f"\n✓ SUCCESS:")
                print(f"  TTFT: {result['ttft_mean_ms']:.2f} ms")
                print(f"  TPOT: {result['tpot_mean_ms']:.2f} ms/token")
                print(f"  Throughput: {result['throughput_tokens_per_sec']:.2f} tokens/sec")
                
                # Save individual result
                safe_model_name = model_name.replace('/', '_').replace('-', '_')
                output_file = output_dir / f"{safe_model_name}_TP{tp_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(output_file, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"  Saved to: {output_file}")
                
            except Exception as e:
                print(f"\n✗ FAILED: {e}")
                error_result = {
                    'model': model_name,
                    'tp_size': tp_size,
                    'status': 'FAILED',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
                results.append(error_result)
                
                # Save error result
                safe_model_name = model_name.replace('/', '_').replace('-', '_')
                error_file = output_dir / f"{safe_model_name}_TP{tp_size}_ERROR_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(error_file, 'w') as f:
                    json.dump(error_result, f, indent=2)
    
    return results


def print_summary(results: list):
    """Print summary of all results"""
    print("\n\n" + "=" * 80)
    print("BATCH MEASUREMENT SUMMARY")
    print("=" * 80)
    
    successful = [r for r in results if r.get('status') == 'SUCCESS']
    failed = [r for r in results if r.get('status') == 'FAILED']
    
    print(f"\nTotal configurations tested: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        print("\n" + "-" * 80)
        print("SUCCESSFUL MEASUREMENTS")
        print("-" * 80)
        print(f"{'Model':<40} {'TP':<5} {'TTFT (ms)':<12} {'TPOT (ms/tok)':<15} {'Throughput (tok/s)':<18}")
        print("-" * 80)
        
        for r in successful:
            model_short = r['model'].split('/')[-1][:38]
            print(f"{model_short:<40} {r['tp_size']:<5} {r['ttft_mean_ms']:<12.2f} {r['tpot_mean_ms']:<15.2f} {r['throughput_tokens_per_sec']:<18.2f}")
    
    if failed:
        print("\n" + "-" * 80)
        print("FAILED MEASUREMENTS")
        print("-" * 80)
        for r in failed:
            print(f"  {r['model']} (TP={r['tp_size']}): {r.get('error', 'Unknown error')}")


def main():
    parser = argparse.ArgumentParser(
        description='Batch measurement of TTFT and TPOT across multiple models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example config file (config.json):
{
    "models": [
        {
            "name": "meta-llama/Llama-3.1-8B-Instruct",
            "tp_sizes": [1, 2, 4],
            "max_model_len": 2048
        },
        {
            "name": "Qwen/Qwen3-4B",
            "tp_sizes": [1, 2],
            "max_model_len": 2048
        }
    ],
    "test_config": {
        "prompt": "Explain quantum computing in simple terms.",
        "num_tokens": 100,
        "temperature": 0.7,
        "top_p": 0.95,
        "num_runs": 3,
        "warmup_runs": 1
    },
    "output_dir": "results"
}
        """
    )
    
    parser.add_argument('--config', type=str, required=True,
                       help='Path to JSON configuration file')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for summary JSON (default: results/summary_TIMESTAMP.json)')
    
    args = parser.parse_args()
    
    # Load config
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"✗ Failed to load config: {e}")
        sys.exit(1)
    
    # Run measurements
    results = run_batch_measurements(config)
    
    # Print summary
    print_summary(results)
    
    # Save summary
    output_dir = Path(config.get('output_dir', 'results'))
    if args.output:
        summary_file = Path(args.output)
    else:
        summary_file = output_dir / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'results': results,
        'summary': {
            'total': len(results),
            'successful': len([r for r in results if r.get('status') == 'SUCCESS']),
            'failed': len([r for r in results if r.get('status') == 'FAILED'])
        }
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Summary saved to: {summary_file}")
    print("\n" + "=" * 80)
    print("Batch measurement complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()


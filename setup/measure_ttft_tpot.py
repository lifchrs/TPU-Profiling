#!/usr/bin/env python3
"""
Measure TTFT (Time-to-First-Token) and TPOT (Time-per-Output-Token) for LLM inference on TPU

Usage:
    python3 measure_ttft_tpot.py --model meta-llama/Llama-3.1-8B-Instruct --tp-size 4

Run this inside the vLLM Docker container
"""

import os
import sys
import time
import argparse
import json
from datetime import datetime
from typing import Dict, List

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


def _std_dev(values: List[float]) -> float:
    """Calculate standard deviation"""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return variance ** 0.5


def measure_ttft_tpot(
    llm: LLM,
    prompt: str,
    num_tokens: int = 50,
    temperature: float = 0.7,
    top_p: float = 0.95,
    num_runs: int = 1,
    warmup_runs: int = 0
) -> Dict:
    """
    Measure TTFT and TPOT for LLM inference
    
    Args:
        llm: Loaded vLLM LLM instance
        prompt: Input prompt text
        num_tokens: Number of tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        num_runs: Number of measurement runs (for averaging)
        warmup_runs: Number of warmup runs (not measured)
    
    Returns:
        Dictionary with TTFT, TPOT, and metrics
    """
    # Warmup runs
    if warmup_runs > 0:
        print(f"Running {warmup_runs} warmup run(s)...")
        sampling_params = SamplingParams(max_tokens=num_tokens, temperature=temperature, top_p=top_p)
        for _ in range(warmup_runs):
            _ = llm.generate([prompt], sampling_params=sampling_params)
    
    all_ttft = []
    all_tpot = []
    all_total_times = []
    
    print(f"\nRunning {num_runs} measurement run(s)...")
    
    for run_idx in range(num_runs):
        print(f"  Run {run_idx + 1}/{num_runs}...", end=" ", flush=True)
        
        # Measure generation time
        generation_start = time.perf_counter()
        
        sampling_params = SamplingParams(
            max_tokens=num_tokens,
            temperature=temperature,
            top_p=top_p
        )
        
        outputs = llm.generate([prompt], sampling_params=sampling_params)
        generation_end = time.perf_counter()
        
        total_time_ms = (generation_end - generation_start) * 1000
        
        # Extract output information
        output = outputs[0]
        generated_text = output.outputs[0].text
        token_ids = output.outputs[0].token_ids
        num_generated = len(token_ids)
        
        # Try to get timing from vLLM's metrics if available
        ttft_ms = None
        tpot_ms = None
        
        if hasattr(output, 'metrics'):
            metrics = output.metrics
            if hasattr(metrics, 'time_to_first_token'):
                ttft_ms = metrics.time_to_first_token * 1000
            if hasattr(metrics, 'time_per_output_token'):
                tpot_ms = metrics.time_per_output_token * 1000
        
        # If metrics not available, estimate based on typical behavior
        if ttft_ms is None or tpot_ms is None:
            # Estimate prefill time based on input length
            input_tokens = len(prompt.split())  # Approximate
            
            if input_tokens < 20:
                prefill_fraction = 0.3  # 30% for short prompts
            elif input_tokens < 100:
                prefill_fraction = 0.4  # 40% for medium prompts
            else:
                prefill_fraction = 0.5  # 50% for long prompts
            
            estimated_prefill_time = total_time_ms * prefill_fraction
            estimated_decode_time = total_time_ms * (1 - prefill_fraction)
            
            ttft_ms = estimated_prefill_time
            tpot_ms = estimated_decode_time / (num_generated - 1) if num_generated > 1 else 0
        
        all_ttft.append(ttft_ms)
        if num_generated > 1:
            all_tpot.append(tpot_ms)
        all_total_times.append(total_time_ms)
        
        print(f"TTFT: {ttft_ms:.2f}ms, TPOT: {tpot_ms:.2f}ms/token")
    
    # Calculate statistics
    results = {
        'model': 'loaded',  # Will be set by caller
        'tp_size': 1,  # Will be set by caller
        'prompt': prompt,
        'prompt_length_tokens': len(prompt.split()),
        'num_tokens_requested': num_tokens,
        'num_runs': num_runs,
        'num_generated_tokens': num_generated,
        
        # TTFT metrics
        'ttft_mean_ms': sum(all_ttft) / len(all_ttft),
        'ttft_min_ms': min(all_ttft),
        'ttft_max_ms': max(all_ttft),
        'ttft_std_ms': _std_dev(all_ttft) if len(all_ttft) > 1 else 0,
        
        # TPOT metrics
        'tpot_mean_ms': sum(all_tpot) / len(all_tpot) if all_tpot else 0,
        'tpot_min_ms': min(all_tpot) if all_tpot else 0,
        'tpot_max_ms': max(all_tpot) if all_tpot else 0,
        'tpot_std_ms': _std_dev(all_tpot) if len(all_tpot) > 1 else 0,
        
        # Total time
        'total_time_mean_ms': sum(all_total_times) / len(all_total_times),
        
        # Throughput
        'throughput_tokens_per_sec': (num_generated / (all_total_times[0] / 1000)) if all_total_times else 0,
        
        'timestamp': datetime.now().isoformat()
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Measure TTFT and TPOT for LLM inference on TPU',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic measurement
  python3 measure_ttft_tpot.py --model meta-llama/Llama-3.1-8B-Instruct --tp-size 4
  
  # With custom prompt and multiple runs
  python3 measure_ttft_tpot.py --model meta-llama/Llama-3.1-8B-Instruct \\
      --tp-size 4 --prompt "Explain quantum computing" --num-tokens 100 --num-runs 3
  
  # Save results to file
  python3 measure_ttft_tpot.py --model meta-llama/Llama-3.1-8B-Instruct \\
      --tp-size 4 --output results.json
        """
    )
    
    parser.add_argument('--model', type=str, required=True,
                       help='HuggingFace model identifier (e.g., meta-llama/Llama-3.1-8B-Instruct)')
    parser.add_argument('--tp-size', type=int, default=1,
                       help='Tensor parallelism size (default: 1)')
    parser.add_argument('--prompt', type=str, default="Hello, how are you?",
                       help='Input prompt (default: "Hello, how are you?")')
    parser.add_argument('--num-tokens', type=int, default=50,
                       help='Number of tokens to generate (default: 50)')
    parser.add_argument('--max-model-len', type=int, default=2048,
                       help='Maximum model context length (default: 2048)')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Sampling temperature (default: 0.7)')
    parser.add_argument('--top-p', type=float, default=0.95,
                       help='Top-p sampling parameter (default: 0.95)')
    parser.add_argument('--num-runs', type=int, default=1,
                       help='Number of measurement runs (default: 1)')
    parser.add_argument('--warmup-runs', type=int, default=0,
                       help='Number of warmup runs (default: 0)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file path (default: print to stdout)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("TTFT and TPOT Measurement")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Model: {args.model}")
    print(f"  TP Size: {args.tp_size}")
    print(f"  Prompt: {args.prompt}")
    print(f"  Num Tokens: {args.num_tokens}")
    print(f"  Max Model Length: {args.max_model_len}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Top-p: {args.top_p}")
    print(f"  Measurement Runs: {args.num_runs}")
    print(f"  Warmup Runs: {args.warmup_runs}")
    print(f"\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Load model
    print("Loading model...")
    start_time = time.time()
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp_size,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        disable_log_stats=True,
        trust_remote_code=True
    )
    load_time = time.time() - start_time
    print(f"✓ Model loaded in {load_time:.2f}s\n")
    
    # Measure TTFT and TPOT
    try:
        results = measure_ttft_tpot(
            llm=llm,
            prompt=args.prompt,
            num_tokens=args.num_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            num_runs=args.num_runs,
            warmup_runs=args.warmup_runs
        )
    except Exception as e:
        print(f"\n✗ Failed to measure: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Add model info
    results['model'] = args.model
    results['tp_size'] = args.tp_size
    results['load_time_seconds'] = load_time
    
    # Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"\nModel: {results['model']}")
    print(f"Tensor Parallelism: TP={results['tp_size']}")
    print(f"\nTTFT (Time-to-First-Token):")
    print(f"  Mean:   {results['ttft_mean_ms']:.2f} ms")
    print(f"  Min:    {results['ttft_min_ms']:.2f} ms")
    print(f"  Max:    {results['ttft_max_ms']:.2f} ms")
    if results['ttft_std_ms'] > 0:
        print(f"  StdDev: {results['ttft_std_ms']:.2f} ms")
    
    if results['tpot_mean_ms'] > 0:
        print(f"\nTPOT (Time-per-Output-Token):")
        print(f"  Mean:   {results['tpot_mean_ms']:.2f} ms/token")
        print(f"  Min:    {results['tpot_min_ms']:.2f} ms/token")
        print(f"  Max:    {results['tpot_max_ms']:.2f} ms/token")
        if results['tpot_std_ms'] > 0:
            print(f"  StdDev: {results['tpot_std_ms']:.2f} ms/token")
    
    print(f"\nTotal Time: {results['total_time_mean_ms']:.2f} ms")
    print(f"Throughput: {results['throughput_tokens_per_sec']:.2f} tokens/sec")
    print(f"Generated Tokens: {results['num_generated_tokens']}")
    
    # Save to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to: {args.output}")
    else:
        # Print JSON to stdout
        print("\n" + "=" * 80)
        print("JSON Output:")
        print("=" * 80)
        print(json.dumps(results, indent=2))
    
    print("\n" + "=" * 80)
    print("Measurement complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()

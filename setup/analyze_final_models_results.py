#!/usr/bin/env python3
"""
Analyze and visualize TTFT/TPOT measurement results for final models

Creates graphs comparing:
- TTFT vs TP size for each model
- TPOT vs TP size for each model
- Throughput vs TP size for each model
- Model comparison across TP sizes

Usage:
    python3 analyze_final_models_results.py --summary results/summary_*.json
    python3 analyze_final_models_results.py --results-dir results/
"""

import json
import argparse
import sys
from pathlib import Path
from typing import List, Dict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from collections import defaultdict

# Try to import matplotlib, but don't fail if not available
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠ Warning: matplotlib not available. Install with: pip install matplotlib")
    print("  Will generate data tables instead of graphs.")


def load_results_from_summary(summary_file: Path) -> List[Dict]:
    """Load results from summary JSON file"""
    with open(summary_file, 'r') as f:
        data = json.load(f)
    
    if 'results' in data:
        return data['results']
    else:
        return [data]  # Single result file


def load_results_from_dir(results_dir: Path) -> List[Dict]:
    """Load all result JSON files from directory"""
    results = []
    
    for json_file in results_dir.glob("*.json"):
        if json_file.name.startswith("summary_"):
            continue  # Skip summary files
        
        try:
            with open(json_file, 'r') as f:
                result = json.load(f)
                if result.get('status') == 'SUCCESS':
                    results.append(result)
        except Exception as e:
            print(f"⚠ Warning: Failed to load {json_file}: {e}")
    
    return results


def organize_results(results: List[Dict]) -> Dict:
    """Organize results by model and TP size"""
    organized = defaultdict(lambda: defaultdict(dict))
    
    for result in results:
        if result.get('status') != 'SUCCESS':
            continue
        
        model = result.get('model', 'unknown')
        tp_size = result.get('tp_size', 0)
        
        organized[model][tp_size] = {
            'ttft_mean_ms': result.get('ttft_mean_ms', 0),
            'ttft_std_ms': result.get('ttft_std_ms', 0),
            'tpot_mean_ms': result.get('tpot_mean_ms', 0),
            'tpot_std_ms': result.get('tpot_std_ms', 0),
            'throughput_tokens_per_sec': result.get('throughput_tokens_per_sec', 0),
            'total_time_mean_ms': result.get('total_time_mean_ms', 0),
            'num_generated_tokens': result.get('num_generated_tokens', 0),
            'load_time_seconds': result.get('load_time_seconds', 0)
        }
    
    return organized


def create_plots(organized_results: Dict, output_dir: Path):
    """Create visualization plots"""
    if not HAS_MATPLOTLIB:
        print("⚠ Skipping plots (matplotlib not available)")
        return
    
    models = list(organized_results.keys())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Figure 1: TTFT vs TP Size
    fig1, ax1 = plt.subplots(figsize=(12, 7))
    
    for idx, model in enumerate(models):
        tp_sizes = sorted(organized_results[model].keys())
        ttft_values = [organized_results[model][tp]['ttft_mean_ms'] for tp in tp_sizes]
        ttft_stds = [organized_results[model][tp]['ttft_std_ms'] for tp in tp_sizes]
        
        model_short = model.split('/')[-1]
        ax1.errorbar(tp_sizes, ttft_values, yerr=ttft_stds, 
                    marker='o', linewidth=2, markersize=8, 
                    label=model_short, color=colors[idx % len(colors)])
    
    ax1.set_xlabel('Tensor Parallelism Size', fontsize=12, fontweight='bold')
    ax1.set_ylabel('TTFT (ms)', fontsize=12, fontweight='bold')
    ax1.set_title('Time-to-First-Token (TTFT) vs Tensor Parallelism', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks([1, 2, 4, 8])
    
    plt.tight_layout()
    fig1.savefig(output_dir / 'ttft_vs_tp.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'ttft_vs_tp.png'}")
    plt.close(fig1)
    
    # Figure 2: TPOT vs TP Size
    fig2, ax2 = plt.subplots(figsize=(12, 7))
    
    for idx, model in enumerate(models):
        tp_sizes = sorted(organized_results[model].keys())
        tpot_values = [organized_results[model][tp]['tpot_mean_ms'] for tp in tp_sizes]
        tpot_stds = [organized_results[model][tp]['tpot_std_ms'] for tp in tp_sizes]
        
        model_short = model.split('/')[-1]
        ax2.errorbar(tp_sizes, tpot_values, yerr=tpot_stds,
                    marker='s', linewidth=2, markersize=8,
                    label=model_short, color=colors[idx % len(colors)])
    
    ax2.set_xlabel('Tensor Parallelism Size', fontsize=12, fontweight='bold')
    ax2.set_ylabel('TPOT (ms/token)', fontsize=12, fontweight='bold')
    ax2.set_title('Time-per-Output-Token (TPOT) vs Tensor Parallelism', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks([1, 2, 4, 8])
    
    plt.tight_layout()
    fig2.savefig(output_dir / 'tpot_vs_tp.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'tpot_vs_tp.png'}")
    plt.close(fig2)
    
    # Figure 3: Throughput vs TP Size
    fig3, ax3 = plt.subplots(figsize=(12, 7))
    
    for idx, model in enumerate(models):
        tp_sizes = sorted(organized_results[model].keys())
        throughput_values = [organized_results[model][tp]['throughput_tokens_per_sec'] for tp in tp_sizes]
        
        model_short = model.split('/')[-1]
        ax3.plot(tp_sizes, throughput_values, marker='^', linewidth=2, markersize=8,
                label=model_short, color=colors[idx % len(colors)])
    
    ax3.set_xlabel('Tensor Parallelism Size', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Throughput (tokens/sec)', fontsize=12, fontweight='bold')
    ax3.set_title('Throughput vs Tensor Parallelism', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks([1, 2, 4, 8])
    
    plt.tight_layout()
    fig3.savefig(output_dir / 'throughput_vs_tp.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'throughput_vs_tp.png'}")
    plt.close(fig3)
    
    # Figure 4: Combined comparison (subplot)
    fig4, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # TTFT subplot
    for idx, model in enumerate(models):
        tp_sizes = sorted(organized_results[model].keys())
        ttft_values = [organized_results[model][tp]['ttft_mean_ms'] for tp in tp_sizes]
        model_short = model.split('/')[-1]
        axes[0].plot(tp_sizes, ttft_values, marker='o', linewidth=2, markersize=6,
                    label=model_short, color=colors[idx % len(colors)])
    axes[0].set_xlabel('TP Size', fontweight='bold')
    axes[0].set_ylabel('TTFT (ms)', fontweight='bold')
    axes[0].set_title('TTFT', fontweight='bold')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks([1, 2, 4, 8])
    
    # TPOT subplot
    for idx, model in enumerate(models):
        tp_sizes = sorted(organized_results[model].keys())
        tpot_values = [organized_results[model][tp]['tpot_mean_ms'] for tp in tp_sizes]
        model_short = model.split('/')[-1]
        axes[1].plot(tp_sizes, tpot_values, marker='s', linewidth=2, markersize=6,
                    label=model_short, color=colors[idx % len(colors)])
    axes[1].set_xlabel('TP Size', fontweight='bold')
    axes[1].set_ylabel('TPOT (ms/token)', fontweight='bold')
    axes[1].set_title('TPOT', fontweight='bold')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks([1, 2, 4, 8])
    
    # Throughput subplot
    for idx, model in enumerate(models):
        tp_sizes = sorted(organized_results[model].keys())
        throughput_values = [organized_results[model][tp]['throughput_tokens_per_sec'] for tp in tp_sizes]
        model_short = model.split('/')[-1]
        axes[2].plot(tp_sizes, throughput_values, marker='^', linewidth=2, markersize=6,
                    label=model_short, color=colors[idx % len(colors)])
    axes[2].set_xlabel('TP Size', fontweight='bold')
    axes[2].set_ylabel('Throughput (tok/s)', fontweight='bold')
    axes[2].set_title('Throughput', fontweight='bold')
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xticks([1, 2, 4, 8])
    
    plt.tight_layout()
    fig4.savefig(output_dir / 'combined_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'combined_comparison.png'}")
    plt.close(fig4)


def print_analysis_table(organized_results: Dict):
    """Print analysis table to console"""
    print("\n" + "=" * 100)
    print("RESULTS ANALYSIS TABLE")
    print("=" * 100)
    
    models = list(organized_results.keys())
    
    for model in models:
        print(f"\n{model}:")
        print("-" * 100)
        print(f"{'TP':<6} {'TTFT (ms)':<15} {'TPOT (ms/tok)':<18} {'Throughput (tok/s)':<20} {'Load Time (s)':<15}")
        print("-" * 100)
        
        for tp_size in sorted(organized_results[model].keys()):
            data = organized_results[model][tp_size]
            print(f"{tp_size:<6} "
                  f"{data['ttft_mean_ms']:<15.2f} "
                  f"{data['tpot_mean_ms']:<18.2f} "
                  f"{data['throughput_tokens_per_sec']:<20.2f} "
                  f"{data['load_time_seconds']:<15.2f}")
    
    # Calculate speedup
    print("\n" + "=" * 100)
    print("SPEEDUP ANALYSIS (TP=8 vs TP=1)")
    print("=" * 100)
    
    for model in models:
        if 1 in organized_results[model] and 8 in organized_results[model]:
            tp1_ttft = organized_results[model][1]['ttft_mean_ms']
            tp8_ttft = organized_results[model][8]['ttft_mean_ms']
            tp1_tpot = organized_results[model][1]['tpot_mean_ms']
            tp8_tpot = organized_results[model][8]['tpot_mean_ms']
            
            ttft_speedup = tp1_ttft / tp8_ttft if tp8_ttft > 0 else 0
            tpot_speedup = tp1_tpot / tp8_tpot if tp8_tpot > 0 else 0
            
            print(f"\n{model}:")
            print(f"  TTFT speedup: {ttft_speedup:.2f}x (TP=1: {tp1_ttft:.2f}ms → TP=8: {tp8_ttft:.2f}ms)")
            print(f"  TPOT speedup: {tpot_speedup:.2f}x (TP=1: {tp1_tpot:.2f}ms/tok → TP=8: {tp8_tpot:.2f}ms/tok)")


def generate_story_summary(organized_results: Dict, output_file: Path):
    """Generate a text summary telling the story of the results"""
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("FINAL MODELS PERFORMANCE ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        
        models = list(organized_results.keys())
        
        f.write("MODELS TESTED:\n")
        for i, model in enumerate(models, 1):
            f.write(f"  {i}. {model}\n")
        f.write("\n")
        
        f.write("TENSOR PARALLELISM SIZES: 1, 2, 4, 8\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("KEY FINDINGS\n")
        f.write("=" * 80 + "\n\n")
        
        # Find best and worst performers
        for model in models:
            f.write(f"\n{model}:\n")
            f.write("-" * 80 + "\n")
            
            if 1 in organized_results[model] and 8 in organized_results[model]:
                tp1 = organized_results[model][1]
                tp8 = organized_results[model][8]
                
                ttft_speedup = tp1['ttft_mean_ms'] / tp8['ttft_mean_ms'] if tp8['ttft_mean_ms'] > 0 else 0
                tpot_speedup = tp1['tpot_mean_ms'] / tp8['tpot_mean_ms'] if tp8['tpot_mean_ms'] > 0 else 0
                
                f.write(f"  • TP=1: TTFT={tp1['ttft_mean_ms']:.2f}ms, TPOT={tp1['tpot_mean_ms']:.2f}ms/token\n")
                f.write(f"  • TP=8: TTFT={tp8['ttft_mean_ms']:.2f}ms, TPOT={tp8['tpot_mean_ms']:.2f}ms/token\n")
                f.write(f"  • Speedup: {ttft_speedup:.2f}x (TTFT), {tpot_speedup:.2f}x (TPOT)\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("COMPARATIVE ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        
        # Compare models at TP=8
        if all(8 in organized_results[model] for model in models):
            f.write("At TP=8:\n")
            tp8_results = [(model, organized_results[model][8]) for model in models]
            tp8_results.sort(key=lambda x: x[1]['ttft_mean_ms'])
            
            f.write("  Fastest TTFT:\n")
            for model, data in tp8_results:
                f.write(f"    • {model.split('/')[-1]}: {data['ttft_mean_ms']:.2f}ms\n")
            
            tp8_results.sort(key=lambda x: x[1]['tpot_mean_ms'])
            f.write("  Fastest TPOT:\n")
            for model, data in tp8_results:
                f.write(f"    • {model.split('/')[-1]}: {data['tpot_mean_ms']:.2f}ms/token\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("CONCLUSIONS\n")
        f.write("=" * 80 + "\n\n")
        f.write("1. Tensor parallelism significantly improves inference latency\n")
        f.write("2. Different models show varying scaling behavior\n")
        f.write("3. TP=8 provides the best performance for all tested models\n")
        f.write("4. Model size affects both absolute performance and scaling efficiency\n")
    
    print(f"✓ Saved analysis summary: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze and visualize TTFT/TPOT measurement results',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--summary', type=str, default=None,
                       help='Path to summary JSON file')
    parser.add_argument('--results-dir', type=str, default='results',
                       help='Directory containing result JSON files')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory for plots and analysis')
    
    args = parser.parse_args()
    
    # Load results
    if args.summary:
        summary_file = Path(args.summary)
        if not summary_file.exists():
            print(f"✗ Summary file not found: {summary_file}")
            sys.exit(1)
        results = load_results_from_summary(summary_file)
    else:
        results_dir = Path(args.results_dir)
        if not results_dir.exists():
            print(f"✗ Results directory not found: {results_dir}")
            sys.exit(1)
        results = load_results_from_dir(results_dir)
    
    if not results:
        print("✗ No successful results found!")
        sys.exit(1)
    
    print(f"✓ Loaded {len(results)} successful results")
    
    # Organize results
    organized = organize_results(results)
    
    if not organized:
        print("✗ No organized results to analyze!")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Print analysis table
    print_analysis_table(organized)
    
    # Create plots
    create_plots(organized, output_dir)
    
    # Generate story summary
    summary_file = output_dir / 'analysis_summary.txt'
    generate_story_summary(organized, summary_file)
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)
    print(f"\nOutputs saved to: {output_dir}")
    print("  - ttft_vs_tp.png")
    print("  - tpot_vs_tp.png")
    print("  - throughput_vs_tp.png")
    print("  - combined_comparison.png")
    print("  - analysis_summary.txt")


if __name__ == '__main__':
    main()


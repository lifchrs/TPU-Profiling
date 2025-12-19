#!/usr/bin/env python3
"""
Plot TTFT and TPOT measurements for Qwen3-4B across different TP sizes

Usage:
    python3 setup/plot_qwen3_4b_results.py
"""

import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Set non-interactive backend
plt.switch_backend('Agg')

def load_results():
    """Load all Qwen3-4B result files"""
    results_dir = Path(__file__).parent.parent / 'results'
    
    results = {}
    for tp in [1, 2, 4, 8]:
        file_path = results_dir / f'qwen3_4b_tp{tp}.json'
        if file_path.exists():
            with open(file_path, 'r') as f:
                results[tp] = json.load(f)
        else:
            print(f"⚠ Warning: {file_path} not found")
    
    return results

def create_plots(results, output_dir):
    """Create TTFT and TPOT plots"""
    tp_sizes = sorted(results.keys())
    
    # Extract data
    ttft_values = [results[tp]['ttft_mean_ms'] for tp in tp_sizes]
    ttft_stds = [results[tp]['ttft_std_ms'] for tp in tp_sizes]
    tpot_values = [results[tp]['tpot_mean_ms'] for tp in tp_sizes]
    tpot_stds = [results[tp]['tpot_std_ms'] for tp in tp_sizes]
    throughput_values = [results[tp]['throughput_tokens_per_sec'] for tp in tp_sizes]
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: TTFT vs TP Size
    axes[0].errorbar(tp_sizes, ttft_values, yerr=ttft_stds, 
                    marker='o', linewidth=2.5, markersize=10, 
                    capsize=5, capthick=2, color='#1f77b4', 
                    label='Qwen3-4B', linestyle='-')
    axes[0].set_xlabel('Tensor Parallelism Size', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('TTFT (ms)', fontsize=12, fontweight='bold')
    axes[0].set_title('Time-to-First-Token (TTFT) vs TP Size', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3, linestyle='--')
    axes[0].set_xticks(tp_sizes)
    axes[0].legend(fontsize=11)
    
    # Add value labels on points
    for i, (tp, val) in enumerate(zip(tp_sizes, ttft_values)):
        axes[0].annotate(f'{val:.1f}ms', 
                         (tp, val), 
                         textcoords="offset points", 
                         xytext=(0,10), 
                         ha='center', fontsize=9)
    
    # Plot 2: TPOT vs TP Size
    axes[1].errorbar(tp_sizes, tpot_values, yerr=tpot_stds,
                    marker='s', linewidth=2.5, markersize=10,
                    capsize=5, capthick=2, color='#ff7f0e',
                    label='Qwen3-4B', linestyle='-')
    axes[1].set_xlabel('Tensor Parallelism Size', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('TPOT (ms/token)', fontsize=12, fontweight='bold')
    axes[1].set_title('Time-per-Output-Token (TPOT) vs TP Size', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3, linestyle='--')
    axes[1].set_xticks(tp_sizes)
    axes[1].legend(fontsize=11)
    
    # Add value labels on points
    for i, (tp, val) in enumerate(zip(tp_sizes, tpot_values)):
        axes[1].annotate(f'{val:.2f}ms', 
                         (tp, val), 
                         textcoords="offset points", 
                         xytext=(0,10), 
                         ha='center', fontsize=9)
    
    # Plot 3: Throughput vs TP Size
    axes[2].plot(tp_sizes, throughput_values, marker='^', linewidth=2.5, markersize=10,
                color='#2ca02c', label='Qwen3-4B', linestyle='-')
    axes[2].set_xlabel('Tensor Parallelism Size', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Throughput (tokens/sec)', fontsize=12, fontweight='bold')
    axes[2].set_title('Throughput vs TP Size', fontsize=13, fontweight='bold')
    axes[2].grid(True, alpha=0.3, linestyle='--')
    axes[2].set_xticks(tp_sizes)
    axes[2].legend(fontsize=11)
    
    # Add value labels on points
    for i, (tp, val) in enumerate(zip(tp_sizes, throughput_values)):
        axes[2].annotate(f'{val:.1f}', 
                         (tp, val), 
                         textcoords="offset points", 
                         xytext=(0,10), 
                         ha='center', fontsize=9)
    
    plt.tight_layout()
    
    # Save combined plot
    output_path = output_dir / 'qwen3_4b_combined_plots.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close(fig)
    
    # Create individual plots
    # TTFT only
    fig1, ax1 = plt.subplots(figsize=(10, 7))
    ax1.errorbar(tp_sizes, ttft_values, yerr=ttft_stds, 
                marker='o', linewidth=3, markersize=12, 
                capsize=6, capthick=2.5, color='#1f77b4', 
                label='Qwen3-4B', linestyle='-', markerfacecolor='white',
                markeredgewidth=2)
    ax1.set_xlabel('Tensor Parallelism Size', fontsize=14, fontweight='bold')
    ax1.set_ylabel('TTFT (ms)', fontsize=14, fontweight='bold')
    ax1.set_title('Qwen3-4B: Time-to-First-Token (TTFT) vs Tensor Parallelism', 
                 fontsize=15, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xticks(tp_sizes)
    ax1.legend(fontsize=12)
    
    # Add value labels
    for tp, val, std in zip(tp_sizes, ttft_values, ttft_stds):
        ax1.annotate(f'{val:.1f}ms\n±{std:.2f}', 
                    (tp, val), 
                    textcoords="offset points", 
                    xytext=(0,15), 
                    ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    output_path = output_dir / 'qwen3_4b_ttft.png'
    fig1.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close(fig1)
    
    # TPOT only
    fig2, ax2 = plt.subplots(figsize=(10, 7))
    ax2.errorbar(tp_sizes, tpot_values, yerr=tpot_stds,
                marker='s', linewidth=3, markersize=12,
                capsize=6, capthick=2.5, color='#ff7f0e',
                label='Qwen3-4B', linestyle='-', markerfacecolor='white',
                markeredgewidth=2)
    ax2.set_xlabel('Tensor Parallelism Size', fontsize=14, fontweight='bold')
    ax2.set_ylabel('TPOT (ms/token)', fontsize=14, fontweight='bold')
    ax2.set_title('Qwen3-4B: Time-per-Output-Token (TPOT) vs Tensor Parallelism', 
                 fontsize=15, fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xticks(tp_sizes)
    ax2.legend(fontsize=12)
    
    # Add value labels
    for tp, val, std in zip(tp_sizes, tpot_values, tpot_stds):
        ax2.annotate(f'{val:.2f}ms\n±{std:.3f}', 
                    (tp, val), 
                    textcoords="offset points", 
                    xytext=(0,15), 
                    ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    output_path = output_dir / 'qwen3_4b_tpot.png'
    fig2.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close(fig2)

def print_summary(results):
    """Print summary table"""
    print("\n" + "=" * 80)
    print("QWEN3-4B PERFORMANCE SUMMARY")
    print("=" * 80)
    print(f"{'TP':<6} {'TTFT (ms)':<15} {'TPOT (ms/tok)':<18} {'Throughput (tok/s)':<20}")
    print("-" * 80)
    
    for tp in sorted(results.keys()):
        data = results[tp]
        print(f"{tp:<6} "
              f"{data['ttft_mean_ms']:<15.2f} "
              f"{data['tpot_mean_ms']:<18.2f} "
              f"{data['throughput_tokens_per_sec']:<20.2f}")
    
    # Calculate speedups
    if 1 in results and 8 in results:
        tp1 = results[1]
        tp8 = results[8]
        
        ttft_speedup = tp1['ttft_mean_ms'] / tp8['ttft_mean_ms']
        tpot_speedup = tp1['tpot_mean_ms'] / tp8['tpot_mean_ms']
        throughput_speedup = tp8['throughput_tokens_per_sec'] / tp1['throughput_tokens_per_sec']
        
        print("\n" + "=" * 80)
        print("SPEEDUP ANALYSIS (TP=8 vs TP=1)")
        print("=" * 80)
        print(f"TTFT: {ttft_speedup:.2f}x faster (TP=1: {tp1['ttft_mean_ms']:.2f}ms → TP=8: {tp8['ttft_mean_ms']:.2f}ms)")
        print(f"TPOT: {tpot_speedup:.2f}x faster (TP=1: {tp1['tpot_mean_ms']:.2f}ms/tok → TP=8: {tp8['tpot_mean_ms']:.2f}ms/tok)")
        print(f"Throughput: {throughput_speedup:.2f}x higher (TP=1: {tp1['throughput_tokens_per_sec']:.2f} tok/s → TP=8: {tp8['throughput_tokens_per_sec']:.2f} tok/s)")

def main():
    # Load results
    results = load_results()
    
    if not results:
        print("✗ No results found! Make sure the JSON files are in results/ directory")
        sys.exit(1)
    
    print(f"✓ Loaded results for TP sizes: {sorted(results.keys())}")
    
    # Create output directory
    output_dir = Path(__file__).parent.parent / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Print summary
    print_summary(results)
    
    # Create plots
    print("\nGenerating plots...")
    create_plots(results, output_dir)
    
    print("\n" + "=" * 80)
    print("Plot generation complete!")
    print("=" * 80)
    print(f"\nPlots saved to: {output_dir}")
    print("  - qwen3_4b_combined_plots.png (all 3 metrics)")
    print("  - qwen3_4b_ttft.png (TTFT only)")
    print("  - qwen3_4b_tpot.png (TPOT only)")

if __name__ == '__main__':
    main()


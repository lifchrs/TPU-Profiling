#!/usr/bin/env python3
"""
LLM Inference Harness for TPU Profiling
Main entry point for running inference experiments with different parallelism strategies
"""

import argparse
import yaml
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
import torch
import torch_xla
import torch_xla.core.xla_model as xm
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from inference.models import load_model
from inference.parallelism import setup_parallelism
from profiling.profiler import TPUProfiler
from profiling.trace_collector import TraceCollector


class InferenceHarness:
    """Main inference harness for running LLM inference on TPUs"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = xm.xla_device()
        self.profiler = None
        self.trace_collector = TraceCollector(config.get('trace_output_dir', './traces'))
        
    def setup_model(self, model_name: str, parallelism_config: Dict):
        """Load and configure model with specified parallelism"""
        print(f"Loading model: {model_name}")
        model, tokenizer = load_model(model_name, self.config.get('models', {}).get(model_name, {}))
        
        # Setup parallelism strategy
        print(f"Setting up parallelism: {parallelism_config}")
        model = setup_parallelism(model, parallelism_config, self.device)
        
        model.eval()
        return model, tokenizer
    
    def run_inference(self, model, inputs: Dict, num_tokens: int = 100):
        """Run inference and collect traces"""
        print(f"Running inference for {num_tokens} tokens...")
        
        # Start profiling
        if self.config.get('enable_profiling', True):
            self.profiler = TPUProfiler()
            self.profiler.start()
        
        # Prefill phase
        print("Prefill phase...")
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Decode phase
        print("Decode phase...")
        generated_tokens = []
        current_inputs = inputs
        
        for i in range(num_tokens):
            with torch.no_grad():
                outputs = model(**current_inputs)
                next_token = outputs.logits[:, -1, :].argmax(dim=-1)
                generated_tokens.append(next_token.item())
                
                # Update inputs for next iteration
                current_inputs = {
                    'input_ids': torch.cat([current_inputs['input_ids'], next_token.unsqueeze(0)], dim=1)
                }
        
        # Stop profiling and collect trace
        if self.profiler:
            trace = self.profiler.stop()
            return generated_tokens, trace
        
        return generated_tokens, None
    
    def run_experiment(self, experiment_config: Dict):
        """Run a single experiment configuration"""
        model_name = experiment_config['model']
        parallelism = experiment_config['parallelism']
        batch_size = experiment_config.get('batch_size', 1)
        sequence_length = experiment_config.get('sequence_length', 2048)
        num_tokens = experiment_config.get('num_tokens', 100)
        
        print(f"\n{'='*60}")
        print(f"Experiment: {experiment_config.get('name', 'unnamed')}")
        print(f"Model: {model_name}")
        print(f"Parallelism: {parallelism}")
        print(f"Batch size: {batch_size}, Sequence length: {sequence_length}")
        print(f"{'='*60}\n")
        
        # Setup model
        model, tokenizer = self.setup_model(model_name, parallelism)
        
        # Prepare inputs
        # Note: This is a placeholder - actual implementation will depend on model
        inputs = {
            'input_ids': torch.randint(0, 1000, (batch_size, sequence_length)).to(self.device)
        }
        
        # Run inference
        tokens, trace = self.run_inference(model, inputs, num_tokens)
        
        # Save trace
        if trace:
            trace_path = self.trace_collector.save_trace(
                trace,
                model_name=model_name,
                parallelism=parallelism,
                batch_size=batch_size,
                sequence_length=sequence_length
            )
            print(f"Trace saved to: {trace_path}")
        
        return {
            'model': model_name,
            'parallelism': parallelism,
            'batch_size': batch_size,
            'sequence_length': sequence_length,
            'num_tokens': len(tokens),
            'trace_path': trace_path if trace else None
        }
    
    def run_all_experiments(self):
        """Run all experiments from configuration"""
        experiments = self.config.get('experiments', [])
        results = []
        
        for exp_config in experiments:
            try:
                result = self.run_experiment(exp_config)
                results.append(result)
            except Exception as e:
                print(f"Error running experiment {exp_config.get('name', 'unnamed')}: {e}")
                import traceback
                traceback.print_exc()
        
        # Save results summary
        results_path = Path(self.config.get('results_dir', './results')) / 'experiment_results.json'
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nAll experiments completed. Results saved to: {results_path}")
        return results


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='LLM Inference Harness for TPU Profiling')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to experiment configuration YAML file')
    parser.add_argument('--experiment', type=str, default=None,
                       help='Run specific experiment by name (default: run all)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize harness
    harness = InferenceHarness(config)
    
    # Run experiments
    if args.experiment:
        # Run specific experiment
        experiments = [e for e in config.get('experiments', []) 
                      if e.get('name') == args.experiment]
        if not experiments:
            print(f"Experiment '{args.experiment}' not found")
            return
        harness.config['experiments'] = experiments
    
    harness.run_all_experiments()


if __name__ == '__main__':
    main()


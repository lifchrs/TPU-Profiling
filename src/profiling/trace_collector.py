"""
Trace collection and organization utilities
"""

from pathlib import Path
from typing import Dict, Optional
import json
from datetime import datetime
import shutil


class TraceCollector:
    """Collects and organizes inference traces"""
    
    def __init__(self, output_dir: str = './traces'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_trace(self, trace_data: Dict, model_name: str, parallelism: Dict,
                   batch_size: int, sequence_length: int,
                   additional_metadata: Optional[Dict] = None) -> str:
        """
        Save trace with organized directory structure
        
        Directory structure:
        traces/
          {model_name}/
            {parallelism_strategy}/
              batch_{batch_size}_seq_{sequence_length}/
                trace_{timestamp}.json
                metadata.json
        """
        # Create directory structure
        parallelism_str = self._parallelism_to_string(parallelism)
        trace_dir = self.output_dir / model_name / parallelism_str / \
                   f"batch_{batch_size}_seq_{sequence_length}"
        trace_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Copy trace file if it exists
        trace_path = trace_data.get('trace_path')
        if trace_path and Path(trace_path).exists():
            dest_trace = trace_dir / f"trace_{timestamp}.trace"
            shutil.copy(trace_path, dest_trace)
        else:
            dest_trace = trace_dir / f"trace_{timestamp}.trace"
        
        # Save metadata
        metadata = {
            'model_name': model_name,
            'parallelism': parallelism,
            'batch_size': batch_size,
            'sequence_length': sequence_length,
            'timestamp': timestamp,
            'trace_path': str(dest_trace),
            **(additional_metadata or {})
        }
        
        metadata_path = trace_dir / f"metadata_{timestamp}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return str(dest_trace)
    
    def _parallelism_to_string(self, parallelism: Dict) -> str:
        """Convert parallelism config to string for directory naming"""
        strategy = parallelism.get('strategy', 'none')
        parts = [strategy]
        
        if 'tp_size' in parallelism:
            parts.append(f"tp{parallelism['tp_size']}")
        if 'pp_size' in parallelism:
            parts.append(f"pp{parallelism['pp_size']}")
        if 'dp_size' in parallelism:
            parts.append(f"dp{parallelism['dp_size']}")
        if 'seq_parallel_size' in parallelism:
            parts.append(f"seq{parallelism['seq_parallel_size']}")
        
        return "_".join(parts)
    
    def list_traces(self, model_name: Optional[str] = None,
                   parallelism: Optional[Dict] = None) -> list:
        """List all collected traces"""
        traces = []
        
        search_dir = self.output_dir
        if model_name:
            search_dir = search_dir / model_name
            if not search_dir.exists():
                return []
        
        for trace_file in search_dir.rglob("*.trace"):
            metadata_file = trace_file.parent / f"metadata_{trace_file.stem.split('_', 1)[1]}.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                traces.append({
                    'trace_path': str(trace_file),
                    'metadata': metadata
                })
        
        # Filter by parallelism if specified
        if parallelism:
            parallelism_str = self._parallelism_to_string(parallelism)
            traces = [t for t in traces 
                     if parallelism_str in t['trace_path']]
        
        return traces


"""
Model loading utilities for LLM inference on TPUs
"""

from typing import Dict, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch_xla.core.xla_model as xm


def load_model(model_name: str, config: Optional[Dict] = None) -> torch.nn.Module:
    """
    Load a model for inference on TPU
    
    Args:
        model_name: Name or path of the model
        config: Additional configuration for model loading
    
    Returns:
        Loaded model ready for inference
    """
    if config is None:
        config = {}
    
    # Model name mappings
    model_map = {
        'deepseek-v3.1-base': 'deepseek-ai/DeepSeek-V3',
        'qwen-32b': 'Qwen/Qwen2.5-32B',
        'deepseek-4.1': 'deepseek-ai/DeepSeek-V2.5',
        'deepseek-v2-lite': 'deepseek-ai/DeepSeek-V2-Lite'
    }
    
    # Get actual model name
    actual_name = model_map.get(model_name.lower(), model_name)
    
    print(f"Loading model: {actual_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(actual_name)
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        actual_name,
        torch_dtype=torch.bfloat16,  # TPUs work well with bfloat16
        device_map='cpu',  # We'll move to TPU after loading
        **config
    )
    
    return model, tokenizer


def prepare_inputs(text: str, tokenizer, max_length: int = 2048, device=None):
    """Prepare tokenized inputs for inference"""
    inputs = tokenizer(
        text,
        return_tensors='pt',
        max_length=max_length,
        truncation=True,
        padding=True
    )
    
    if device:
        inputs = {k: v.to(device) for k, v in inputs.items()}
    
    return inputs


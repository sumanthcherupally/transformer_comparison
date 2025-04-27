import torch
import numpy as np
from typing import Dict, List

def calculate_perplexity(model_output: torch.Tensor, target: torch.Tensor) -> float:
    """Calculate perplexity for language modeling tasks."""
    loss = torch.nn.functional.cross_entropy(
        model_output.view(-1, model_output.size(-1)),
        target.view(-1),
        reduction='mean'
    )
    return torch.exp(loss).item()

def calculate_attention_entropy(attention_weights: torch.Tensor) -> float:
    """Calculate entropy of attention distributions."""
    entropy = -torch.sum(attention_weights * torch.log(attention_weights + 1e-9), dim=-1)
    return entropy.mean().item()

def calculate_memory_usage(model: torch.nn.Module) -> Dict[str, float]:
    """Calculate model memory usage in MB."""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        
    size_mb = (param_size + buffer_size) / 1024 / 1024
    
    return {
        "parameters_mb": param_size / 1024 / 1024,
        "buffers_mb": buffer_size / 1024 / 1024,
        "total_mb": size_mb
    } 
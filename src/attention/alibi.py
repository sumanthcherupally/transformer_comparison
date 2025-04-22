import torch
import torch.nn as nn
import math

class AlibiPositionEmbedding(nn.Module):
    def __init__(self, num_heads, max_seq_len=2048):
        super().__init__()
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        
        # Create ALiBi slopes
        # The slope for head i is 2^(-8/i)
        slopes = torch.tensor([2 ** (-8 / i) for i in range(1, num_heads + 1)])
        self.register_buffer('slopes', slopes)
        
    def forward(self, x):
        """
        Apply ALiBi position biases to the input
        Args:
            x: Input tensor of shape (batch_size, seq_len, dim)
        Returns:
            Tensor with ALiBi position biases applied
        """
        # This is a placeholder - ALiBi is actually applied during attention computation
        return x
    
    def get_alibi_biases(self, seq_len):
        """
        Generate ALiBi position biases
        Args:
            seq_len: Sequence length
        Returns:
            Position biases of shape (num_heads, seq_len, seq_len)
        """
        # Create position indices
        positions = torch.arange(seq_len, device=self.slopes.device)
        
        # Create position differences matrix
        # Shape: (seq_len, seq_len)
        pos_diff = positions.unsqueeze(0) - positions.unsqueeze(1)
        
        # Apply slopes to position differences
        # Shape: (num_heads, seq_len, seq_len)
        alibi_biases = -self.slopes.view(-1, 1, 1) * pos_diff.unsqueeze(0)
        
        return alibi_biases 
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LocalAttention(nn.Module):
    def __init__(self, dim_k, window_size=128):
        super().__init__()
        self.dim_k = dim_k
        self.window_size = window_size
        self.scale = 1.0 / math.sqrt(dim_k)
        
    def forward(self, Q, K, V, mask=None):
        """
        Local attention implementation using sliding window approach
        Args:
            Q: Query matrix (batch_size, num_heads, seq_len, dim_k)
            K: Key matrix (batch_size, num_heads, seq_len, dim_k)
            V: Value matrix (batch_size, num_heads, seq_len, dim_v)
            mask: Optional mask (batch_size, 1, seq_len, seq_len)
        """
        B, H, L, D = Q.shape
        
        # Create local attention mask
        if mask is None:
            # Create a mask that only allows attention within the window
            local_mask = torch.ones(L, L, device=Q.device)
            for i in range(L):
                start = max(0, i - self.window_size // 2)
                end = min(L, i + self.window_size // 2)
                local_mask[i, :start] = 0
                local_mask[i, end:] = 0
            mask = local_mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # Apply mask
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights 
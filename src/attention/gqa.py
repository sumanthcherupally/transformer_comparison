import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GroupedQueryAttention(nn.Module):
    def __init__(self, dim_k, num_heads, num_groups=2):
        super().__init__()
        self.dim_k = dim_k
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.heads_per_group = num_heads // num_groups
        assert num_heads % num_groups == 0, "Number of heads must be divisible by number of groups"
        
        self.scale = 1.0 / math.sqrt(dim_k)
        
    def forward(self, Q, K, V, mask=None):
        """
        Grouped-Query Attention implementation
        Args:
            Q: Query matrix (batch_size, num_heads, seq_len, dim_k)
            K: Key matrix (batch_size, num_heads, seq_len, dim_k)
            V: Value matrix (batch_size, num_heads, seq_len, dim_v)
            mask: Optional mask (batch_size, 1, seq_len, seq_len)
        """
        B, H, L, D = Q.shape
        
        # Reshape for grouped attention
        # Group queries, but keep keys and values as they are
        Q = Q.view(B, self.num_groups, self.heads_per_group, L, D)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, V)
        
        # Reshape back to original dimensions
        output = output.view(B, H, L, D)
        
        return output, attention_weights 
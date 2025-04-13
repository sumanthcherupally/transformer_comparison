import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DotProductAttention(nn.Module):
    def __init__(self, dim_k):
        super().__init__()
        self.scale = 1.0 / math.sqrt(dim_k)
    
    def forward(self, Q, K, V, mask=None):
        """
        Standard scaled dot-product attention
        Args:
            Q: Query matrix (batch_size, num_heads, seq_len, dim_k)
            K: Key matrix (batch_size, num_heads, seq_len, dim_k)
            V: Value matrix (batch_size, num_heads, seq_len, dim_v)
            mask: Optional mask (batch_size, 1, seq_len, seq_len)
        """
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights 
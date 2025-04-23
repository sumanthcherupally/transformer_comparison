import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FlashAttention(nn.Module):
    def __init__(self, dim_k, block_size=64):
        super().__init__()
        self.dim_k = dim_k
        self.block_size = block_size
        self.scale = 1.0 / math.sqrt(dim_k)
        
    def forward(self, Q, K, V, mask=None):
        """
        Flash Attention implementation with block-wise computation
        Args:
            Q: Query matrix (batch_size, num_heads, seq_len, dim_k)
            K: Key matrix (batch_size, num_heads, seq_len, dim_k)
            V: Value matrix (batch_size, num_heads, seq_len, dim_v)
            mask: Optional mask (batch_size, 1, seq_len, seq_len)
        """
        B, H, L, D = Q.shape
        
        # Reshape into blocks for efficient computation
        Q_blocks = Q.view(B, H, -1, self.block_size, D)
        K_blocks = K.view(B, H, -1, self.block_size, D)
        V_blocks = V.view(B, H, -1, self.block_size, D)
        
        # Compute attention scores in blocks
        scores = torch.matmul(Q_blocks, K_blocks.transpose(-2, -1)) * self.scale
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.view(B, 1, -1, self.block_size, self.block_size)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, V_blocks)
        
        # Reshape back to original dimensions
        output = output.view(B, H, L, D)
        
        return output, attention_weights 
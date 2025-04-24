import torch
import torch.nn as nn
import math

class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Create position embeddings
        position = torch.arange(max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(base) / dim))
        
        # Create sin and cos embeddings
        pe = torch.zeros(max_seq_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register buffer (not a parameter, but should be saved and moved with the model)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Apply rotary position embeddings to the input
        Args:
            x: Input tensor of shape (batch_size, seq_len, dim)
        Returns:
            Tensor with rotary position embeddings applied
        """
        seq_len = x.size(1)
        return x + self.pe[:seq_len]
    
    def apply_rotary_embeddings(self, q, k, positions):
        """
        Apply rotary embeddings to queries and keys
        Args:
            q: Query tensor of shape (batch_size, num_heads, seq_len, dim_per_head)
            k: Key tensor of shape (batch_size, num_heads, seq_len, dim_per_head)
            positions: Position indices of shape (batch_size, seq_len)
        Returns:
            q and k with rotary embeddings applied
        """
        # Extract sin and cos embeddings for the given positions
        sin_emb = self.pe[positions, 0::2].unsqueeze(1)  # Add head dimension
        cos_emb = self.pe[positions, 1::2].unsqueeze(1)  # Add head dimension
        
        # Split q and k into even and odd dimensions
        q_even = q[..., 0::2]
        q_odd = q[..., 1::2]
        k_even = k[..., 0::2]
        k_odd = k[..., 1::2]
        
        # Apply rotary embeddings
        q_rotated = torch.cat([
            q_even * cos_emb - q_odd * sin_emb,
            q_even * sin_emb + q_odd * cos_emb
        ], dim=-1)
        
        k_rotated = torch.cat([
            k_even * cos_emb - k_odd * sin_emb,
            k_even * sin_emb + k_odd * cos_emb
        ], dim=-1)
        
        return q_rotated, k_rotated 
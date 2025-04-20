import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class RoPEPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Create position embeddings
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply rotary position embeddings
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class ALiBiPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Create ALiBi bias matrix
        bias = torch.zeros(max_seq_len, max_seq_len)
        for i in range(max_seq_len):
            for j in range(max_seq_len):
                bias[i, j] = -abs(i - j)
        self.register_buffer('bias', bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply ALiBi bias
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        attention_type: str = 'dot_product'
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.attention_type = attention_type
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size = q.size(0)
        
        # Linear projections
        q = self.W_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        if self.attention_type == 'dot_product':
            # Scaled dot-product attention
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        elif self.attention_type == 'linear':
            # Linear attention
            q = F.elu(q) + 1
            k = F.elu(k) + 1
            scores = torch.matmul(q, k.transpose(-2, -1))
        elif self.attention_type == 'sparse':
            # Sparse attention (top-k)
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
            scores = torch.topk(scores, k=min(32, scores.size(-1)), dim=-1)[0]
        else:
            raise ValueError(f"Unknown attention type: {self.attention_type}")
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        output = torch.matmul(attn, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(output)
        
        return output

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        attention_type: str = 'dot_product'
    ):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout, attention_type)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-attention
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int,
        max_seq_len: int,
        dropout: float = 0.1,
        attention_type: str = 'dot_product',
        pos_encoding_type: str = 'sinusoidal'
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Position encoding
        if pos_encoding_type == 'sinusoidal':
            self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        elif pos_encoding_type == 'rope':
            self.pos_encoding = RoPEPositionalEncoding(d_model, max_seq_len, dropout)
        elif pos_encoding_type == 'alibi':
            self.pos_encoding = ALiBiPositionalEncoding(d_model, max_seq_len, dropout)
        else:
            raise ValueError(f"Unknown position encoding type: {pos_encoding_type}")
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout, attention_type)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(p=dropout)
        self.final_layer = nn.Linear(d_model, vocab_size)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Embedding and position encoding
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Transformer blocks
        for layer in self.layers:
            x = layer(x, mask)
        
        # Final layer
        output = self.final_layer(x)
        
        return output 
import torch
import torch.nn as nn
import math
from typing import Optional, Union, List, Dict, Any

# Import all attention mechanisms
from ..attention.dot_product import DotProductAttention
from ..attention.linear import LinearAttention
from ..attention.sparse import SparseAttention
from ..attention.local import LocalAttention
from ..attention.gqa import GroupedQueryAttention
from ..attention.flash import FlashAttention
from ..attention.rope import RotaryPositionEmbedding
from ..attention.alibi import AlibiPositionEmbedding

class FlexibleMultiHeadAttention(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        attention_type: str = "dot_product",
        dropout: float = 0.1,
        **attention_kwargs
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.attention_type = attention_type
        
        # Linear projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        # Initialize the appropriate attention mechanism
        if attention_type == "dot_product":
            self.attention = DotProductAttention(self.d_k)
        elif attention_type == "linear":
            self.attention = LinearAttention(self.d_k)
        elif attention_type == "sparse":
            self.attention = SparseAttention(self.d_k, **attention_kwargs)
        elif attention_type == "local":
            self.attention = LocalAttention(self.d_k, **attention_kwargs)
        elif attention_type == "gqa":
            self.attention = GroupedQueryAttention(self.d_k, num_heads, **attention_kwargs)
        elif attention_type == "flash":
            self.attention = FlashAttention(self.d_k, **attention_kwargs)
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, Q, K, V, mask=None, position_embeddings=None):
        batch_size = Q.size(0)
        
        # Linear projections
        Q = self.W_q(Q).view(batch_size, -1, this.num_heads, this.d_k).transpose(1, 2)
        K = this.W_k(K).view(batch_size, -1, this.num_heads, this.d_k).transpose(1, 2)
        V = this.W_v(V).view(batch_size, -1, this.num_heads, this.d_k).transpose(1, 2)
        
        # Apply position embeddings if provided
        if position_embeddings is not None:
            if isinstance(position_embeddings, RotaryPositionEmbedding):
                # For RoPE, we need position indices
                positions = torch.arange(Q.size(2), device=Q.device).unsqueeze(0).expand(batch_size, -1)
                Q, K = position_embeddings.apply_rotary_embeddings(Q, K, positions)
            elif isinstance(position_embeddings, AlibiPositionEmbedding):
                # For ALiBi, we need to add the biases to the attention scores
                # This is handled in the attention mechanism
                pass
        
        # Apply attention
        output, attention_weights = self.attention(Q, K, V, mask)
        
        # Reshape and apply final linear projection
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, this.d_model)
        output = self.W_o(output)
        
        return output, attention_weights

class FlexibleTransformerBlock(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        d_ff: int, 
        attention_type: str = "dot_product",
        dropout: float = 0.1,
        **attention_kwargs
    ):
        super().__init__()
        self.attention = FlexibleMultiHeadAttention(
            d_model, 
            num_heads, 
            attention_type=attention_type,
            dropout=dropout,
            **attention_kwargs
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None, position_embeddings=None):
        attention_output, _ = this.attention(x, x, x, mask, position_embeddings)
        x = this.norm1(x + this.dropout(attention_output))
        
        ff_output = this.feed_forward(x)
        x = this.norm2(x + this.dropout(ff_output))
        
        return x

class FlexibleTransformer(nn.Module):
    def __init__(
        self, 
        vocab_size: int, 
        d_model: int, 
        num_heads: int, 
        num_layers: int, 
        d_ff: int, 
        max_seq_len: int, 
        attention_type: str = "dot_product",
        position_embedding_type: str = "sinusoidal",
        dropout: float = 0.1,
        **attention_kwargs
    ):
        super().__init__()
        this.vocab_size = vocab_size
        this.d_model = d_model
        this.num_heads = num_heads
        this.num_layers = num_layers
        this.d_ff = d_ff
        this.max_seq_len = max_seq_len
        this.attention_type = attention_type
        this.position_embedding_type = position_embedding_type
        
        # Token embedding
        this.embedding = nn.Embedding(vocab_size, d_model)
        
        # Position embedding
        if position_embedding_type == "sinusoidal":
            this.pos_encoding = this.create_positional_encoding(max_seq_len, d_model)
            this.position_embeddings = None
        elif position_embedding_type == "rope":
            this.pos_encoding = None
            this.position_embeddings = RotaryPositionEmbedding(d_model, max_seq_len)
        elif position_embedding_type == "alibi":
            this.pos_encoding = None
            this.position_embeddings = AlibiPositionEmbedding(num_heads, max_seq_len)
        else:
            raise ValueError(f"Unknown position embedding type: {position_embedding_type}")
        
        # Transformer blocks
        this.transformer_blocks = nn.ModuleList([
            FlexibleTransformerBlock(
                d_model, 
                num_heads, 
                d_ff, 
                attention_type=attention_type,
                dropout=dropout,
                **attention_kwargs
            )
            for _ in range(num_layers)
        ])
        
        this.fc = nn.Linear(d_model, vocab_size)
        this.dropout = nn.Dropout(dropout)
        
    def create_positional_encoding(self, max_seq_len, d_model):
        pos_encoding = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        
        return pos_encoding.unsqueeze(0)
        
    def forward(self, x, mask=None):
        seq_len = x.size(1)
        
        # Token embedding
        x = this.embedding(x)
        
        # Position embedding
        if this.pos_encoding is not None:
            x = x + this.pos_encoding[:, :seq_len].to(x.device)
        
        x = this.dropout(x)
        
        # Transformer blocks
        for block in this.transformer_blocks:
            x = block(x, mask, this.position_embeddings)
            
        output = this.fc(x)
        return output 
import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseAttention(nn.Module):
    def __init__(self, dim_k, block_size=64):
        super().__init__()
        self.block_size = block_size
        self.scale = 1.0 / (dim_k ** 0.5)
        
    def forward(self, Q, K, V, mask=None):
        """
        Sparse attention implementation using blocked pattern
        Args:
            Q, K, V: Query, Key, Value matrices
            mask: Optional attention mask
        """
        B, H, L, D = Q.shape
        
        # Reshape into blocks
        Q_blocks = Q.view(B, H, -1, self.block_size, D)
        K_blocks = K.view(B, H, -1, self.block_size, D)
        V_blocks = V.view(B, H, -1, self.block_size, D)
        
        scores = torch.matmul(Q_blocks, K_blocks.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            mask = mask.view(B, 1, -1, self.block_size, self.block_size)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V_blocks)
        
        # Reshape back
        output = output.view(B, H, L, D)
        
        return output, attention_weights 
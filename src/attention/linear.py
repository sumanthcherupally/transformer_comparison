import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearAttention(nn.Module):
    def __init__(self, dim_k):
        super().__init__()
        self.dim_k = dim_k
        
    def forward(self, Q, K, V, mask=None):
        """
        Linear attention implementation with O(n) complexity
        Args:
            Q, K, V: Query, Key, Value matrices
            mask: Optional attention mask
        """
        Q = F.elu(Q) + 1
        K = F.elu(K) + 1
        
        if mask is not None:
            K = K * mask.unsqueeze(-1)
        
        KV = torch.matmul(K.transpose(-2, -1), V)
        Z = 1 / (torch.matmul(Q, K.sum(dim=-2).unsqueeze(-1)))
        output = torch.matmul(Q, KV) * Z
        
        return output, None 
import math
import torch
from torch import nn
import torch.nn.functional as F


def self_attn(Q, K, V):
    """
    Compute Scaled Dot-Product Attention output given Q, K, V
    
    Args:
         Q: Query matrix
         K: Key matrix
         V: Value matrix
    """
    d_k = K.size()[-1] # dimensionality of q or v
    out = torch.matmul(Q, K.transpose(-2, -1)) # shape: (b, h, l, l)
    out = out * (1 / math.sqrt(d_k))
    out = F.softmax(out, dim=-1)
    out = torch.matmul(out, V)

    return out # output shape: (b, h, l, d)


class PreLinear(torch.nn):

    def __init__(self, d, h):
        super().__init__()
        self.d = d
        self.h = h
        self.linear = nn.Linear(d, d)

    def forward(self, x):
        b, l = x.shape[:-1] # (batch_size, seq_len)
        x = self.linear(x) # apply linear transformation
        x = x.view(b, l, self.h, self.d) # split into separate heads

        return x.tranpose(1, 2) # output shape: (b, h, l, d)


class MultiHeadedAttention(torch.nn):

    def __init__(self, n_heads, d_model):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.Q_linear = PreLinear(self.d_model, self.n_heads)
        self.K_linear = PreLinear(self.d_model, self.n_heads)
        self.V_linear = PreLinear(self.d_model, self.n_heads)
        self.O_linear = nn.Linear(self.n_heads * self.d_model, self.d_model) # recheck the shape

    def forward(self, Q, K, V):
        b, l = Q.shape[0], Q.shape[1] # batch first
        Q = self.Q_linear(Q)
        K = self.K_linear(K)
        V = self.V_linear(V)

        # Scaled Dot-Product Attention for each head
        out = self_attn(Q, K, V)

        # Concatenate heads into only one vector (one head)
        out = out.transpose(1, 2).view(b, l, self.n_heads * self.d_model)

        # Final linear layer
        out = self.O_linear(out)

        return out

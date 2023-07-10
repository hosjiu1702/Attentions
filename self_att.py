import math
import torch
import torch.nn.functional as F


def self_attn(Q, K, V):
    """
    Compute Scaled Dot-Product Attention output given Q, K, V
    
    Args:
         Q: Query matrix
         K: Key matrix
         V: Value matrix
    """
    d_k = K.size()[1] # dimensionality of q or v
    out = torch.mm(Q, K.t()) # can we use this .t() function with batched input?
    out = out * (1 / math.sqrt(d_k))
    out = F.softmax(out, dim=1)
    out = torch.mm(out, V)

    return out

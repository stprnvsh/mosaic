"""
Local attention backend using PyTorch's scaled_dot_product_attention.
No communication - runs entirely on local GPU.
"""

import torch
import torch.nn.functional as F


def local_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    **kwargs
) -> torch.Tensor:
    """
    Standard scaled dot-product attention.
    
    Args:
        q: Query tensor (batch, heads, seq, head_dim)
        k: Key tensor (batch, heads, seq, head_dim)
        v: Value tensor (batch, heads, seq, head_dim)
        
    Returns:
        Attention output (batch, heads, seq, head_dim)
    """
    return F.scaled_dot_product_attention(q, k, v)


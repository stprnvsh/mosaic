"""
Local attention backend using PyTorch's scaled_dot_product_attention.
No communication - runs entirely on local GPU.
"""

import torch
import torch.nn.functional as F
from typing import Optional


def local_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    **kwargs
) -> torch.Tensor:
    """
    Standard scaled dot-product attention.
    
    Args:
        q: Query tensor (batch, heads, seq, head_dim)
        k: Key tensor (batch, heads, seq, head_dim)
        v: Value tensor (batch, heads, seq, head_dim)
        attn_mask: Optional attention mask
        is_causal: If True, use causal masking (ignored if attn_mask provided)
        
    Returns:
        Attention output (batch, heads, seq, head_dim)
    """
    return F.scaled_dot_product_attention(
        q, k, v, 
        attn_mask=attn_mask,
        is_causal=is_causal and attn_mask is None
    )

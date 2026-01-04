"""
Ring attention backend using ring-flash-attn.
Distributes attention computation across GPUs using ring communication.
"""

import torch
from typing import Optional
import torch.distributed as dist

# Check if ring-flash-attn is available
try:
    from ring_flash_attn import ring_flash_attn_func
    RING_AVAILABLE = True
except ImportError:
    RING_AVAILABLE = False
    ring_flash_attn_func = None


def ring_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    group: Optional[dist.ProcessGroup] = None,
    **kwargs
) -> torch.Tensor:
    """
    Ring attention via ring-flash-attn.
    
    Each GPU holds a shard of the sequence. K/V are passed around in a ring
    so each GPU's Q can attend to all K/V.
    
    Args:
        q: Query tensor (batch, heads, seq_local, head_dim)
        k: Key tensor (batch, heads, seq_local, head_dim)
        v: Value tensor (batch, heads, seq_local, head_dim)
        group: Process group for ring communication
        
    Returns:
        Attention output (batch, heads, seq_local, head_dim)
        
    Raises:
        ImportError: If ring-flash-attn is not installed
    """
    if not RING_AVAILABLE:
        raise ImportError(
            "ring-flash-attn not installed. Install with: pip install ring-flash-attn flash-attn"
        )
    
    if group is None:
        group = dist.group.WORLD
    
    # ring_flash_attn_func expects (batch, seq, heads, head_dim)
    # but we have (batch, heads, seq, head_dim), so transpose
    q = q.transpose(1, 2).contiguous()
    k = k.transpose(1, 2).contiguous()
    v = v.transpose(1, 2).contiguous()
    
    out = ring_flash_attn_func(q, k, v, group=group)
    
    # Transpose back to (batch, heads, seq, head_dim)
    return out.transpose(1, 2).contiguous()


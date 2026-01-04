"""
Ring attention backend using ring-flash-attn.
Distributes attention computation across GPUs using ring communication.

For causal attention, uses zigzag/striped partitioning to balance load.
"""

import torch
from typing import Optional
import torch.distributed as dist

# Check if ring-flash-attn is available
try:
    from ring_flash_attn import ring_flash_attn_func
    # Check for causal variant
    try:
        from ring_flash_attn import zigzag_ring_flash_attn_func
        ZIGZAG_AVAILABLE = True
    except ImportError:
        ZIGZAG_AVAILABLE = False
        zigzag_ring_flash_attn_func = None
    RING_AVAILABLE = True
except ImportError:
    RING_AVAILABLE = False
    ZIGZAG_AVAILABLE = False
    ring_flash_attn_func = None
    zigzag_ring_flash_attn_func = None


def ring_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    group: Optional[dist.ProcessGroup] = None,
    causal: bool = False,
    **kwargs
) -> torch.Tensor:
    """
    Ring attention via ring-flash-attn.
    
    Each GPU holds a shard of the sequence. K/V are passed around in a ring
    so each GPU's Q can attend to all K/V.
    
    For causal attention, uses zigzag partitioning to balance load across GPUs.
    
    Args:
        q: Query tensor (batch, heads, seq_local, head_dim)
        k: Key tensor (batch, heads, seq_local, head_dim)
        v: Value tensor (batch, heads, seq_local, head_dim)
        group: Process group for ring communication
        causal: If True, use causal masking with striped partitioning
        
    Returns:
        Attention output (batch, heads, seq_local, head_dim)
    """
    if not RING_AVAILABLE:
        raise ImportError(
            "ring-flash-attn not installed. Install with: pip install ring-flash-attn flash-attn"
        )
    
    if group is None:
        group = dist.group.WORLD
    
    # ring_flash_attn_func expects (batch, seq, heads, head_dim)
    q = q.transpose(1, 2).contiguous()
    k = k.transpose(1, 2).contiguous()
    v = v.transpose(1, 2).contiguous()
    
    if causal:
        if ZIGZAG_AVAILABLE:
            # Use zigzag for balanced causal attention
            out = zigzag_ring_flash_attn_func(q, k, v, group=group, causal=True)
        else:
            # Fallback: standard ring with causal (may have load imbalance)
            out = ring_flash_attn_func(q, k, v, group=group, causal=True)
    else:
        out = ring_flash_attn_func(q, k, v, group=group, causal=False)
    
    # Transpose back to (batch, heads, seq, head_dim)
    return out.transpose(1, 2).contiguous()


def stripe_partition(x: torch.Tensor, world_size: int, rank: int) -> torch.Tensor:
    """
    Striped partitioning for balanced causal attention.
    
    Instead of contiguous chunks [0:N/p, N/p:2N/p, ...], use stripes:
    GPU 0: positions [0, p, 2p, ...]
    GPU 1: positions [1, p+1, 2p+1, ...]
    
    This balances computation for causal attention.
    """
    seq_len = x.shape[-2]
    indices = torch.arange(rank, seq_len, world_size, device=x.device)
    return x.index_select(-2, indices)


def unstripe_partition(x: torch.Tensor, world_size: int, rank: int, total_seq: int) -> torch.Tensor:
    """
    Reverse striped partitioning - gather stripes back to contiguous.
    """
    # This requires all-gather across the group
    # For now, return as-is (caller handles gathering)
    return x

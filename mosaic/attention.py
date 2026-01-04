"""
MultiAxisAttention: Attention over arbitrary tensor axis with backend selection.

The key insight: different axes may need different sharding strategies.
- Small axes (features, ~5-10) -> local attention
- Large axes (rows, ~150k) -> ring attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, Optional, Tuple, List

from mosaic.context import MosaicContext
from mosaic.backends.local import local_attention
from mosaic.backends.ring import ring_attention, RING_AVAILABLE


class MultiAxisAttention(nn.Module):
    """
    Attention over arbitrary tensor axis with configurable sharding backend.
    
    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        attention_axis: Which tensor axis to compute attention over
        backend: Sharding backend ("local" or "ring")
        
    Example:
        # Attention over axis 1 (rows) with ring sharding
        attn = MultiAxisAttention(embed_dim=96, num_heads=4, attention_axis=1, backend="ring")
        
        # Input: (batch, rows_local, features, embed)
        # Each GPU has rows_local = total_rows / num_gpus
        out = attn(x)  # Ring attention across GPUs
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attention_axis: int,
        backend: Literal["local", "ring"] = "local",
        bias: bool = True,
    ):
        super().__init__()
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.attention_axis = attention_axis
        self.backend = backend
        
        # Projections
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # Select attention function
        if backend == "ring":
            if not RING_AVAILABLE:
                raise ImportError(
                    "ring-flash-attn not installed. Install with: pip install ring-flash-attn flash-attn"
                )
            self._attn_fn = self._ring_attention
        else:
            self._attn_fn = self._local_attention
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute attention over attention_axis.
        
        Args:
            x: N-dimensional tensor with embed_dim as last dimension
            
        Returns:
            Tensor of same shape as input
        """
        original_shape = x.shape
        ndim = x.ndim
        
        # Step 1: Move attention axis to position -2 (seq position)
        x, perm, inv_perm = self._permute_to_seq(x)
        
        # Step 2: Flatten batch dimensions -> (batch, seq, embed)
        batch_shape = x.shape[:-2]
        batch_size = x.shape[:-2].numel() if len(batch_shape) > 0 else 1
        seq_len = x.shape[-2]
        x = x.reshape(batch_size, seq_len, self.embed_dim)
        
        # Step 3: Project Q, K, V
        qkv = self.qkv_proj(x)  # (batch, seq, 3 * embed)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, heads, seq, head_dim)
        q, k, v = qkv.unbind(0)
        
        # Step 4: Attention via selected backend
        out = self._attn_fn(q, k, v)
        
        # Step 5: Reshape and project output
        out = out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, self.embed_dim)
        out = self.out_proj(out)
        
        # Step 6: Restore original shape
        out = out.reshape(*batch_shape, seq_len, self.embed_dim)
        out = out.permute(inv_perm)
        
        return out
    
    def _permute_to_seq(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[int], List[int]]:
        """
        Move attention_axis to position -2 (standard seq position).
        
        Returns:
            (permuted_tensor, permutation, inverse_permutation)
        """
        ndim = x.ndim
        ax = self.attention_axis if self.attention_axis >= 0 else ndim + self.attention_axis
        
        if ax == ndim - 2:
            # Already in correct position
            identity = list(range(ndim))
            return x, identity, identity
        
        # Build permutation: all dims except ax, then ax, then last dim
        perm = [i for i in range(ndim) if i != ax and i != ndim - 1]
        perm.append(ax)
        perm.append(ndim - 1)
        
        # Compute inverse permutation
        inv_perm = [0] * ndim
        for i, p in enumerate(perm):
            inv_perm[p] = i
        
        return x.permute(perm), perm, inv_perm
    
    def _local_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Standard local attention (no communication)."""
        return local_attention(q, k, v)
    
    def _ring_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Ring attention across GPUs."""
        ctx = MosaicContext.get()
        return ring_attention(q, k, v, group=ctx.sp_group)


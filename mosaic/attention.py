"""
MultiAxisAttention: Attention over arbitrary tensor axis with backend selection.

The key insight: different axes may need different sharding strategies.
- Small axes (features, ~5-10) -> local attention
- Large axes (rows, ~150k) -> ring attention
- Very large axes or 2D patterns -> mesh2d attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, Optional, Tuple, List, Union

from mosaic.context import MosaicContext
from mosaic.backends.local import local_attention
from mosaic.backends.ring import ring_attention, RING_AVAILABLE
from mosaic.backends.mesh2d import mesh2d_attention, Mesh2DContext, MESH2D_AVAILABLE


BackendType = Literal["local", "ring", "mesh2d"]


class MultiAxisAttention(nn.Module):
    """
    Attention over arbitrary tensor axis with configurable sharding backend.
    
    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        attention_axis: Which tensor axis to compute attention over
        backend: Sharding backend
            - "local": No communication, single GPU
            - "ring": 1D sequence sharding with ring communication
            - "mesh2d": 2D mesh sharding for very large attention matrices
        mesh_shape: Required for mesh2d backend, (rows, cols) of GPU mesh
        
    Example:
        # Ring attention over axis 1 (rows)
        attn = MultiAxisAttention(embed_dim=96, num_heads=4, attention_axis=1, backend="ring")
        
        # 2D mesh attention for very large matrices
        attn = MultiAxisAttention(
            embed_dim=96, num_heads=4, attention_axis=1, 
            backend="mesh2d", mesh_shape=(2, 2)
        )
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attention_axis: int,
        backend: BackendType = "local",
        mesh_shape: Optional[Tuple[int, int]] = None,
        bias: bool = True,
    ):
        super().__init__()
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.attention_axis = attention_axis
        self.backend = backend
        self.mesh_shape = mesh_shape
        
        # Projections
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # Initialize mesh2d context if needed
        self._mesh2d_ctx = None
        if backend == "mesh2d":
            if mesh_shape is None:
                raise ValueError("mesh_shape required for mesh2d backend")
            self._mesh2d_ctx = Mesh2DContext(mesh_shape)
        
        # Select attention function
        if backend == "ring":
            if not RING_AVAILABLE:
                raise ImportError(
                    "ring-flash-attn not installed. Install with: pip install ring-flash-attn flash-attn"
                )
            self._attn_fn = self._ring_attention
        elif backend == "mesh2d":
            self._attn_fn = self._mesh2d_attention
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
        # Step 1: Move attention axis to seq position (-2)
        x, inv_perm = self._permute_to_seq(x)
        
        # Step 2: Flatten batch dims -> (batch, seq, embed)
        batch_shape = x.shape[:-2]
        seq_len = x.shape[-2]
        x = x.view(-1, seq_len, self.embed_dim)
        batch_size = x.shape[0]
        
        # Step 3: Project Q, K, V (fused projection)
        qkv = self.qkv_proj(x).view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # (batch, heads, seq, head_dim)
        
        # Step 4: Attention via backend
        out = self._attn_fn(q, k, v)
        
        # Step 5: Reshape and project
        out = out.transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
        out = self.out_proj(out)
        
        # Step 6: Restore shape
        if batch_shape:
            out = out.view(*batch_shape, seq_len, self.embed_dim)
        if inv_perm:
            out = out.permute(inv_perm)
        
        return out
    
    def _permute_to_seq(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[List[int]]]:
        """
        Move attention_axis to position -2 (standard seq position).
        
        Returns:
            (permuted_tensor, inverse_permutation or None if no permute needed)
        """
        ndim = x.ndim
        ax = self.attention_axis if self.attention_axis >= 0 else ndim + self.attention_axis
        
        if ax == ndim - 2:
            return x, None  # Already in correct position
        
        # Build permutation: all dims except ax, then ax, then last dim
        perm = [i for i in range(ndim) if i != ax and i != ndim - 1]
        perm.append(ax)
        perm.append(ndim - 1)
        
        # Compute inverse permutation
        inv_perm = [0] * ndim
        for i, p in enumerate(perm):
            inv_perm[p] = i
        
        return x.permute(perm), inv_perm
    
    def _local_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Standard local attention (no communication)."""
        return local_attention(q, k, v)
    
    def _ring_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Ring attention across GPUs (1D sequence sharding)."""
        ctx = MosaicContext.get()
        return ring_attention(q, k, v, group=ctx.sp_group)
    
    def _mesh2d_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """2D mesh attention (2D sharding of attention matrix)."""
        return mesh2d_attention(
            q, k, v,
            row_group=self._mesh2d_ctx.row_group,
            col_group=self._mesh2d_ctx.col_group
        )


class MultiAxisAttention2D(nn.Module):
    """
    Attention over TWO axes simultaneously with 2D mesh sharding.
    
    For patterns where you need to attend across both rows and columns
    of a 2D structure (e.g., image patches, tables).
    
    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        axis1: First axis to attend over
        axis2: Second axis to attend over
        mesh_shape: (rows, cols) GPU mesh shape
        
    Example:
        # Attend over both rows and columns of a table
        attn = MultiAxisAttention2D(
            embed_dim=96, num_heads=4,
            axis1=1, axis2=2,  # rows and columns
            mesh_shape=(2, 2)
        )
        # Input: (batch, rows, cols, embed)
        # Flattens to (batch, rows*cols, embed) internally
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        axis1: int,
        axis2: int,
        mesh_shape: Tuple[int, int],
        bias: bool = True,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.axis1 = axis1
        self.axis2 = axis2
        self.mesh_shape = mesh_shape
        
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self._mesh2d_ctx = Mesh2DContext(mesh_shape)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute 2D attention.
        
        Args:
            x: Tensor with shape (..., dim1, dim2, embed)
        """
        # Flatten two attention axes into one sequence dim
        x, restore_info = self._flatten_axes(x)
        batch_size, seq_len = x.shape[0], x.shape[1]
        
        # Project Q, K, V (fused)
        qkv = self.qkv_proj(x).view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        
        # 2D mesh attention
        out = mesh2d_attention(q, k, v, self._mesh2d_ctx.row_group, self._mesh2d_ctx.col_group)
        
        # Reshape and project
        out = out.transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
        out = self.out_proj(out)
        
        # Restore original shape
        return self._unflatten_axes(out, restore_info)
    
    def _flatten_axes(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple]:
        """Flatten axis1 and axis2 into a single sequence dimension."""
        ndim = x.ndim
        ax1 = self.axis1 if self.axis1 >= 0 else ndim + self.axis1
        ax2 = self.axis2 if self.axis2 >= 0 else ndim + self.axis2
        if ax1 > ax2:
            ax1, ax2 = ax2, ax1
        
        dim1, dim2 = x.shape[ax1], x.shape[ax2]
        
        # Permute ax1, ax2 to positions -3, -2
        other_dims = [i for i in range(ndim) if i not in (ax1, ax2, ndim-1)]
        perm = other_dims + [ax1, ax2, ndim-1]
        x = x.permute(perm)
        
        batch_shape = x.shape[:-3]
        x = x.view(-1, dim1 * dim2, self.embed_dim)
        
        # Precompute inverse permutation
        inv_perm = [0] * len(perm)
        for i, p in enumerate(perm):
            inv_perm[p] = i
        
        return x, (batch_shape, dim1, dim2, inv_perm)
    
    def _unflatten_axes(self, x: torch.Tensor, info: Tuple) -> torch.Tensor:
        """Restore original shape after attention."""
        batch_shape, dim1, dim2, inv_perm = info
        x = x.view(*batch_shape, dim1, dim2, self.embed_dim)
        return x.permute(inv_perm)


"""
MultiAxisAttention: Attention over arbitrary tensor axis with backend selection.

Features:
- Multi-axis routing (any attention_axis)
- Backend dispatch (local/ring/mesh2d/auto)
- Causal masking with striped partitioning for ring
- Gradient checkpointing
- Variable sequence lengths
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Literal, Optional, Tuple, List, Union

from mosaic.context import MosaicContext
from mosaic.backends.local import local_attention
from mosaic.backends.ring import ring_attention, RING_AVAILABLE
from mosaic.backends.mesh2d import mesh2d_attention, Mesh2DContext, MESH2D_AVAILABLE


BackendType = Literal["local", "ring", "mesh2d", "auto"]


def _infer_backend(seq_len: int, world_size: int) -> str:
    """Auto-select backend based on sequence length and GPU count."""
    if world_size == 1 or seq_len < 8192:
        return "local"
    elif seq_len < 100000 or world_size <= 8:
        return "ring"
    else:
        return "mesh2d"


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
            - "auto": Automatically select based on seq length and GPU count
        mesh_shape: Required for mesh2d backend, (rows, cols) of GPU mesh
        causal: If True, use causal (autoregressive) masking
        use_checkpoint: If True, use gradient checkpointing to save memory
        
    Example:
        # Causal ring attention over axis 1
        attn = MultiAxisAttention(
            embed_dim=96, num_heads=4, attention_axis=1,
            backend="ring", causal=True
        )
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attention_axis: int,
        backend: BackendType = "local",
        mesh_shape: Optional[Tuple[int, int]] = None,
        causal: bool = False,
        use_checkpoint: bool = False,
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
        self.causal = causal
        self.use_checkpoint = use_checkpoint
        
        # Projections
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # Initialize mesh2d context if needed
        self._mesh2d_ctx = None
        if backend == "mesh2d":
            if mesh_shape is None:
                raise ValueError("mesh_shape required for mesh2d backend")
            self._mesh2d_ctx = Mesh2DContext(mesh_shape)
        
        # Select attention function (deferred for "auto")
        self._backend_resolved = backend if backend != "auto" else None
        if backend == "ring":
            if not RING_AVAILABLE:
                raise ImportError("ring-flash-attn not installed")
    
    def _get_attn_fn(self, seq_len: int):
        """Get attention function, resolving 'auto' if needed."""
        backend = self._backend_resolved
        if backend is None:
            ctx = MosaicContext.get()
            backend = _infer_backend(seq_len, ctx.world_size if ctx else 1)
        
        if backend == "ring":
            return self._ring_attention
        elif backend == "mesh2d":
            return self._mesh2d_attention
        else:
            return self._local_attention
    
    def forward(
        self, 
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        seq_lens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute attention over attention_axis.
        
        Args:
            x: N-dimensional tensor with embed_dim as last dimension
            mask: Optional attention mask (batch, seq, seq) or (batch, 1, seq, seq)
            seq_lens: Optional per-sample sequence lengths for variable-length batches
            
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
        
        # Step 3: Core attention (optionally checkpointed)
        if self.use_checkpoint and self.training:
            out = checkpoint(
                self._attention_core, x, mask, seq_lens, seq_len,
                use_reentrant=False
            )
        else:
            out = self._attention_core(x, mask, seq_lens, seq_len)
        
        # Step 4: Restore shape
        if batch_shape:
            out = out.view(*batch_shape, seq_len, self.embed_dim)
        if inv_perm:
            out = out.permute(inv_perm)
        
        return out
    
    def _attention_core(
        self, x: torch.Tensor, mask: Optional[torch.Tensor], 
        seq_lens: Optional[torch.Tensor], seq_len: int
    ) -> torch.Tensor:
        """Core attention computation (can be checkpointed)."""
        batch_size = x.shape[0]
        
        # Project Q, K, V
        qkv = self.qkv_proj(x).view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        
        # Get attention function
        attn_fn = self._get_attn_fn(seq_len)
        
        # Build mask if causal or custom
        attn_mask = self._build_mask(mask, seq_len, seq_lens, q.device, q.dtype)
        
        # Compute attention
        out = attn_fn(q, k, v, attn_mask)
        
        # Reshape and project
        out = out.transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
        return self.out_proj(out)
    
    def _build_mask(
        self, 
        mask: Optional[torch.Tensor],
        seq_len: int,
        seq_lens: Optional[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype
    ) -> Optional[torch.Tensor]:
        """Build attention mask from causal flag, custom mask, and seq_lens."""
        if not self.causal and mask is None and seq_lens is None:
            return None
        
        # Start with causal mask if needed
        if self.causal:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
                diagonal=1
            )
            attn_mask = causal_mask.float() * -1e9
        else:
            attn_mask = torch.zeros(seq_len, seq_len, device=device, dtype=dtype)
        
        # Apply custom mask
        if mask is not None:
            if mask.dtype == torch.bool:
                attn_mask = attn_mask + mask.float() * -1e9
            else:
                attn_mask = attn_mask + mask
        
        # Apply sequence length masking (for variable lengths)
        if seq_lens is not None:
            batch_size = seq_lens.shape[0]
            # Create position indices
            pos = torch.arange(seq_len, device=device)
            # Mask positions beyond each sample's length
            len_mask = pos.unsqueeze(0) >= seq_lens.unsqueeze(1)  # (batch, seq)
            # Expand to (batch, 1, 1, seq) for broadcasting
            len_mask = len_mask.unsqueeze(1).unsqueeze(2)
            attn_mask = attn_mask.unsqueeze(0) + len_mask.float() * -1e9
        
        return attn_mask
    
    def _permute_to_seq(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[List[int]]]:
        """Move attention_axis to position -2."""
        ndim = x.ndim
        ax = self.attention_axis if self.attention_axis >= 0 else ndim + self.attention_axis
        
        if ax == ndim - 2:
            return x, None
        
        perm = [i for i in range(ndim) if i != ax and i != ndim - 1]
        perm.append(ax)
        perm.append(ndim - 1)
        
        inv_perm = [0] * ndim
        for i, p in enumerate(perm):
            inv_perm[p] = i
        
        return x.permute(perm), inv_perm
    
    def _local_attention(self, q, k, v, mask=None) -> torch.Tensor:
        """Standard local attention."""
        return local_attention(q, k, v, attn_mask=mask, is_causal=self.causal and mask is None)
    
    def _ring_attention(self, q, k, v, mask=None) -> torch.Tensor:
        """Ring attention across GPUs."""
        ctx = MosaicContext.get()
        return ring_attention(q, k, v, group=ctx.sp_group, causal=self.causal)
    
    def _mesh2d_attention(self, q, k, v, mask=None) -> torch.Tensor:
        """2D mesh attention."""
        return mesh2d_attention(
            q, k, v,
            row_group=self._mesh2d_ctx.row_group,
            col_group=self._mesh2d_ctx.col_group,
            attn_mask=mask
        )


class CrossAxisAttention(nn.Module):
    """
    Cross-attention between two tensors with different sharding.
    
    Q comes from one tensor, K/V from another. They can have different
    axes and sharding strategies.
    
    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        q_axis: Attention axis for queries
        kv_axis: Attention axis for keys/values
        backend: Sharding backend
        
    Example:
        cross = CrossAxisAttention(
            embed_dim=128, num_heads=8,
            q_axis=1, kv_axis=2,
            backend="ring"
        )
        out = cross(q_source=features, kv_source=context)
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        q_axis: int,
        kv_axis: int,
        backend: BackendType = "local",
        causal: bool = False,
        use_checkpoint: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_axis = q_axis
        self.kv_axis = kv_axis
        self.backend = backend
        self.causal = causal
        self.use_checkpoint = use_checkpoint
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.kv_proj = nn.Linear(embed_dim, 2 * embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
    
    def forward(
        self,
        q_source: torch.Tensor,
        kv_source: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Cross-attention from q_source to kv_source.
        
        Args:
            q_source: Tensor for queries
            kv_source: Tensor for keys/values
            mask: Optional cross-attention mask
        """
        # Permute q_axis to seq position in q_source
        q_x, q_inv = self._permute_axis(q_source, self.q_axis)
        kv_x, kv_inv = self._permute_axis(kv_source, self.kv_axis)
        
        # Flatten batch dims
        q_batch_shape = q_x.shape[:-2]
        kv_batch_shape = kv_x.shape[:-2]
        q_seq = q_x.shape[-2]
        kv_seq = kv_x.shape[-2]
        
        q_x = q_x.view(-1, q_seq, self.embed_dim)
        kv_x = kv_x.view(-1, kv_seq, self.embed_dim)
        batch_size = q_x.shape[0]
        
        # Project
        q = self.q_proj(q_x).view(batch_size, q_seq, self.num_heads, self.head_dim)
        kv = self.kv_proj(kv_x).view(batch_size, kv_seq, 2, self.num_heads, self.head_dim)
        k, v = kv.permute(2, 0, 3, 1, 4).unbind(0)
        q = q.transpose(1, 2)  # (batch, heads, q_seq, head_dim)
        
        # Attention
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=self.causal)
        
        # Reshape and project
        out = out.transpose(1, 2).reshape(batch_size, q_seq, self.embed_dim)
        out = self.out_proj(out)
        
        # Restore shape
        if q_batch_shape:
            out = out.view(*q_batch_shape, q_seq, self.embed_dim)
        if q_inv:
            out = out.permute(q_inv)
        
        return out
    
    def _permute_axis(self, x: torch.Tensor, axis: int) -> Tuple[torch.Tensor, Optional[List[int]]]:
        """Permute axis to position -2."""
        ndim = x.ndim
        ax = axis if axis >= 0 else ndim + axis
        
        if ax == ndim - 2:
            return x, None
        
        perm = [i for i in range(ndim) if i != ax and i != ndim - 1]
        perm.extend([ax, ndim - 1])
        
        inv_perm = [0] * ndim
        for i, p in enumerate(perm):
            inv_perm[p] = i
        
        return x.permute(perm), inv_perm


class MultiAxisAttention2D(nn.Module):
    """
    Attention over TWO axes simultaneously with 2D mesh sharding.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        axis1: int,
        axis2: int,
        mesh_shape: Tuple[int, int],
        causal: bool = False,
        use_checkpoint: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.axis1 = axis1
        self.axis2 = axis2
        self.mesh_shape = mesh_shape
        self.causal = causal
        self.use_checkpoint = use_checkpoint
        
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self._mesh2d_ctx = Mesh2DContext(mesh_shape)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x, restore_info = self._flatten_axes(x)
        batch_size, seq_len = x.shape[0], x.shape[1]
        
        if self.use_checkpoint and self.training:
            out = checkpoint(self._attention_core, x, mask, use_reentrant=False)
        else:
            out = self._attention_core(x, mask)
        
        return self._unflatten_axes(out, restore_info)
    
    def _attention_core(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        batch_size, seq_len = x.shape[:2]
        
        qkv = self.qkv_proj(x).view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        
        out = mesh2d_attention(
            q, k, v, 
            self._mesh2d_ctx.row_group, 
            self._mesh2d_ctx.col_group,
            attn_mask=mask
        )
        
        out = out.transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
        return self.out_proj(out)
    
    def _flatten_axes(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple]:
        ndim = x.ndim
        ax1 = self.axis1 if self.axis1 >= 0 else ndim + self.axis1
        ax2 = self.axis2 if self.axis2 >= 0 else ndim + self.axis2
        if ax1 > ax2:
            ax1, ax2 = ax2, ax1
        
        dim1, dim2 = x.shape[ax1], x.shape[ax2]
        
        other_dims = [i for i in range(ndim) if i not in (ax1, ax2, ndim-1)]
        perm = other_dims + [ax1, ax2, ndim-1]
        x = x.permute(perm)
        
        batch_shape = x.shape[:-3]
        x = x.view(-1, dim1 * dim2, self.embed_dim)
        
        inv_perm = [0] * len(perm)
        for i, p in enumerate(perm):
            inv_perm[p] = i
        
        return x, (batch_shape, dim1, dim2, inv_perm)
    
    def _unflatten_axes(self, x: torch.Tensor, info: Tuple) -> torch.Tensor:
        batch_shape, dim1, dim2, inv_perm = info
        x = x.view(*batch_shape, dim1, dim2, self.embed_dim)
        return x.permute(inv_perm)

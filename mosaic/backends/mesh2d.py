"""
2D Mesh attention backend.

Shards attention computation across a 2D grid of GPUs.
Each GPU computes one tile of the attention matrix.

Example with 4 GPUs (2x2 mesh):
    Q sharded along rows (dim 0)
    K sharded along cols (dim 1)
    
         K0      K1
       ┌──────┬──────┐
    Q0 │GPU 0 │GPU 1 │
       ├──────┼──────┤
    Q1 │GPU 2 │GPU 3 │
       └──────┴──────┘
"""

import torch
import torch.distributed as dist
from typing import Optional, Tuple

# Check if MagiAttention is available
try:
    from magi_attention import magi_attn_func
    MAGI_AVAILABLE = True
except ImportError:
    MAGI_AVAILABLE = False
    magi_attn_func = None

MESH2D_AVAILABLE = True  # Our implementation always available


def create_2d_mesh(
    mesh_shape: Tuple[int, int],
    device_ids: Optional[list] = None
) -> Tuple[dist.ProcessGroup, dist.ProcessGroup, int, int]:
    """
    Create 2D mesh process groups.
    
    Args:
        mesh_shape: (rows, cols) of the mesh
        device_ids: Optional list of device IDs (defaults to range(world_size))
        
    Returns:
        (row_group, col_group, row_rank, col_rank)
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    rows, cols = mesh_shape
    assert rows * cols == world_size, f"Mesh {mesh_shape} != world_size {world_size}"
    
    if device_ids is None:
        device_ids = list(range(world_size))
    
    # Reshape device IDs into mesh
    mesh = torch.tensor(device_ids).reshape(rows, cols)
    
    # Find this rank's position in mesh
    pos = (mesh == rank).nonzero()
    row_idx, col_idx = pos[0, 0].item(), pos[0, 1].item()
    
    # Create row groups (GPUs in same row communicate)
    row_groups = []
    for r in range(rows):
        ranks_in_row = mesh[r, :].tolist()
        group = dist.new_group(ranks_in_row)
        row_groups.append(group)
    
    # Create col groups (GPUs in same column communicate)
    col_groups = []
    for c in range(cols):
        ranks_in_col = mesh[:, c].tolist()
        group = dist.new_group(ranks_in_col)
        col_groups.append(group)
    
    return row_groups[row_idx], col_groups[col_idx], row_idx, col_idx


def mesh2d_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    row_group: dist.ProcessGroup,
    col_group: dist.ProcessGroup,
    attn_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    **kwargs
) -> torch.Tensor:
    """
    2D mesh attention using all-gather along cols + FlashAttention.
    
    Each GPU has:
        q: (batch, heads, seq_local_q, head_dim) - local Q shard
        k: (batch, heads, seq_local_k, head_dim) - local K shard
        v: (batch, heads, seq_local_k, head_dim) - local V shard
    
    Algorithm:
        1. All-gather K, V along column group (get full K, V for this Q shard)
        2. Compute attention via F.scaled_dot_product_attention (uses FlashAttention)
        3. Result is already correctly sharded
    
    Args:
        q: Query tensor (batch, heads, seq_local, head_dim)
        k: Key tensor (batch, heads, seq_local, head_dim)
        v: Value tensor (batch, heads, seq_local, head_dim)
        row_group: Process group for row communication
        col_group: Process group for column communication
        attn_mask: Optional attention mask
        is_causal: If True, use causal masking
        
    Returns:
        Attention output (batch, heads, seq_local, head_dim)
    """
    import torch.nn.functional as F
    
    col_world_size = dist.get_world_size(col_group)
    
    if col_world_size == 1:
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, 
            is_causal=is_causal and attn_mask is None
        )
    
    # All-gather K, V along column dimension
    k_gathered = _all_gather_along_seq(k, col_group, col_world_size)
    v_gathered = _all_gather_along_seq(v, col_group, col_world_size)
    
    # For causal with gathered K,V, need to adjust mask for local Q position
    # TODO: Handle causal masking properly for 2D mesh
    return F.scaled_dot_product_attention(
        q, k_gathered, v_gathered,
        attn_mask=attn_mask,
        is_causal=is_causal and attn_mask is None
    )


def _all_gather_along_seq(
    x: torch.Tensor, 
    group: dist.ProcessGroup,
    world_size: int
) -> torch.Tensor:
    """All-gather tensor along sequence dimension."""
    if world_size == 1:
        return x
    
    # x: (batch, heads, seq_local, head_dim)
    batch, heads, seq_local, head_dim = x.shape
    
    # Pre-allocate output tensor
    output = torch.empty(
        batch, heads, seq_local * world_size, head_dim,
        dtype=x.dtype, device=x.device
    )
    
    # Gather into pre-allocated slices (avoids torch.cat allocation)
    output_list = list(output.chunk(world_size, dim=2))
    dist.all_gather(output_list, x.contiguous(), group=group)
    
    return output


class Mesh2DContext:
    """Context for 2D mesh parallelism."""
    
    def __init__(self, mesh_shape: Tuple[int, int]):
        """
        Initialize 2D mesh context.
        
        Args:
            mesh_shape: (rows, cols) shape of GPU mesh
        """
        self.mesh_shape = mesh_shape
        self.row_group, self.col_group, self.row_rank, self.col_rank = create_2d_mesh(mesh_shape)
        self.row_size = mesh_shape[0]
        self.col_size = mesh_shape[1]
    
    def shard_tensor(self, x: torch.Tensor, shard_dims: Tuple[int, int]) -> torch.Tensor:
        """
        Get local shard of tensor for this GPU's position in mesh.
        
        Args:
            x: Full tensor
            shard_dims: (row_dim, col_dim) - which dims to shard
            
        Returns:
            Local shard for this GPU
        """
        row_dim, col_dim = shard_dims
        
        # Shard along row dimension
        row_size = x.shape[row_dim] // self.row_size
        row_start = self.row_rank * row_size
        row_end = row_start + row_size
        
        # Shard along col dimension  
        col_size = x.shape[col_dim] // self.col_size
        col_start = self.col_rank * col_size
        col_end = col_start + col_size
        
        # Build slice
        slices = [slice(None)] * x.ndim
        slices[row_dim] = slice(row_start, row_end)
        slices[col_dim] = slice(col_start, col_end)
        
        return x[tuple(slices)].contiguous()


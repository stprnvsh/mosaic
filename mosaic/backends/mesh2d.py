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
    **kwargs
) -> torch.Tensor:
    """
    2D mesh attention using all-gather along rows and cols.
    
    Each GPU has:
        q: (batch, heads, seq_local_q, head_dim) - local Q shard
        k: (batch, heads, seq_local_k, head_dim) - local K shard
        v: (batch, heads, seq_local_k, head_dim) - local V shard
    
    Algorithm:
        1. All-gather K, V along column group (get full K, V for this Q shard)
        2. Compute local attention
        3. Result is already correctly sharded
    
    Args:
        q: Query tensor (batch, heads, seq_local, head_dim)
        k: Key tensor (batch, heads, seq_local, head_dim)
        v: Value tensor (batch, heads, seq_local, head_dim)
        row_group: Process group for row communication
        col_group: Process group for column communication
        
    Returns:
        Attention output (batch, heads, seq_local, head_dim)
    """
    # Get sizes
    col_world_size = dist.get_world_size(col_group)
    
    # All-gather K, V along column dimension
    # Each GPU in the same row needs all K, V from its column
    k_gathered = _all_gather_along_seq(k, col_group)
    v_gathered = _all_gather_along_seq(v, col_group)
    
    # Compute attention with local Q against gathered K, V
    # Using scaled dot-product attention
    scale = q.shape[-1] ** -0.5
    attn_weights = torch.matmul(q, k_gathered.transpose(-2, -1)) * scale
    attn_weights = torch.softmax(attn_weights, dim=-1)
    output = torch.matmul(attn_weights, v_gathered)
    
    return output


def _all_gather_along_seq(x: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    """All-gather tensor along sequence dimension."""
    world_size = dist.get_world_size(group)
    
    if world_size == 1:
        return x
    
    # x: (batch, heads, seq_local, head_dim)
    gathered = [torch.empty_like(x) for _ in range(world_size)]
    dist.all_gather(gathered, x, group=group)
    
    # Concatenate along sequence dimension
    return torch.cat(gathered, dim=2)


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


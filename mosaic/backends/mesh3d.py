"""
3D Mesh attention backend.

For models with 3D structure: video (time, h, w), weather (time, lat, lon),
medical imaging (depth, h, w).

Shards across a 3D grid of GPUs.
"""

import torch
import torch.nn.functional as F
import torch.distributed as dist
from typing import Optional, Tuple, Dict


def create_3d_mesh(
    mesh_shape: Tuple[int, int, int],
) -> Dict[str, dist.ProcessGroup]:
    """
    Create 3D mesh process groups.
    
    Args:
        mesh_shape: (dim0, dim1, dim2) of the mesh
        
    Returns:
        Dict with groups for each dimension and position info
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    d0, d1, d2 = mesh_shape
    assert d0 * d1 * d2 == world_size, f"Mesh {mesh_shape} != world_size {world_size}"
    
    # Convert rank to 3D position
    pos2 = rank % d2
    pos1 = (rank // d2) % d1
    pos0 = rank // (d1 * d2)
    
    # Create groups for each dimension
    # Dim 0 groups: GPUs with same (pos1, pos2)
    dim0_groups = {}
    for i1 in range(d1):
        for i2 in range(d2):
            ranks = [i0 * d1 * d2 + i1 * d2 + i2 for i0 in range(d0)]
            group = dist.new_group(ranks)
            dim0_groups[(i1, i2)] = group
    
    # Dim 1 groups: GPUs with same (pos0, pos2)
    dim1_groups = {}
    for i0 in range(d0):
        for i2 in range(d2):
            ranks = [i0 * d1 * d2 + i1 * d2 + i2 for i1 in range(d1)]
            group = dist.new_group(ranks)
            dim1_groups[(i0, i2)] = group
    
    # Dim 2 groups: GPUs with same (pos0, pos1)
    dim2_groups = {}
    for i0 in range(d0):
        for i1 in range(d1):
            ranks = [i0 * d1 * d2 + i1 * d2 + i2 for i2 in range(d2)]
            group = dist.new_group(ranks)
            dim2_groups[(i0, i1)] = group
    
    return {
        'dim0_group': dim0_groups[(pos1, pos2)],
        'dim1_group': dim1_groups[(pos0, pos2)],
        'dim2_group': dim2_groups[(pos0, pos1)],
        'pos': (pos0, pos1, pos2),
        'shape': mesh_shape,
    }


class Mesh3DContext:
    """Context for 3D mesh parallelism."""
    
    def __init__(self, mesh_shape: Tuple[int, int, int]):
        self.mesh_shape = mesh_shape
        info = create_3d_mesh(mesh_shape)
        
        self.dim0_group = info['dim0_group']
        self.dim1_group = info['dim1_group']
        self.dim2_group = info['dim2_group']
        self.pos = info['pos']
        self.d0, self.d1, self.d2 = mesh_shape
    
    def shard_tensor(
        self, 
        x: torch.Tensor, 
        shard_dims: Tuple[int, int, int]
    ) -> torch.Tensor:
        """
        Get local shard of tensor for this GPU's position.
        
        Args:
            x: Full tensor
            shard_dims: Which tensor dims map to mesh dims (tensor_dim0, tensor_dim1, tensor_dim2)
        """
        slices = [slice(None)] * x.ndim
        
        for mesh_idx, tensor_dim in enumerate(shard_dims):
            mesh_size = self.mesh_shape[mesh_idx]
            pos = self.pos[mesh_idx]
            size = x.shape[tensor_dim] // mesh_size
            start = pos * size
            slices[tensor_dim] = slice(start, start + size)
        
        return x[tuple(slices)].contiguous()


def mesh3d_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ctx: Mesh3DContext,
    gather_dims: Tuple[int, ...] = (1, 2),
    attn_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
) -> torch.Tensor:
    """
    3D mesh attention.
    
    Gathers K, V along specified mesh dimensions before computing attention.
    
    Args:
        q: Query tensor (batch, heads, seq_local, head_dim)
        k: Key tensor (batch, heads, seq_local, head_dim)
        v: Value tensor (batch, heads, seq_local, head_dim)
        ctx: 3D mesh context
        gather_dims: Which mesh dimensions to gather K, V along (0, 1, or 2)
        attn_mask: Optional attention mask
        is_causal: If True, use causal masking
    """
    # Gather K, V along specified dimensions
    k_gathered = k
    v_gathered = v
    
    for dim in gather_dims:
        if dim == 0:
            group = ctx.dim0_group
            world_size = ctx.d0
        elif dim == 1:
            group = ctx.dim1_group
            world_size = ctx.d1
        else:
            group = ctx.dim2_group
            world_size = ctx.d2
        
        if world_size > 1:
            k_gathered = _all_gather_along_seq(k_gathered, group, world_size)
            v_gathered = _all_gather_along_seq(v_gathered, group, world_size)
    
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
    
    batch, heads, seq_local, head_dim = x.shape
    
    output = torch.empty(
        batch, heads, seq_local * world_size, head_dim,
        dtype=x.dtype, device=x.device
    )
    
    output_list = list(output.chunk(world_size, dim=2))
    dist.all_gather(output_list, x.contiguous(), group=group)
    
    return output


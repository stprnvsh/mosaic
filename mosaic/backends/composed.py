"""
Composed attention backend.

Combines multiple sharding strategies for complex attention patterns.

Example: Ring + Mesh2D
    - Use ring for sequence dimension (rows)
    - Use mesh2d for head dimension
    
Example: Hierarchical
    - Intra-node: local attention
    - Inter-node: ring attention
"""

import torch
import torch.nn.functional as F
import torch.distributed as dist
from typing import Optional, List, Tuple, Literal
from dataclasses import dataclass

from mosaic.backends.mesh2d import create_2d_mesh, mesh2d_attention
from mosaic.backends.ring import ring_attention, RING_AVAILABLE


@dataclass
class ShardingSpec:
    """Specification for how to shard a dimension."""
    dim: int                    # Which dimension to shard
    strategy: str               # "ring", "mesh2d", "local"
    group: Optional[dist.ProcessGroup] = None


class ComposedAttention:
    """
    Compose multiple sharding strategies.
    
    This allows complex patterns like:
    - Shard heads across one mesh dimension
    - Shard sequence across another (ring)
    
    Example:
        # 8 GPUs: 2-way head parallel, 4-way sequence parallel
        composed = ComposedAttention(
            mesh_shape=(2, 4),
            head_strategy="mesh_row",    # Shard heads across rows
            seq_strategy="mesh_col_ring" # Ring within each column
        )
    """
    
    def __init__(
        self,
        mesh_shape: Tuple[int, int],
        head_parallel: bool = False,
        seq_parallel: str = "ring"  # "ring", "mesh2d", "none"
    ):
        """
        Initialize composed attention.
        
        Args:
            mesh_shape: (rows, cols) GPU mesh shape
            head_parallel: Whether to shard attention heads across mesh rows
            seq_parallel: How to shard sequence ("ring" across cols, "mesh2d", or "none")
        """
        self.mesh_shape = mesh_shape
        self.head_parallel = head_parallel
        self.seq_parallel = seq_parallel
        
        # Create mesh groups once at init
        self.row_group, self.col_group, self.row_rank, self.col_rank = create_2d_mesh(mesh_shape)
        
        self.row_size = mesh_shape[0]
        self.col_size = mesh_shape[1]
        
        # Pre-select attention function (avoid branching in forward)
        if seq_parallel == "ring":
            self._attn_fn = self._ring_attention
        elif seq_parallel == "mesh2d":
            self._attn_fn = self._mesh2d_attention
        else:
            self._attn_fn = self._local_attention
    
    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute attention with composed sharding.
        
        Args:
            q: (batch, heads_local, seq_local, head_dim)
            k: (batch, heads_local, seq_local, head_dim)
            v: (batch, heads_local, seq_local, head_dim)
        """
        return self._attn_fn(q, k, v)
    
    def _local_attention(self, q, k, v):
        """Standard local attention."""
        return F.scaled_dot_product_attention(q, k, v)
    
    def _ring_attention(self, q, k, v):
        """Ring attention within column group."""
        return ring_attention(q, k, v, group=self.col_group)
    
    def _mesh2d_attention(self, q, k, v):
        """2D mesh attention."""
        return mesh2d_attention(q, k, v, self.row_group, self.col_group)
    
    def shard_qkv(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Shard Q, K, V according to mesh position.
        
        Input shapes: (batch, heads, seq, head_dim)
        
        If head_parallel: shard heads across rows
        If seq_parallel: shard seq across cols (handled by ring/mesh2d)
        """
        if self.head_parallel:
            # Shard heads across row dimension of mesh
            heads = q.shape[1]
            heads_per_gpu = heads // self.row_size
            start = self.row_rank * heads_per_gpu
            end = start + heads_per_gpu
            
            q = q[:, start:end, :, :].contiguous()
            k = k[:, start:end, :, :].contiguous()
            v = v[:, start:end, :, :].contiguous()
        
        if self.seq_parallel != "none":
            # Shard sequence across column dimension
            seq_len = q.shape[2]
            seq_per_gpu = seq_len // self.col_size
            start = self.col_rank * seq_per_gpu
            end = start + seq_per_gpu
            
            q = q[:, :, start:end, :].contiguous()
            k = k[:, :, start:end, :].contiguous()
            v = v[:, :, start:end, :].contiguous()
        
        return q, k, v
    
    def gather_output(self, out: torch.Tensor, gather_heads: bool = True) -> torch.Tensor:
        """
        Gather output from all GPUs.
        
        Args:
            out: Local output (batch, heads_local, seq_local, head_dim)
            gather_heads: Whether to gather heads (if head_parallel)
            
        Returns:
            Full output tensor
        """
        if self.head_parallel and gather_heads:
            # All-gather along head dimension across rows
            gathered = [torch.empty_like(out) for _ in range(self.row_size)]
            dist.all_gather(gathered, out, group=self.row_group)
            out = torch.cat(gathered, dim=1)
        
        # Sequence gathering is handled by ring/mesh2d backends
        return out


class HierarchicalAttention:
    """
    Hierarchical attention for multi-level parallelism.
    
    Useful for multi-node setups:
    - Intra-node: Fast local/NVLink communication
    - Inter-node: Slower network communication
    
    Example:
        # 4 nodes × 8 GPUs = 32 GPUs
        hier = HierarchicalAttention(
            intra_node_size=8,
            inter_node_strategy="ring",
            intra_node_strategy="local"  # or "mesh2d"
        )
    """
    
    def __init__(
        self,
        intra_node_size: int,
        inter_node_strategy: Literal["ring", "mesh2d"] = "ring",
        intra_node_strategy: Literal["local", "mesh2d"] = "local"
    ):
        self.intra_node_size = intra_node_size
        self.inter_node_strategy = inter_node_strategy
        self.intra_node_strategy = intra_node_strategy
        
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        
        num_nodes = world_size // intra_node_size
        self.node_id = rank // intra_node_size
        self.local_rank = rank % intra_node_size
        self.leader_src = self.node_id * intra_node_size
        
        # Create intra-node groups
        intra_ranks = list(range(self.leader_src, self.leader_src + intra_node_size))
        self.intra_group = dist.new_group(intra_ranks)
        
        # Create inter-node groups (one representative per node)
        inter_ranks = [i * intra_node_size for i in range(num_nodes)]
        self.inter_group = dist.new_group(inter_ranks)
        self.is_node_leader = (self.local_rank == 0)
        
        # Pre-select intra-node attention function
        if intra_node_strategy == "mesh2d":
            self._intra_attn = lambda q, k, v: mesh2d_attention(q, k, v, self.intra_group, self.intra_group)
        else:
            self._intra_attn = F.scaled_dot_product_attention
    
    def __call__(self, q, k, v):
        """
        Two-level attention:
        1. Intra-node attention (fast)
        2. Inter-node aggregation (if needed)
        """
        # Intra-node attention
        local_out = self._intra_attn(q, k, v)
        
        # Inter-node: ring attention between node leaders
        if self.is_node_leader and self.inter_node_strategy == "ring":
            out = ring_attention(local_out, local_out, local_out, group=self.inter_group)
        else:
            out = local_out
        
        # Broadcast from leader to rest of node
        dist.broadcast(out, src=self.leader_src, group=self.intra_group)
        
        return out


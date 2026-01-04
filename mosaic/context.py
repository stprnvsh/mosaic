"""
MosaicContext: Global context for multi-axis parallelism.
Thin wrapper around PyTorch's DeviceMesh.
"""

import torch
import torch.distributed as dist
from typing import Optional


class MosaicContext:
    """Singleton managing parallel execution state."""
    
    _instance: Optional["MosaicContext"] = None
    
    def __init__(self, sp_size: int = 1):
        """
        Initialize Mosaic context.
        
        Args:
            sp_size: Sequence parallel size (number of GPUs to shard sequences across)
        """
        if not dist.is_initialized():
            dist.init_process_group("nccl")
        
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.sp_size = sp_size
        self.sp_rank = self.rank % sp_size
        
        # Use PyTorch's DeviceMesh if available (PyTorch 2.0+)
        try:
            from torch.distributed.device_mesh import init_device_mesh
            self.mesh = init_device_mesh("cuda", (sp_size,), mesh_dim_names=("sp",))
            self._sp_group = self.mesh.get_group("sp")
        except ImportError:
            # Fallback for older PyTorch
            self.mesh = None
            self._sp_group = dist.group.WORLD if sp_size == self.world_size else None
        
        self.device = torch.device(f"cuda:{self.rank}")
        torch.cuda.set_device(self.device)
    
    @property
    def sp_group(self):
        """Get the sequence parallel process group."""
        return self._sp_group
    
    @classmethod
    def init(cls, sp_size: int = 1) -> "MosaicContext":
        """Initialize and return the global context."""
        if cls._instance is not None:
            raise RuntimeError("MosaicContext already initialized. Call get_context() instead.")
        cls._instance = cls(sp_size=sp_size)
        return cls._instance
    
    @classmethod
    def get(cls) -> "MosaicContext":
        """Get the global context (must be initialized first)."""
        if cls._instance is None:
            raise RuntimeError("MosaicContext not initialized. Call mosaic.init() first.")
        return cls._instance
    
    @classmethod
    def reset(cls):
        """Reset the global context (for testing)."""
        cls._instance = None


def init(sp_size: int = 1) -> MosaicContext:
    """
    Initialize Mosaic with sequence parallel size.
    
    Args:
        sp_size: Number of GPUs to shard sequences across
        
    Returns:
        MosaicContext instance
        
    Example:
        ctx = mosaic.init(sp_size=4)  # 4-way sequence parallel
    """
    return MosaicContext.init(sp_size=sp_size)


def get_context() -> MosaicContext:
    """Get the global Mosaic context."""
    return MosaicContext.get()


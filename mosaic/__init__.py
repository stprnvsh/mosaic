"""
Mosaic: Multi-Axis Attention Sharding for PyTorch

A lightweight coordination layer for models with multiple attention axes.

Features:
- Multi-axis routing (any attention_axis)
- Backend dispatch (local/ring/mesh2d/mesh3d/auto)
- Causal masking with striped partitioning
- Gradient checkpointing
- Variable sequence lengths
- Cross-attention support
- HuggingFace integration

Backends:
- local: Standard attention, no communication
- ring: 1D sequence sharding with ring communication
- mesh2d: 2D mesh sharding for large attention matrices
- mesh3d: 3D mesh for video/weather/medical
- auto: Automatic backend selection
"""

from mosaic.context import MosaicContext, init, get_context
from mosaic.attention import (
    MultiAxisAttention, 
    MultiAxisAttention2D,
    CrossAxisAttention,
)
from mosaic.spec import AxisSpec
from mosaic.backends import get_available_backends
from mosaic.backends.composed import ComposedAttention, HierarchicalAttention
from mosaic.backends.mesh2d import Mesh2DContext
from mosaic.backends.mesh3d import Mesh3DContext

__version__ = "0.2.0"  # Keep in sync with pyproject.toml

__all__ = [
    # Core
    "MosaicContext",
    "MultiAxisAttention",
    "MultiAxisAttention2D",
    "CrossAxisAttention",
    "AxisSpec",
    "init",
    "get_context",
    # Backends
    "get_available_backends",
    "ComposedAttention",
    "HierarchicalAttention",
    "Mesh2DContext",
    "Mesh3DContext",
]

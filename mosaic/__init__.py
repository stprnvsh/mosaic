"""
Mosaic: Multi-Axis Attention Sharding for PyTorch

A lightweight coordination layer for models with multiple attention axes.

Backends:
- local: Standard attention, no communication
- ring: 1D sequence sharding with ring communication
- mesh2d: 2D mesh sharding for large attention matrices
- composed: Combine multiple backends

Classes:
- MultiAxisAttention: Attention over any single axis
- MultiAxisAttention2D: Attention over two axes simultaneously
- ComposedAttention: Combine ring + mesh2d + head parallel
"""

from mosaic.context import MosaicContext, init, get_context
from mosaic.attention import MultiAxisAttention, MultiAxisAttention2D
from mosaic.spec import AxisSpec
from mosaic.backends import get_available_backends
from mosaic.backends.composed import ComposedAttention, HierarchicalAttention
from mosaic.backends.mesh2d import Mesh2DContext

__version__ = "0.1.0"
__all__ = [
    # Core
    "MosaicContext",
    "MultiAxisAttention",
    "MultiAxisAttention2D",
    "AxisSpec",
    "init",
    "get_context",
    # Backends
    "get_available_backends",
    "ComposedAttention",
    "HierarchicalAttention",
    "Mesh2DContext",
]


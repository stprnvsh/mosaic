"""
Attention backends for Mosaic.

Detects available backends and provides unified interface.

Backends:
- local: Standard attention, no communication
- ring: 1D sequence sharding with ring communication
- mesh2d: 2D mesh sharding for large attention matrices
- composed: Combine multiple backends (e.g., ring + mesh2d + head parallel)
"""

from mosaic.backends.local import local_attention
from mosaic.backends.ring import ring_attention, RING_AVAILABLE
from mosaic.backends.mesh2d import mesh2d_attention, Mesh2DContext, MESH2D_AVAILABLE
from mosaic.backends.composed import ComposedAttention, HierarchicalAttention


def get_available_backends() -> list[str]:
    """Return list of available backends."""
    backends = ["local", "mesh2d", "composed"]
    if RING_AVAILABLE:
        backends.append("ring")
    return backends


__all__ = [
    "local_attention",
    "ring_attention",
    "mesh2d_attention",
    "Mesh2DContext",
    "ComposedAttention",
    "HierarchicalAttention",
    "RING_AVAILABLE",
    "MESH2D_AVAILABLE",
    "get_available_backends",
]


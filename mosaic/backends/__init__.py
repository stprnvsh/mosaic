"""
Attention backends for Mosaic.

Detects available backends and provides unified interface.
"""

from mosaic.backends.local import local_attention
from mosaic.backends.ring import ring_attention, RING_AVAILABLE


def get_available_backends() -> list[str]:
    """Return list of available backends."""
    backends = ["local"]
    if RING_AVAILABLE:
        backends.append("ring")
    return backends


__all__ = [
    "local_attention",
    "ring_attention",
    "RING_AVAILABLE",
    "get_available_backends",
]


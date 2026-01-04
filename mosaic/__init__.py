"""
Mosaic: Multi-Axis Attention Sharding for PyTorch

A lightweight coordination layer for models with multiple attention axes.
"""

from mosaic.context import MosaicContext, init, get_context
from mosaic.attention import MultiAxisAttention
from mosaic.spec import AxisSpec

__version__ = "0.1.0"
__all__ = [
    "MosaicContext",
    "MultiAxisAttention",
    "AxisSpec",
    "init",
    "get_context",
]


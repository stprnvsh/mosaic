"""
AxisSpec: Specification for attention axis sharding.
"""

from dataclasses import dataclass
from typing import Literal


BackendType = Literal["local", "ring", "magi"]


@dataclass
class AxisSpec:
    """
    Specification for how an attention axis should be handled.
    
    Attributes:
        axis: Tensor axis to compute attention over
        backend: Sharding backend to use
            - "local": Standard attention, no communication
            - "ring": Ring attention via ring-flash-attn
            - "magi": 2D tile attention via MagiAttention (not yet implemented)
    """
    axis: int
    backend: BackendType = "local"
    
    def __post_init__(self):
        if self.backend not in ("local", "ring", "magi"):
            raise ValueError(f"Unknown backend: {self.backend}. Use 'local', 'ring', or 'magi'.")
        if self.backend == "magi":
            raise NotImplementedError("MagiAttention backend not yet implemented.")


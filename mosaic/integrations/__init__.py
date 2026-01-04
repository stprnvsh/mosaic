"""
Framework integrations for Mosaic.

- HuggingFace: patch_attention() for transformers models
- PyTorch Lightning: MosaicCallback for distributed training
"""

from mosaic.integrations.huggingface import patch_attention, unpatch_attention

__all__ = ["patch_attention", "unpatch_attention"]


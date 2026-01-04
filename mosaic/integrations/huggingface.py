"""
HuggingFace Transformers integration.

Patches attention layers in HuggingFace models to use Mosaic sharding.

Usage:
    from transformers import AutoModel
    import mosaic
    from mosaic.integrations import patch_attention
    
    model = AutoModel.from_pretrained("bert-base-uncased")
    patch_attention(model, backend="ring")
    
    # Now attention uses ring attention across GPUs
    out = model(input_ids)
"""

import torch
import torch.nn as nn
from typing import Optional, List
from functools import wraps

from mosaic.backends.local import local_attention
from mosaic.backends.ring import ring_attention, RING_AVAILABLE
from mosaic.context import MosaicContext


# Store original methods for unpatching
_original_methods = {}


def patch_attention(
    model: nn.Module,
    backend: str = "ring",
    layer_names: Optional[List[str]] = None,
    causal: bool = False,
) -> nn.Module:
    """
    Patch attention layers in a HuggingFace model to use Mosaic.
    
    Args:
        model: HuggingFace model
        backend: Mosaic backend ("local", "ring", "mesh2d")
        layer_names: Optional list of layer names to patch. If None, patches all
                     layers containing "attention" or "attn"
        causal: If True, use causal masking
        
    Returns:
        Patched model (same object, modified in-place)
        
    Example:
        model = AutoModel.from_pretrained("gpt2")
        patch_attention(model, backend="ring", causal=True)
    """
    patched_count = 0
    
    for name, module in model.named_modules():
        # Skip if layer_names specified and this isn't in the list
        if layer_names and not any(ln in name for ln in layer_names):
            continue
        
        # Look for common attention patterns
        if _is_attention_module(module):
            _patch_module(module, backend, causal, name)
            patched_count += 1
    
    if patched_count == 0:
        import warnings
        warnings.warn(
            "No attention layers found to patch. "
            "Specify layer_names or check model architecture."
        )
    
    return model


def unpatch_attention(model: nn.Module) -> nn.Module:
    """
    Restore original attention methods.
    
    Args:
        model: Previously patched model
        
    Returns:
        Unpatched model
    """
    for name, module in model.named_modules():
        module_id = id(module)
        if module_id in _original_methods:
            for attr, original in _original_methods[module_id].items():
                setattr(module, attr, original)
            del _original_methods[module_id]
    
    return model


def _is_attention_module(module: nn.Module) -> bool:
    """Check if module is likely an attention layer."""
    class_name = module.__class__.__name__.lower()
    
    # Common attention class patterns
    attention_patterns = [
        'attention', 'multiheadattention', 'selfattention',
        'crossattention', 'mha', 'sdpa'
    ]
    
    if any(p in class_name for p in attention_patterns):
        return True
    
    # Check for scaled_dot_product_attention calls
    if hasattr(module, 'forward'):
        import inspect
        try:
            source = inspect.getsource(module.forward)
            if 'scaled_dot_product_attention' in source or 'softmax' in source:
                return True
        except (TypeError, OSError):
            pass
    
    return False


def _patch_module(module: nn.Module, backend: str, causal: bool, name: str):
    """Patch a single attention module."""
    module_id = id(module)
    _original_methods[module_id] = {}
    
    # Try different patching strategies
    if _try_patch_sdpa(module, backend, causal, module_id):
        return
    
    if _try_patch_forward(module, backend, causal, module_id):
        return


def _try_patch_sdpa(module: nn.Module, backend: str, causal: bool, module_id: int) -> bool:
    """
    Patch by intercepting torch.nn.functional.scaled_dot_product_attention calls.
    """
    import torch.nn.functional as F
    
    # Check if module uses SDPA
    if not hasattr(module, 'forward'):
        return False
    
    original_forward = module.forward
    _original_methods[module_id]['forward'] = original_forward
    
    @wraps(original_forward)
    def patched_forward(*args, **kwargs):
        # Temporarily replace F.scaled_dot_product_attention
        original_sdpa = F.scaled_dot_product_attention
        
        def mosaic_sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, **kw):
            # Route through Mosaic backend
            if backend == "ring" and RING_AVAILABLE:
                ctx = MosaicContext.get()
                if ctx and ctx.world_size > 1:
                    return ring_attention(q, k, v, group=ctx.sp_group, causal=causal or is_causal)
            
            # Fallback to local
            return local_attention(q, k, v, attn_mask=attn_mask, is_causal=causal or is_causal)
        
        F.scaled_dot_product_attention = mosaic_sdpa
        try:
            result = original_forward(*args, **kwargs)
        finally:
            F.scaled_dot_product_attention = original_sdpa
        
        return result
    
    module.forward = patched_forward
    return True


def _try_patch_forward(module: nn.Module, backend: str, causal: bool, module_id: int) -> bool:
    """
    Patch by wrapping forward method entirely.
    For modules that don't use SDPA.
    """
    # This is a fallback - most HF models now use SDPA
    return False


class MosaicLightningCallback:
    """
    PyTorch Lightning callback for Mosaic initialization.
    
    Usage:
        from mosaic.integrations.huggingface import MosaicLightningCallback
        
        trainer = L.Trainer(
            callbacks=[MosaicLightningCallback(sp_size=4)]
        )
    """
    
    def __init__(self, sp_size: int = 1):
        self.sp_size = sp_size
        self.ctx = None
    
    def setup(self, trainer, pl_module, stage):
        """Initialize Mosaic context."""
        import mosaic
        self.ctx = mosaic.init(sp_size=self.sp_size)
    
    def teardown(self, trainer, pl_module, stage):
        """Cleanup Mosaic context."""
        pass


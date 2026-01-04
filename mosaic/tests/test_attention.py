"""Tests for MultiAxisAttention."""

import pytest
import torch

from mosaic.attention import MultiAxisAttention


class TestAxisPermutation:
    """Test axis permutation logic."""
    
    def test_3d_tensor_axis_0(self):
        """Attention over axis 0 of 3D tensor."""
        attn = MultiAxisAttention(embed_dim=32, num_heads=4, attention_axis=0, backend="local")
        x = torch.randn(8, 4, 32)  # (seq, batch, embed)
        out = attn(x)
        assert out.shape == x.shape
    
    def test_3d_tensor_axis_1(self):
        """Attention over axis 1 of 3D tensor (standard case)."""
        attn = MultiAxisAttention(embed_dim=32, num_heads=4, attention_axis=1, backend="local")
        x = torch.randn(4, 8, 32)  # (batch, seq, embed)
        out = attn(x)
        assert out.shape == x.shape
    
    def test_4d_tensor_axis_1(self):
        """Attention over axis 1 of 4D tensor (nanoTabPFN rows)."""
        attn = MultiAxisAttention(embed_dim=32, num_heads=4, attention_axis=1, backend="local")
        x = torch.randn(2, 100, 5, 32)  # (batch, rows, features, embed)
        out = attn(x)
        assert out.shape == x.shape
    
    def test_4d_tensor_axis_2(self):
        """Attention over axis 2 of 4D tensor (nanoTabPFN features)."""
        attn = MultiAxisAttention(embed_dim=32, num_heads=4, attention_axis=2, backend="local")
        x = torch.randn(2, 100, 5, 32)  # (batch, rows, features, embed)
        out = attn(x)
        assert out.shape == x.shape
    
    def test_negative_axis(self):
        """Negative axis indexing."""
        attn = MultiAxisAttention(embed_dim=32, num_heads=4, attention_axis=-2, backend="local")
        x = torch.randn(2, 100, 5, 32)
        out = attn(x)
        assert out.shape == x.shape
    
    def test_5d_tensor(self):
        """5D tensor support."""
        attn = MultiAxisAttention(embed_dim=32, num_heads=4, attention_axis=2, backend="local")
        x = torch.randn(2, 4, 8, 5, 32)
        out = attn(x)
        assert out.shape == x.shape


class TestGradients:
    """Test gradient flow."""
    
    def test_gradients_flow(self):
        """Gradients should flow through attention."""
        attn = MultiAxisAttention(embed_dim=32, num_heads=4, attention_axis=1, backend="local")
        x = torch.randn(2, 8, 32, requires_grad=True)
        out = attn(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape
    
    def test_gradients_4d(self):
        """Gradients for 4D tensor."""
        attn = MultiAxisAttention(embed_dim=32, num_heads=4, attention_axis=1, backend="local")
        x = torch.randn(2, 8, 4, 32, requires_grad=True)
        out = attn(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None


class TestBackendSelection:
    """Test backend selection."""
    
    def test_local_backend(self):
        """Local backend should work without distributed."""
        attn = MultiAxisAttention(embed_dim=32, num_heads=4, attention_axis=1, backend="local")
        assert attn.backend == "local"
    
    def test_ring_backend_requires_package(self):
        """Ring backend requires ring-flash-attn."""
        try:
            attn = MultiAxisAttention(embed_dim=32, num_heads=4, attention_axis=1, backend="ring")
            # If we get here, ring-flash-attn is installed
            assert attn.backend == "ring"
        except ImportError:
            # Expected if ring-flash-attn not installed
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


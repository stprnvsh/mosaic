"""Tests for hierarchical meta-attention."""

import pytest
import torch
import torch.nn as nn

from mosaic.hierarchical import (
    ChunkSummaryEncoder,
    MetaAttention,
    HierarchicalMetaAttention,
    MultiAxisHierarchicalAttention,
)


class TestChunkSummaryEncoder:
    """Test ChunkSummaryEncoder."""
    
    def test_output_shape(self):
        encoder = ChunkSummaryEncoder(embed_dim=64, summary_dim=32, num_heads=4)
        x = torch.randn(2, 100, 64)
        
        summary, attn = encoder(x)
        
        assert summary.shape == (2, 32)
        assert attn.shape == (2, 100)
    
    def test_attention_sums_to_one(self):
        encoder = ChunkSummaryEncoder(embed_dim=64, summary_dim=32, num_heads=4)
        x = torch.randn(2, 100, 64)
        
        _, attn = encoder(x)
        
        # Attention should sum to ~1 (softmax)
        assert torch.allclose(attn.sum(dim=-1), torch.ones(2), atol=1e-5)
    
    def test_gradient_flow(self):
        encoder = ChunkSummaryEncoder(embed_dim=64, summary_dim=32, num_heads=4)
        x = torch.randn(2, 100, 64, requires_grad=True)
        
        summary, _ = encoder(x)
        loss = summary.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestMetaAttention:
    """Test MetaAttention."""
    
    def test_output_shape(self):
        meta = MetaAttention(summary_dim=32, num_heads=4)
        summaries = torch.randn(2, 4, 32)  # 4 chunks
        
        routing, refined, chunk_attn = meta(summaries)
        
        assert routing.shape == (2, 4)
        assert refined.shape == (2, 4, 32)
        assert chunk_attn.shape == (2, 4, 4)
    
    def test_routing_sums_to_one(self):
        meta = MetaAttention(summary_dim=32, num_heads=4)
        summaries = torch.randn(2, 4, 32)
        
        routing, _, _ = meta(summaries)
        
        # Routing weights should sum to 1 (softmax)
        assert torch.allclose(routing.sum(dim=-1), torch.ones(2), atol=1e-5)
    
    def test_gradient_flow(self):
        meta = MetaAttention(summary_dim=32, num_heads=4)
        summaries = torch.randn(2, 4, 32, requires_grad=True)
        
        routing, _, _ = meta(summaries)
        loss = routing.sum()
        loss.backward()
        
        assert summaries.grad is not None


class TestHierarchicalMetaAttention:
    """Test HierarchicalMetaAttention (single GPU)."""
    
    def test_output_shape(self):
        attn = HierarchicalMetaAttention(
            embed_dim=64, num_heads=4, summary_dim=32
        )
        x = torch.randn(2, 100, 64)
        
        out, meta = attn(x)
        
        assert out.shape == x.shape
        assert 'routing_weights' in meta
        assert 'chunk_attention' in meta
        assert 'gate_values' in meta
    
    def test_gradient_flow(self):
        attn = HierarchicalMetaAttention(
            embed_dim=64, num_heads=4, summary_dim=32
        )
        x = torch.randn(2, 100, 64, requires_grad=True)
        
        out, _ = attn(x)
        loss = out.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
    
    def test_checkpointing(self):
        attn = HierarchicalMetaAttention(
            embed_dim=64, num_heads=4, summary_dim=32, use_checkpoint=True
        )
        attn.train()
        x = torch.randn(2, 100, 64, requires_grad=True)
        
        out, _ = attn(x)
        loss = out.sum()
        loss.backward()
        
        assert x.grad is not None
    
    def test_single_gpu_routing(self):
        """On single GPU, routing should be trivial (only self)."""
        attn = HierarchicalMetaAttention(
            embed_dim=64, num_heads=4, summary_dim=32
        )
        x = torch.randn(2, 100, 64)
        
        _, meta = attn(x)
        
        # Single GPU: routing is over 1 chunk
        assert meta['routing_weights'].shape == (2, 1)


class TestMultiAxisHierarchicalAttention:
    """Test MultiAxisHierarchicalAttention."""
    
    def test_axis_1(self):
        attn = MultiAxisHierarchicalAttention(
            embed_dim=64, num_heads=4, attention_axis=1
        )
        x = torch.randn(2, 100, 64)  # (batch, seq, embed)
        
        out, meta = attn(x)
        
        assert out.shape == x.shape
    
    def test_axis_0(self):
        attn = MultiAxisHierarchicalAttention(
            embed_dim=64, num_heads=4, attention_axis=0
        )
        x = torch.randn(100, 2, 64)  # (seq, batch, embed)
        
        out, meta = attn(x)
        
        assert out.shape == x.shape
    
    def test_4d_tensor(self):
        attn = MultiAxisHierarchicalAttention(
            embed_dim=64, num_heads=4, attention_axis=1
        )
        x = torch.randn(2, 50, 10, 64)  # (batch, rows, features, embed)
        
        out, meta = attn(x)
        
        assert out.shape == x.shape
    
    def test_negative_axis(self):
        attn = MultiAxisHierarchicalAttention(
            embed_dim=64, num_heads=4, attention_axis=-2
        )
        x = torch.randn(2, 100, 64)
        
        out, meta = attn(x)
        
        assert out.shape == x.shape


class TestIntegration:
    """Integration tests."""
    
    def test_in_transformer_layer(self):
        """Test as part of a transformer layer."""
        
        class SimpleTransformerLayer(nn.Module):
            def __init__(self, embed_dim, num_heads):
                super().__init__()
                self.attn = MultiAxisHierarchicalAttention(
                    embed_dim=embed_dim, num_heads=num_heads, attention_axis=1
                )
                self.norm1 = nn.LayerNorm(embed_dim)
                self.ffn = nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * 4),
                    nn.GELU(),
                    nn.Linear(embed_dim * 4, embed_dim)
                )
                self.norm2 = nn.LayerNorm(embed_dim)
            
            def forward(self, x):
                attn_out, meta = self.attn(self.norm1(x))
                x = x + attn_out
                x = x + self.ffn(self.norm2(x))
                return x, meta
        
        layer = SimpleTransformerLayer(embed_dim=64, num_heads=4)
        x = torch.randn(2, 100, 64, requires_grad=True)
        
        out, meta = layer(x)
        loss = out.sum()
        loss.backward()
        
        assert out.shape == x.shape
        assert x.grad is not None
    
    def test_stacked_layers(self):
        """Test multiple stacked hierarchical layers."""
        layers = nn.ModuleList([
            MultiAxisHierarchicalAttention(embed_dim=64, num_heads=4, attention_axis=1)
            for _ in range(3)
        ])
        
        x = torch.randn(2, 100, 64, requires_grad=True)
        
        all_meta = []
        for layer in layers:
            x, meta = layer(x)
            all_meta.append(meta)
        
        loss = x.sum()
        loss.backward()
        
        assert len(all_meta) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


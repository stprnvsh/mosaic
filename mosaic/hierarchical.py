"""
Hierarchical Meta-Attention for Mosaic.

Learned routing via chunk summaries + meta-attention, enabling selective
cross-chunk communication instead of O(p) ring rounds.

Architecture:
    1. Local attention within each GPU's chunk
    2. Compress chunks into summaries (content + pattern stats)
    3. all_gather summaries (small)
    4. Meta-attention computes routing weights
    5. Selective cross-attention using soft masking
    6. Gate and combine local + cross outputs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple, Dict

from mosaic.context import MosaicContext


class ChunkSummaryEncoder(nn.Module):
    """Compress a chunk's content + attention pattern into a fixed vector."""
    
    def __init__(self, embed_dim: int, summary_dim: int, num_heads: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.summary_dim = summary_dim
        self.num_heads = num_heads
        
        # Learned query that attends over chunk to get content summary
        self.summary_query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.content_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        # Pattern encoder: 4 stats (entropy, gini, max, top5)
        self.pattern_encoder = nn.Sequential(
            nn.Linear(4, summary_dim),
            nn.ReLU(),
            nn.Linear(summary_dim, summary_dim)
        )
        
        # Combine content + pattern
        self.combine = nn.Linear(embed_dim + summary_dim, summary_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq, embed_dim)
        Returns:
            summary: (batch, summary_dim)
            attn_weights: (batch, seq) - attention over chunk
        """
        B, S, D = x.shape
        
        # Content summary via cross-attention
        query = self.summary_query.expand(B, -1, -1)
        content, attn_weights = self.content_attn(query, x, x, need_weights=True)
        content = content.squeeze(1)  # (B, D)
        attn_weights = attn_weights.squeeze(1)  # (B, S)
        
        # Pattern statistics
        pattern_stats = self._compute_stats(attn_weights)  # (B, 4)
        pattern = self.pattern_encoder(pattern_stats)  # (B, summary_dim)
        
        # Combine
        combined = torch.cat([content, pattern], dim=-1)
        summary = self.combine(combined)
        
        return summary, attn_weights
    
    def _compute_stats(self, attn: torch.Tensor) -> torch.Tensor:
        """Extract attention statistics: entropy, gini, max, top5."""
        # Entropy
        entropy = -(attn * (attn + 1e-8).log()).sum(-1, keepdim=True)
        
        # Gini coefficient (sparsity measure)
        sorted_attn, _ = attn.sort(dim=-1)
        n = attn.size(-1)
        indices = torch.arange(1, n + 1, device=attn.device, dtype=attn.dtype)
        gini = (2 * (indices * sorted_attn).sum(-1) / (n * sorted_attn.sum(-1) + 1e-8) - (n + 1) / n)
        
        # Max attention
        max_attn = attn.max(dim=-1, keepdim=True).values
        
        # Top-5 concentration
        k = min(5, attn.size(-1))
        top5 = attn.topk(k, dim=-1).values.sum(-1, keepdim=True)
        
        return torch.cat([entropy, gini.unsqueeze(-1), max_attn, top5], dim=-1)


class MetaAttention(nn.Module):
    """Attention over chunk summaries to compute routing weights."""
    
    def __init__(self, summary_dim: int, num_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(summary_dim, num_heads, batch_first=True)
        self.routing_head = nn.Sequential(
            nn.Linear(summary_dim, summary_dim),
            nn.ReLU(),
            nn.Linear(summary_dim, 1)
        )
    
    def forward(self, summaries: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            summaries: (batch, num_chunks, summary_dim)
        Returns:
            routing_weights: (batch, num_chunks) - importance of each chunk
            refined: (batch, num_chunks, summary_dim)
            chunk_attn: (batch, num_chunks, num_chunks) - inter-chunk attention
        """
        refined, chunk_attn = self.attn(summaries, summaries, summaries, need_weights=True)
        
        routing_logits = self.routing_head(refined).squeeze(-1)  # (B, num_chunks)
        routing_weights = F.softmax(routing_logits, dim=-1)
        
        return routing_weights, refined, chunk_attn


class HierarchicalMetaAttention(nn.Module):
    """
    Two-level attention with learned routing:
    1. Local attention within each GPU's chunk
    2. Meta-attention over chunk summaries
    3. Selective cross-chunk attention via soft routing
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        summary_dim: int = 64,
        meta_heads: int = 4,
        top_k: int = 2,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.summary_dim = summary_dim
        self.top_k = top_k
        self.use_checkpoint = use_checkpoint
        
        # Local attention
        self.local_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        # Chunk summarization
        self.summarizer = ChunkSummaryEncoder(embed_dim, summary_dim, num_heads)
        
        # Meta-attention over summaries
        self.meta_attn = MetaAttention(summary_dim, meta_heads)
        
        # Cross-chunk attention
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        # Gating
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x_local: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            x_local: (batch, seq_local, embed_dim) - this GPU's chunk
        Returns:
            output: (batch, seq_local, embed_dim)
            meta_info: dict with routing_weights, chunk_attention, gate_values
        """
        ctx = MosaicContext.get() if MosaicContext._instance else None
        B, S, D = x_local.shape
        
        # Stage 1: Local attention (optionally checkpointed)
        if self.use_checkpoint and self.training:
            local_out = checkpoint(self._local_attention, x_local, use_reentrant=False)
        else:
            local_out = self._local_attention(x_local)
        
        # Stage 2: Summarize chunk
        chunk_summary, _ = self.summarizer(local_out)  # (B, summary_dim)
        
        # Stage 3: Gather all summaries
        all_summaries = self._all_gather_summaries(chunk_summary, ctx)  # (B, world_size, summary_dim)
        
        # Stage 4: Meta-attention
        routing_weights, refined, chunk_attn = self.meta_attn(all_summaries)
        
        # Stage 5: Selective cross-attention (soft routing)
        cross_out = self._selective_cross_attention(local_out, routing_weights, ctx)
        
        # Stage 6: Gate and combine
        gate_input = torch.cat([local_out, cross_out], dim=-1)
        gate = self.gate(gate_input)
        combined = gate * cross_out + (1 - gate) * local_out
        
        output = self.out_proj(combined)
        
        meta_info = {
            'routing_weights': routing_weights,
            'chunk_attention': chunk_attn,
            'gate_values': gate.mean(dim=(1, 2))
        }
        
        return output, meta_info
    
    def _local_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Local attention on chunk."""
        out, _ = self.local_attn(x, x, x)
        return out
    
    def _all_gather_summaries(
        self, local_summary: torch.Tensor, ctx: Optional[MosaicContext]
    ) -> torch.Tensor:
        """Gather chunk summaries from all GPUs."""
        if ctx is None or ctx.world_size == 1:
            return local_summary.unsqueeze(1)
        
        B, D = local_summary.shape
        world_size = ctx.world_size
        
        # Pre-allocate output
        all_summaries = torch.zeros(B, world_size, D, device=local_summary.device, dtype=local_summary.dtype)
        
        # All-gather
        gathered = [torch.zeros_like(local_summary) for _ in range(world_size)]
        dist.all_gather(gathered, local_summary.contiguous(), group=ctx.sp_group)
        
        for i, s in enumerate(gathered):
            all_summaries[:, i, :] = s
        
        return all_summaries
    
    def _selective_cross_attention(
        self, 
        local_out: torch.Tensor,
        routing_weights: torch.Tensor,
        ctx: Optional[MosaicContext]
    ) -> torch.Tensor:
        """Cross-attention using soft routing weights."""
        if ctx is None or ctx.world_size == 1:
            return local_out
        
        B, S, D = local_out.shape
        world_size = ctx.world_size
        my_rank = ctx.rank
        
        # Gather all chunks (necessary for soft routing)
        all_chunks = self._all_gather_chunks(local_out, ctx)  # (B, world_size, S, D)
        
        # Zero out self-contribution in routing (don't attend to own chunk)
        routing_masked = routing_weights.clone()
        routing_masked[:, my_rank] = 0
        routing_masked = routing_masked / (routing_masked.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Soft-weighted KV: weighted sum of all remote chunks
        # routing_masked: (B, world_size)
        # all_chunks: (B, world_size, S, D)
        weighted_kv = torch.einsum('bw,bwsd->bsd', routing_masked, all_chunks)  # (B, S, D)
        
        # Cross-attention: local queries, weighted remote KV
        cross_out, _ = self.cross_attn(local_out, weighted_kv, weighted_kv)
        
        return cross_out
    
    def _all_gather_chunks(
        self, local_out: torch.Tensor, ctx: MosaicContext
    ) -> torch.Tensor:
        """Gather full chunks from all GPUs."""
        B, S, D = local_out.shape
        world_size = ctx.world_size
        
        # Pre-allocate
        all_chunks = torch.zeros(B, world_size, S, D, device=local_out.device, dtype=local_out.dtype)
        
        # All-gather using async ops to avoid deadlocks
        local_flat = local_out.contiguous()
        gathered = [torch.zeros_like(local_flat) for _ in range(world_size)]
        dist.all_gather(gathered, local_flat, group=ctx.sp_group)
        
        for i, chunk in enumerate(gathered):
            all_chunks[:, i, :, :] = chunk
        
        return all_chunks


class MultiAxisHierarchicalAttention(nn.Module):
    """
    Convenience wrapper for HierarchicalMetaAttention with axis permutation.
    Same interface as MultiAxisAttention but uses hierarchical meta-attention.
    
    Good for:
    - Genomics: Long sequences with sparse long-range dependencies
    - Time series: Seasonal patterns that repeat far apart
    - Documents: Topics that span distant paragraphs
    
    Example:
        attn = MultiAxisHierarchicalAttention(
            embed_dim=256, num_heads=8, attention_axis=1,
            summary_dim=64, top_k=2
        )
        out, meta = attn(x)  # meta['routing_weights'] shows which chunks interact
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attention_axis: int,
        summary_dim: int = 64,
        meta_heads: int = 4,
        top_k: int = 2,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.attention_axis = attention_axis
        self.embed_dim = embed_dim
        
        self.hierarchical = HierarchicalMetaAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            summary_dim=summary_dim,
            meta_heads=meta_heads,
            top_k=top_k,
            use_checkpoint=use_checkpoint,
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            x: N-dimensional tensor with embed_dim as last dimension
        Returns:
            output: Same shape as input
            meta_info: Routing weights and diagnostics
        """
        # Move attention axis to position -2
        x, inv_perm = self._permute_to_seq(x)
        
        # Flatten batch dims (contiguous needed after permute)
        batch_shape = x.shape[:-2]
        seq_len = x.shape[-2]
        x = x.contiguous().view(-1, seq_len, self.embed_dim)
        
        # Hierarchical attention
        out, meta_info = self.hierarchical(x)
        
        # Restore shape
        if batch_shape:
            out = out.view(*batch_shape, seq_len, self.embed_dim)
        if inv_perm:
            out = out.permute(inv_perm)
        
        return out, meta_info
    
    def _permute_to_seq(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[list]]:
        """Move attention_axis to position -2."""
        ndim = x.ndim
        ax = self.attention_axis if self.attention_axis >= 0 else ndim + self.attention_axis
        
        if ax == ndim - 2:
            return x, None
        
        perm = [i for i in range(ndim) if i != ax and i != ndim - 1]
        perm.append(ax)
        perm.append(ndim - 1)
        
        inv_perm = [0] * ndim
        for i, p in enumerate(perm):
            inv_perm[p] = i
        
        return x.permute(perm), inv_perm


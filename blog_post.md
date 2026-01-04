# Mosaic: Sharding Attention Across GPUs When Your Sequence Doesn't Fit

*How we built a lightweight library to distribute 150,000-token attention across multiple GPUs*

---

## The Problem: Attention Doesn't Scale

You've probably heard that transformers have a "quadratic attention bottleneck." Here's what that actually means in practice.

Attention computes:

```
Attention(Q, K, V) = softmax(QKᵀ / √d) × V
```

The killer is **QKᵀ** — a matrix of shape `(sequence_length × sequence_length)`. For a 150,000-token sequence:

```
Memory = 150,000² × 4 bytes = 90 billion bytes = 84 GB
```

That's just for the attention weights. One layer. One head. An A100 has 80GB total.

**You can't fit it.**

## Existing Solutions (and Their Limits)

**FlashAttention** reduces memory from O(n²) to O(n) by computing attention in tiles without materializing the full matrix. But it still requires the entire sequence on one GPU.

**Ring Attention** (from ring-flash-attn) shards the sequence across GPUs. Each GPU holds a chunk of Q and passes K, V around in a ring. Beautiful for 1D sequences.

**The gap:** What about models with multiple attention patterns? 

Consider a tabular transformer with shape `(batch, rows, features, embed)`:
- Attention over **features** (axis 2): 5 tokens — fits easily
- Attention over **rows** (axis 1): 150,000 tokens — needs sharding

No library handled this cleanly. You'd write custom code for each axis, manage different process groups, handle the tensor reshaping yourself.

## Mosaic: Multi-Axis Attention Sharding

Mosaic is a thin coordination layer that routes different attention axes to appropriate backends:

```python
import mosaic

# Small axis: run locally
feature_attn = mosaic.MultiAxisAttention(
    embed_dim=96, num_heads=4,
    attention_axis=2,    # features dimension
    backend="local"      # no communication needed
)

# Large axis: shard across GPUs
row_attn = mosaic.MultiAxisAttention(
    embed_dim=96, num_heads=4,
    attention_axis=1,    # rows dimension  
    backend="ring"       # ring attention across GPUs
)
```

That's it. Mosaic handles:
1. Permuting the attention axis to the sequence position
2. Reshaping for QKV projection
3. Dispatching to the right backend
4. Restoring the original tensor shape

## How Ring Attention Works

The key insight: you don't need all of K and V at once. You can compute partial attention scores, accumulate them, and normalize at the end.

```
4 GPUs, sequence split into 4 chunks:

Initial state:
  GPU 0: Q₀, K₀, V₀
  GPU 1: Q₁, K₁, V₁
  GPU 2: Q₂, K₂, V₂
  GPU 3: Q₃, K₃, V₃

Step 1: Each GPU computes attention with its local K, V
  GPU 0: score₀₀ = Q₀ @ K₀ᵀ
  ...

Step 2: Pass K, V to the next GPU in the ring
  GPU 0 receives K₃, V₃ from GPU 3
  GPU 0 sends K₀, V₀ to GPU 1
  
Step 3: Compute attention with received K, V
  GPU 0: score₀₃ = Q₀ @ K₃ᵀ
  Accumulate with score₀₀
  
Repeat for all chunks...

Final: Each GPU has complete attention output for its Q chunk
```

**Memory per GPU:** O(n²/p) where p = number of GPUs

With 8 GPUs, you've reduced memory 8×. A 150k sequence now needs ~10GB per GPU instead of 84GB.

## Beyond 1D: Mesh2D Attention

For very long sequences, even ring attention isn't enough. Mesh2D shards both Q and K:

```
4 GPUs in 2×2 mesh:

         K₀      K₁
       ┌──────┬──────┐
    Q₀ │GPU 0 │GPU 1 │
       ├──────┼──────┤
    Q₁ │GPU 2 │GPU 3 │
       └──────┴──────┘

Each GPU computes one tile of QKᵀ
```

**Memory per GPU:** O(n²/p²)

With 64 GPUs in an 8×8 mesh, memory drops 64× per GPU.

```python
attn = mosaic.MultiAxisAttention(
    embed_dim=128, num_heads=8,
    attention_axis=1,
    backend="mesh2d",
    mesh_shape=(8, 8)
)
```

## Composed Strategies

Real clusters have topology. GPUs within a node communicate via fast NVLink (900 GB/s). GPUs across nodes use slower InfiniBand (200 GB/s).

Mosaic's `ComposedAttention` exploits this:

```python
# 4 nodes × 8 GPUs = 32 total
composed = mosaic.ComposedAttention(
    mesh_shape=(4, 8),       # (nodes, gpus_per_node)
    head_parallel=True,      # Split heads across nodes (slow link)
    seq_parallel="ring"      # Ring within nodes (fast link)
)
```

Or use `HierarchicalAttention` for explicit control:

```python
hier = mosaic.HierarchicalAttention(
    intra_node_size=8,
    intra_node_strategy="local",  # Compute locally within node
    inter_node_strategy="ring"    # Ring between node leaders
)
```

## The Implementation

Mosaic is ~800 lines of Python. Here's the core pattern:

```python
class MultiAxisAttention(nn.Module):
    def forward(self, x):
        # 1. Move attention axis to seq position
        x, inv_perm = self._permute_to_seq(x)
        
        # 2. Flatten batch dims, project QKV
        x = x.view(-1, seq_len, embed_dim)
        qkv = self.qkv_proj(x).view(batch, seq, 3, heads, head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        
        # 3. Dispatch to backend
        out = self._attn_fn(q, k, v)  # local, ring, or mesh2d
        
        # 4. Project output, restore shape
        out = self.out_proj(out.transpose(1, 2).reshape(...))
        return out.permute(inv_perm)
```

The backends wrap existing libraries:
- `local`: `F.scaled_dot_product_attention` (FlashAttention)
- `ring`: `ring_flash_attn_func` from ring-flash-attn
- `mesh2d`: Custom all-gather + SDPA

All use FlashAttention kernels for the actual attention computation.

## Usage

```bash
pip install git+https://github.com/stprnvsh/mosaic.git

# With ring attention support
pip install flash-attn ring-flash-attn
```

Single node:
```bash
torchrun --nproc_per_node=4 train.py
```

Multi-node:
```bash
# Node 0
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 \
    --master_addr=192.168.1.100 --master_port=29500 train.py

# Node 1  
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=1 \
    --master_addr=192.168.1.100 --master_port=29500 train.py
```

Training script:
```python
import mosaic
import torch.distributed as dist

dist.init_process_group("nccl")
ctx = mosaic.init(sp_size=dist.get_world_size())

model = MyModel().to(ctx.device)

# Data is pre-sharded: each GPU has seq_total / world_size tokens
x_local = load_my_shard()
out = model(x_local)  # Communication handled by Mosaic
```

## When to Use What

| Sequence | GPUs | Backend | Memory/GPU |
|----------|------|---------|------------|
| < 10k | 1 | `local` | O(n²) |
| 10k–100k | 2–8 | `ring` | O(n²/p) |
| 100k–1M | 8–64 | `mesh2d` | O(n²/p²) |
| > 1M | 64+ | `composed` | O(n²/(p²·h)) |

## Performance

We optimized for zero overhead:

1. **FlashAttention everywhere** — All backends use `F.scaled_dot_product_attention` for fused GEMM + softmax
2. **Pre-selected dispatch** — Backend function bound at init, no branching in forward
3. **View not copy** — `x.view()` instead of `x.reshape()` when contiguous
4. **Pre-allocated collectives** — `all_gather` into pre-sized tensors, no `torch.cat`
5. **Module-level imports** — No import overhead per forward pass

## What Mosaic Is Not

Mosaic doesn't:
- Auto-parallelize your model (use nnScaler for that)
- Handle data parallelism (use PyTorch DDP/FSDP)
- Manage model sharding (use FSDP or Megatron)

It does one thing: **route multi-axis attention to the right sharding backend**.

## The Origin Story

This came from profiling nanoTabPFN, a transformer for tabular data. The model has attention over both rows (150k) and features (5). Standard ring attention doesn't understand "rows" vs "features" — it just sees a sequence dimension.

We needed:
- Local attention for small axes
- Ring attention for large axes  
- Clean axis routing without rewriting the model

Mosaic is the result.

---

**Code:** [github.com/stprnvsh/mosaic](https://github.com/stprnvsh/mosaic)

**Dependencies:** PyTorch 2.0+, NCCL, optionally flash-attn + ring-flash-attn


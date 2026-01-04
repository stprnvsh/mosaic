# Mosaic

**Multi-Axis Attention Sharding for PyTorch**

Distribute attention computation across GPUs when your sequence is too long to fit on one device.

## The Problem

Standard attention has **O(n²) memory** complexity. A 150,000-token sequence needs ~84GB just for attention weights:

```
Memory = n² × 4 bytes = 150,000² × 4 = 84 GB
```

**Mosaic** splits the sequence across GPUs and coordinates communication so each GPU only holds a fraction.

## How It Works

### Attention Refresher

For queries Q, keys K, values V with sequence length n:

```
Attention(Q, K, V) = softmax(QKᵀ / √d) × V
```

The bottleneck is **QKᵀ** — an (n × n) matrix.

### Sharding Strategies

Mosaic supports three backends:

#### 1. Local (No Sharding)
Each GPU has the full sequence. Use for small dimensions.

```
GPU 0: [Q₀, K₀, V₀] → Attention → [Out₀]
```

#### 2. Ring Attention (1D Sharding)
Split sequence across GPUs. Each GPU holds Q_local but needs all K, V.

**Solution:** Pass K, V chunks around in a ring while accumulating partial attention.

```
Step 0:  GPU₀ has (Q₀, K₀, V₀)    GPU₁ has (Q₁, K₁, V₁)
         ↓ compute Q₀K₀ᵀ          ↓ compute Q₁K₁ᵀ
         
Step 1:  GPU₀ receives K₁, V₁    GPU₁ receives K₀, V₀
         ↓ compute Q₀K₁ᵀ          ↓ compute Q₁K₀ᵀ
         
Final:   Each GPU has full attention output for its Q chunk
```

**Memory per GPU:** O(n²/p) where p = number of GPUs

**Communication:** Each GPU sends/receives (n/p × d) per step, p-1 steps total

#### 3. Mesh2D Attention (2D Sharding)
For very large sequences, shard both Q and K:

```
         K₀      K₁
       ┌──────┬──────┐
    Q₀ │GPU 0 │GPU 1 │  ← Each GPU computes one tile of QKᵀ
       ├──────┼──────┤
    Q₁ │GPU 2 │GPU 3 │
       └──────┴──────┘
```

**Memory per GPU:** O(n²/p²)

**Trade-off:** More communication (all-gather K, V along columns)

## Installation

```bash
git clone https://github.com/stprnvsh/mosaic.git
cd mosaic
pip install -e .

# With ring attention (requires FlashAttention)
pip install -e ".[ring]"
```

## Quick Start

```python
import mosaic
import torch.nn as nn

# Initialize: 4 GPUs for sequence parallelism
ctx = mosaic.init(sp_size=4)

# Attention over axis 1, sharded across GPUs
attn = mosaic.MultiAxisAttention(
    embed_dim=128,
    num_heads=8,
    attention_axis=1,    # Which axis to attend over
    backend="ring"       # Use ring attention
)

# Input: (batch, sequence, features) where sequence is sharded
# Each GPU has (batch, seq_local, features) where seq_local = seq_total / 4
x_local = torch.randn(2, 37500, 128).cuda()  # 150k / 4 = 37.5k per GPU

out = attn(x_local)  # Ring communication happens automatically
```

Launch:
```bash
torchrun --nproc_per_node=4 train.py
```

## Multi-Axis Models

Models like **nanoTabPFN** have tensors with shape `(batch, rows, features, embed)` and need attention over **multiple axes**:

| Axis | Dimension | Strategy |
|------|-----------|----------|
| Features (axis 2) | ~5 | Local (small, fits on GPU) |
| Rows (axis 1) | ~150,000 | Ring (too large, must shard) |

```python
class TabularTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        # Small axis: local attention
        self.feature_attn = mosaic.MultiAxisAttention(
            embed_dim=96, num_heads=4,
            attention_axis=2,  # features
            backend="local"
        )
        # Large axis: ring attention
        self.row_attn = mosaic.MultiAxisAttention(
            embed_dim=96, num_heads=4,
            attention_axis=1,  # rows
            backend="ring"
        )
    
    def forward(self, x):
        # x: (batch, rows_local, features, embed)
        x = self.feature_attn(x) + x  # No communication
        x = self.row_attn(x) + x      # Ring across GPUs
        return x
```

## API Reference

### Core

#### `mosaic.init(sp_size=1) → MosaicContext`
Initialize distributed context.
- `sp_size`: Number of GPUs for sequence parallelism

#### `mosaic.MultiAxisAttention(embed_dim, num_heads, attention_axis, backend, mesh_shape=None)`
Attention over any single tensor axis.

| Parameter | Description |
|-----------|-------------|
| `embed_dim` | Hidden dimension (must be divisible by num_heads) |
| `num_heads` | Number of attention heads |
| `attention_axis` | Which axis to attend over (supports negative indexing) |
| `backend` | `"local"`, `"ring"`, or `"mesh2d"` |
| `mesh_shape` | Required for mesh2d: `(rows, cols)` GPU grid |

### Advanced

#### `mosaic.MultiAxisAttention2D(embed_dim, num_heads, axis1, axis2, mesh_shape)`
Attend over **two axes simultaneously** (flattens them internally).

```python
# Attention over both rows and columns
attn = mosaic.MultiAxisAttention2D(
    embed_dim=128, num_heads=8,
    axis1=1, axis2=2,      # rows × columns
    mesh_shape=(2, 2)      # 4 GPUs in 2×2 grid
)
# Input: (batch, rows, cols, embed)
# Internally: flatten to (batch, rows*cols, embed), run mesh2d attention
```

#### `mosaic.ComposedAttention(mesh_shape, head_parallel, seq_parallel)`
Combine head parallelism with sequence parallelism.

```python
# 8 GPUs: 2-way head parallel × 4-way sequence parallel
composed = mosaic.ComposedAttention(
    mesh_shape=(2, 4),
    head_parallel=True,    # Split heads across dim 0 (2 ways)
    seq_parallel="ring"    # Ring attention across dim 1 (4 ways)
)
```

**Memory:** Heads sharded 2×, sequence sharded 4× → **8× memory reduction**

#### `mosaic.HierarchicalAttention(intra_node_size, inter_node_strategy, intra_node_strategy)`
Two-level parallelism for multi-node clusters.

```python
# 4 nodes × 8 GPUs = 32 GPUs
hier = mosaic.HierarchicalAttention(
    intra_node_size=8,           # GPUs per node
    intra_node_strategy="local", # Fast NVLink within node
    inter_node_strategy="ring"   # Slower network between nodes
)
```

## Performance

All backends use **FlashAttention** (`F.scaled_dot_product_attention`) for the local computation:
- Fused GEMM + softmax + GEMM
- O(n) memory instead of O(n²) for attention weights
- 2-4× faster than naive implementation

Communication uses NCCL collectives:
- Ring: `send`/`recv` in ring topology
- Mesh2D: `all_gather` along grid dimensions

## When to Use What

| Sequence Length | GPUs | Backend | Memory per GPU |
|-----------------|------|---------|----------------|
| < 10k | 1 | `local` | O(n²) |
| 10k - 100k | 2-8 | `ring` | O(n²/p) |
| 100k - 1M | 8-64 | `ring` or `mesh2d` | O(n²/p) or O(n²/p²) |
| > 1M | 64+ | `mesh2d` + `head_parallel` | O(n²/(p²·h)) |

## Architecture

```
┌─────────────────────────────────────────┐
│              User Model                 │
│  (defines attention_axis per layer)     │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│               Mosaic                    │
│  • Axis routing (permute to seq dim)    │
│  • Backend selection                    │
│  • Tensor reshape for QKV projection    │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│             Backends                    │
│  • local: F.scaled_dot_product_attention│
│  • ring: ring_flash_attn_func           │
│  • mesh2d: all_gather + SDPA            │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│          PyTorch / NCCL                 │
└─────────────────────────────────────────┘
```

## License

MIT

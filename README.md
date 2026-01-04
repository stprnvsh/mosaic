# Mosaic

Multi-Axis Attention Sharding for PyTorch.

A lightweight coordination layer for models with multiple attention axes, built on top of existing attention sharding libraries (ring-flash-attn, MagiAttention).

## Problem

Models like nanoTabPFN have 4D tensors `(batch, rows, features, embed)` with attention over different axes:
- **Feature attention** (axis 2): Small dimension (~5), runs locally
- **Row attention** (axis 1): Large dimension (~150k), needs sharding

Existing libraries (ring-flash-attn, MagiAttention) solve single-axis sharding. Mosaic coordinates them for multi-axis models.

## Installation

```bash
pip install -e .

# With ring attention support
pip install -e ".[ring]"
```

## Usage

```python
import mosaic
import torch.nn as nn

# Initialize (4-way sequence parallel)
ctx = mosaic.init(sp_size=4)

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Feature attention: local (small axis)
        self.feature_attn = mosaic.MultiAxisAttention(
            embed_dim=96,
            num_heads=4,
            attention_axis=2,
            backend="local"
        )
        # Row attention: ring (large axis, sharded across GPUs)
        self.row_attn = mosaic.MultiAxisAttention(
            embed_dim=96,
            num_heads=4,
            attention_axis=1,
            backend="ring"
        )
    
    def forward(self, x):
        # x: (batch, rows_local, features, embed)
        x = self.feature_attn(x) + x
        x = self.row_attn(x) + x
        return x

model = MyModel().to(ctx.device)
```

Launch with:
```bash
torchrun --nproc_per_node=4 train.py
```

## Architecture

```
User Model
    |
Mosaic (axis routing, tensor reshaping)
    |
Backends (ring-flash-attn, local FlashAttention)
    |
PyTorch / NCCL
```

## API

### `mosaic.init(sp_size=1)`
Initialize Mosaic context with sequence parallel size.

### `mosaic.MultiAxisAttention(embed_dim, num_heads, attention_axis, backend)`
Attention layer that operates over any tensor axis.

- `attention_axis`: Which axis to compute attention over
- `backend`: `"local"` (no communication) or `"ring"` (ring attention across GPUs)

## License

MIT


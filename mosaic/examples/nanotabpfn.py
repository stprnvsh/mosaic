"""
nanoTabPFN with Mosaic multi-axis attention sharding.

This example shows how to parallelize nanoTabPFN across multiple GPUs
by sharding the row (datapoint) dimension using ring attention.

Usage:
    torchrun --nproc_per_node=4 mosaic/examples/nanotabpfn.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

import mosaic


class FeatureEncoder(nn.Module):
    """Encode input features."""
    def __init__(self, embedding_size: int):
        super().__init__()
        self.linear = nn.Linear(1, embedding_size)

    def forward(self, x: torch.Tensor, train_test_split_index: int) -> torch.Tensor:
        x = x.unsqueeze(-1)
        mean = torch.mean(x[:, :train_test_split_index], dim=1, keepdims=True)
        std = torch.std(x[:, :train_test_split_index], dim=1, keepdims=True) + 1e-20
        x = (x - mean) / std
        x = torch.clip(x, min=-100, max=100)
        return self.linear(x)


class TargetEncoder(nn.Module):
    """Encode target labels."""
    def __init__(self, embedding_size: int):
        super().__init__()
        self.linear = nn.Linear(1, embedding_size)

    def forward(self, y_train: torch.Tensor, num_rows: int) -> torch.Tensor:
        mean = torch.mean(y_train, dim=1, keepdim=True)
        padding = mean.repeat(1, num_rows - y_train.shape[1], 1)
        y = torch.cat([y_train, padding], dim=1)
        y = y.unsqueeze(-1)
        return self.linear(y)


class TransformerLayerParallel(nn.Module):
    """
    Transformer layer with multi-axis attention using Mosaic.
    
    - Feature attention (axis 2): Local, no communication
    - Row attention (axis 1): Ring attention across GPUs
    """
    def __init__(self, embedding_size: int, num_heads: int, mlp_hidden_size: int):
        super().__init__()
        
        # Feature attention: local (features axis is small ~5-10)
        self.feature_attn = mosaic.MultiAxisAttention(
            embed_dim=embedding_size,
            num_heads=num_heads,
            attention_axis=2,  # features axis
            backend="local"
        )
        
        # Row attention: ring (rows axis can be huge ~150k)
        self.row_attn = mosaic.MultiAxisAttention(
            embed_dim=embedding_size,
            num_heads=num_heads,
            attention_axis=1,  # rows axis
            backend="ring"
        )
        
        # MLP
        self.linear1 = nn.Linear(embedding_size, mlp_hidden_size)
        self.linear2 = nn.Linear(mlp_hidden_size, embedding_size)
        
        # Norms
        self.norm1 = nn.LayerNorm(embedding_size)
        self.norm2 = nn.LayerNorm(embedding_size)
        self.norm3 = nn.LayerNorm(embedding_size)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src: (batch, rows_local, features, embed)
                 rows_local = total_rows / num_gpus
        """
        # Feature attention (local)
        src = self.feature_attn(src) + src
        src = self.norm1(src)
        
        # Row attention (ring across GPUs)
        src = self.row_attn(src) + src
        src = self.norm2(src)
        
        # MLP
        src = self.linear2(F.gelu(self.linear1(src))) + src
        src = self.norm3(src)
        
        return src


class Decoder(nn.Module):
    """Decode embeddings to class logits."""
    def __init__(self, embedding_size: int, mlp_hidden_size: int, num_outputs: int):
        super().__init__()
        self.linear1 = nn.Linear(embedding_size, mlp_hidden_size)
        self.linear2 = nn.Linear(mlp_hidden_size, num_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(F.gelu(self.linear1(x)))


class NanoTabPFNParallel(nn.Module):
    """
    nanoTabPFN parallelized with Mosaic.
    
    Rows are sharded across GPUs. Feature attention is local,
    row attention uses ring communication.
    """
    def __init__(
        self,
        embedding_size: int = 96,
        num_attention_heads: int = 4,
        mlp_hidden_size: int = 192,
        num_layers: int = 3,
        num_outputs: int = 2,
    ):
        super().__init__()
        self.feature_encoder = FeatureEncoder(embedding_size)
        self.target_encoder = TargetEncoder(embedding_size)
        
        self.layers = nn.ModuleList([
            TransformerLayerParallel(embedding_size, num_attention_heads, mlp_hidden_size)
            for _ in range(num_layers)
        ])
        
        self.decoder = Decoder(embedding_size, mlp_hidden_size, num_outputs)

    def forward(
        self,
        src: Tuple[torch.Tensor, torch.Tensor],
        train_test_split_index: int
    ) -> torch.Tensor:
        """
        Args:
            src: (x, y) where x is features and y is labels
            train_test_split_index: Index separating train from test rows
            
        Returns:
            Logits for test rows
        """
        x_src, y_src = src
        
        if len(y_src.shape) < len(x_src.shape):
            y_src = y_src.unsqueeze(-1)
        
        x_src = self.feature_encoder(x_src, train_test_split_index)
        num_rows = x_src.shape[1]
        y_src = self.target_encoder(y_src, num_rows)
        
        # Concatenate along feature axis
        src = torch.cat([x_src, y_src], dim=2)
        
        # Apply transformer layers
        for layer in self.layers:
            src = layer(src)
        
        # Decode test rows only
        output = src[:, train_test_split_index:, -1, :]
        output = self.decoder(output)
        
        return output


def shard_data(x: torch.Tensor, y: torch.Tensor, ctx: mosaic.MosaicContext):
    """
    Shard data along the rows axis for sequence parallelism.
    
    Args:
        x: Full data tensor (batch, rows, features)
        y: Full labels tensor (batch, rows)
        ctx: Mosaic context
        
    Returns:
        Local shards for this GPU
    """
    total_rows = x.shape[1]
    rows_per_gpu = total_rows // ctx.sp_size
    
    start = ctx.sp_rank * rows_per_gpu
    end = start + rows_per_gpu
    
    x_local = x[:, start:end, :]
    y_local = y[:, start:end]
    
    return x_local, y_local


def main():
    # Initialize Mosaic
    ctx = mosaic.init(sp_size=torch.cuda.device_count())
    print(f"Rank {ctx.rank}: Initialized with {ctx.sp_size}-way sequence parallelism")
    
    # Create model
    model = NanoTabPFNParallel(
        embedding_size=96,
        num_attention_heads=4,
        mlp_hidden_size=192,
        num_layers=3,
        num_outputs=2,
    ).to(ctx.device)
    
    # Create dummy data (normally would load from file)
    batch_size = 4
    total_rows = 1000  # In real use: 150000
    features = 5
    
    x = torch.randn(batch_size, total_rows, features, device=ctx.device)
    y = torch.randn(batch_size, total_rows, device=ctx.device)
    
    # Shard data across GPUs
    x_local, y_local = shard_data(x, y, ctx)
    print(f"Rank {ctx.rank}: Local data shape: {x_local.shape}")
    
    # Forward pass
    train_test_split = x_local.shape[1] // 2  # Half train, half test
    output = model((x_local, y_local[:, :train_test_split]), train_test_split)
    print(f"Rank {ctx.rank}: Output shape: {output.shape}")
    
    # Backward pass
    loss = output.mean()
    loss.backward()
    print(f"Rank {ctx.rank}: Backward pass complete")


if __name__ == "__main__":
    main()


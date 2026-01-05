"""
Mosaic training script for Google Cloud Vertex AI.

Usage:
    # Local test
    python train_gcp.py --local
    
    # On Vertex AI (set by environment)
    python train_gcp.py
"""

import os
import argparse
import torch
import torch.distributed as dist
import mosaic
from mosaic.attention import MultiAxisAttention


def setup_distributed():
    """Initialize distributed training from GCP environment variables."""
    # Vertex AI sets these environment variables
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = os.environ.get("MASTER_PORT", "29500")
    
    if world_size > 1:
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = master_port
        dist.init_process_group(
            backend="nccl",
            world_size=world_size,
            rank=rank
        )
    
    torch.cuda.set_device(local_rank)
    
    return world_size, rank, local_rank


def create_model(embed_dim=128, num_heads=8, seq_parallel_size=1):
    """Create a simple model with Mosaic attention."""
    
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Linear(64, embed_dim)
            
            # Use ring attention for large sequences
            backend = "ring" if seq_parallel_size > 1 else "local"
            self.attn = MultiAxisAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                attention_axis=1,
                backend=backend,
                use_checkpoint=True
            )
            self.out = torch.nn.Linear(embed_dim, 10)
        
        def forward(self, x):
            x = self.embed(x)
            x = self.attn(x)
            return self.out(x.mean(dim=1))
    
    return SimpleModel()


def train(args):
    """Main training loop."""
    world_size, rank, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    
    is_main = rank == 0
    if is_main:
        print(f"World size: {world_size}")
        print(f"Device: {device}")
        print(f"GPU: {torch.cuda.get_device_name()}")
    
    # Initialize Mosaic
    ctx = mosaic.init(sp_size=world_size)
    
    # Create model
    model = create_model(
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        seq_parallel_size=world_size
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # Training loop
    model.train()
    for step in range(args.steps):
        # Generate synthetic data
        # Each GPU has seq_len / world_size tokens
        local_seq_len = args.seq_len // world_size
        x = torch.randn(args.batch_size, local_seq_len, 64, device=device)
        y = torch.randint(0, 10, (args.batch_size,), device=device)
        
        # Forward
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if is_main and step % 10 == 0:
            print(f"Step {step}: loss = {loss.item():.4f}")
    
    # Save model (only on main process)
    if is_main and args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        torch.save(model.state_dict(), f"{args.output_dir}/model.pt")
        print(f"Model saved to {args.output_dir}/model.pt")
    
    if world_size > 1:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true", help="Run locally")
    parser.add_argument("--seq_len", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output_dir", type=str, default=None)
    
    args = parser.parse_args()
    
    if args.local:
        os.environ["WORLD_SIZE"] = "1"
        os.environ["RANK"] = "0"
        os.environ["LOCAL_RANK"] = "0"
    
    train(args)


if __name__ == "__main__":
    main()


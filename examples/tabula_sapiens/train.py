"""
Train hierarchical attention model on Tabula Sapiens.

For 8 T4 GPUs on Vertex AI:
    torchrun --nproc_per_node=8 train.py

The model learns to predict cell type from gene expression.
With 500k cells × 5k genes, hierarchical attention learns which gene
regions interact (e.g., transcription factors and their targets).
"""

import os
import json
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import numpy as np

# Will be installed in container
try:
    import mosaic
    from mosaic.hierarchical import MultiAxisHierarchicalAttention
    MOSAIC_AVAILABLE = True
except ImportError:
    MOSAIC_AVAILABLE = False
    print("Warning: mosaic not installed, using local attention only")


# === Config ===
BUCKET = os.environ.get("DATA_BUCKET", "gs://mosaic-gpu-tabula-sapiens")
EMBED_DIM = 128
NUM_HEADS = 8
NUM_LAYERS = 4
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-4


class GCSDataset(Dataset):
    """Load data from GCS."""
    
    def __init__(self, bucket_path):
        import subprocess
        
        # Download data from GCS
        local_dir = "/tmp/data"
        os.makedirs(local_dir, exist_ok=True)
        
        if not os.path.exists(f"{local_dir}/X.npy"):
            print(f"Downloading data from {bucket_path}...")
            subprocess.run(
                f"gcloud storage cp {bucket_path}/* {local_dir}/",
                shell=True, check=True
            )
        
        self.X = np.load(f"{local_dir}/X.npy")
        self.y = np.load(f"{local_dir}/y.npy")
        
        with open(f"{local_dir}/metadata.json") as f:
            self.metadata = json.load(f)
        
        print(f"Loaded: {self.X.shape[0]} samples, {self.X.shape[1]} features")
        print(f"Cell types: {self.metadata['n_cell_types']}")
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X[idx]).float(),
            torch.tensor(self.y[idx]).long()
        )


class HierarchicalCellTypeClassifier(nn.Module):
    """
    Cell type classifier using hierarchical meta-attention.
    
    Architecture:
    - Input: (batch, n_genes) gene expression
    - Embed: project to (batch, n_genes, embed_dim)
    - Hierarchical attention: learns which gene regions interact
    - Pool + classify
    
    The hierarchical attention discovers:
    - Which gene modules co-express
    - Long-range regulatory relationships
    - Cell-type specific gene programs
    """
    
    def __init__(self, n_genes, n_classes, embed_dim=128, num_heads=8, num_layers=4):
        super().__init__()
        
        self.n_genes = n_genes
        self.embed_dim = embed_dim
        
        # Gene embedding
        self.gene_embed = nn.Linear(1, embed_dim)
        
        # Positional encoding for genes
        self.pos_embed = nn.Parameter(torch.randn(1, n_genes, embed_dim) * 0.02)
        
        # Hierarchical attention layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            if MOSAIC_AVAILABLE:
                # Use hierarchical meta-attention
                attn = MultiAxisHierarchicalAttention(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    attention_axis=1,  # attention over genes
                    summary_dim=64,
                    top_k=2,
                    use_checkpoint=True,
                )
            else:
                # Fallback to standard attention
                attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
            
            self.layers.append(nn.ModuleDict({
                'attn': attn,
                'norm1': nn.LayerNorm(embed_dim),
                'ffn': nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * 4),
                    nn.GELU(),
                    nn.Linear(embed_dim * 4, embed_dim),
                ),
                'norm2': nn.LayerNorm(embed_dim),
            }))
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, n_classes),
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, n_genes) gene expression values
        Returns:
            logits: (batch, n_classes)
            meta_info: list of routing info from each layer
        """
        B, G = x.shape
        
        # Embed genes: (batch, n_genes, 1) -> (batch, n_genes, embed_dim)
        x = x.unsqueeze(-1)
        x = self.gene_embed(x)
        x = x + self.pos_embed
        
        # Hierarchical attention layers
        all_meta = []
        for layer in self.layers:
            # Attention
            if MOSAIC_AVAILABLE:
                attn_out, meta = layer['attn'](layer['norm1'](x))
                all_meta.append(meta)
            else:
                normed = layer['norm1'](x)
                attn_out, _ = layer['attn'](normed, normed, normed)
            
            x = x + attn_out
            
            # FFN
            x = x + layer['ffn'](layer['norm2'](x))
        
        # Pool over genes and classify
        x = x.mean(dim=1)  # (batch, embed_dim)
        logits = self.classifier(x)
        
        return logits, all_meta


def setup_distributed():
    """Initialize distributed training."""
    if 'RANK' in os.environ:
        dist.init_process_group('nccl')
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


def main():
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f'cuda:{local_rank}')
    is_main = rank == 0
    
    if is_main:
        print(f"=== Tabula Sapiens Training ===")
        print(f"GPUs: {world_size}")
        print(f"Mosaic available: {MOSAIC_AVAILABLE}")
    
    # Load data
    dataset = GCSDataset(BUCKET)
    n_genes = dataset.metadata['n_genes']
    n_classes = dataset.metadata['n_cell_types']
    
    # Distributed sampler
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank) if world_size > 1 else None
    loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=4,
        pin_memory=True,
    )
    
    # Model
    model = HierarchicalCellTypeClassifier(
        n_genes=n_genes,
        n_classes=n_classes,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
    ).to(device)
    
    if world_size > 1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    
    if is_main:
        print(f"\nModel: {sum(p.numel() for p in model.parameters()):,} parameters")
        print(f"Data: {len(dataset):,} cells, {n_genes} genes, {n_classes} cell types")
        print(f"Batch size: {BATCH_SIZE} × {world_size} = {BATCH_SIZE * world_size}")
        print()
    
    # Training loop
    for epoch in range(EPOCHS):
        if sampler:
            sampler.set_epoch(epoch)
        
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (X, y) in enumerate(loader):
            X, y = X.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits, meta = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            
            if is_main and batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS} | Batch {batch_idx}/{len(loader)} | "
                      f"Loss: {loss.item():.4f} | Acc: {100*correct/total:.1f}%")
        
        # Epoch stats
        if world_size > 1:
            stats = torch.tensor([total_loss, correct, total], device=device)
            dist.all_reduce(stats)
            total_loss, correct, total = stats.tolist()
        
        if is_main:
            avg_loss = total_loss / len(loader)
            acc = 100 * correct / total
            print(f"\nEpoch {epoch+1} complete | Loss: {avg_loss:.4f} | Accuracy: {acc:.1f}%\n")
            
            # Log routing weights from last batch
            if meta and MOSAIC_AVAILABLE:
                routing = meta[0].get('routing_weights')
                if routing is not None:
                    print(f"Routing weights (layer 0): {routing[0].cpu().numpy()}")
    
    # Save model
    if is_main:
        torch.save(model.state_dict(), "/tmp/model.pt")
        os.system(f"gcloud storage cp /tmp/model.pt {BUCKET}/model.pt")
        print(f"\nModel saved to {BUCKET}/model.pt")
    
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()


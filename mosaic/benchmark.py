"""
Mosaic Benchmarking Suite.

Run with:
    python -m mosaic.benchmark --seq_lengths 1k,10k,100k --backends local,ring

Or via torchrun for multi-GPU:
    torchrun --nproc_per_node=4 -m mosaic.benchmark --backends ring,mesh2d
"""

import argparse
import json
import time
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
import torch
import torch.distributed as dist

import mosaic
from mosaic.attention import MultiAxisAttention


@dataclass
class BenchmarkResult:
    backend: str
    seq_length: int
    batch_size: int
    num_heads: int
    head_dim: int
    num_gpus: int
    forward_ms: float
    backward_ms: float
    memory_mb: float
    throughput_tokens_per_sec: float


def parse_size(s: str) -> int:
    """Parse size string like '1k', '10k', '1M'."""
    s = s.strip().lower()
    multipliers = {'k': 1000, 'm': 1000000, 'b': 1000000000}
    if s[-1] in multipliers:
        return int(float(s[:-1]) * multipliers[s[-1]])
    return int(s)


def benchmark_attention(
    backend: str,
    seq_length: int,
    batch_size: int = 2,
    embed_dim: int = 128,
    num_heads: int = 8,
    warmup_iters: int = 3,
    bench_iters: int = 10,
    mesh_shape: tuple = None,
) -> BenchmarkResult:
    """
    Benchmark a single attention configuration.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Determine world size
    if dist.is_initialized():
        world_size = dist.get_world_size()
        local_seq = seq_length // world_size
    else:
        world_size = 1
        local_seq = seq_length
    
    # Create attention layer
    if backend == "mesh2d" and mesh_shape is None:
        # Infer mesh shape
        import math
        side = int(math.sqrt(world_size))
        mesh_shape = (side, world_size // side)
    
    attn = MultiAxisAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        attention_axis=1,
        backend=backend,
        mesh_shape=mesh_shape,
    ).to(device)
    
    head_dim = embed_dim // num_heads
    
    # Create input
    x = torch.randn(batch_size, local_seq, embed_dim, device=device, requires_grad=True)
    
    # Warmup
    for _ in range(warmup_iters):
        out = attn(x)
        out.sum().backward()
        x.grad = None
    
    torch.cuda.synchronize()
    
    # Benchmark forward
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    
    for _ in range(bench_iters):
        out = attn(x)
        torch.cuda.synchronize()
    
    forward_time = (time.perf_counter() - start) / bench_iters
    
    # Benchmark backward
    start = time.perf_counter()
    
    for _ in range(bench_iters):
        out = attn(x)
        out.sum().backward()
        x.grad = None
        torch.cuda.synchronize()
    
    total_time = (time.perf_counter() - start) / bench_iters
    backward_time = total_time - forward_time
    
    # Memory
    memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    
    # Throughput
    tokens_per_iter = batch_size * seq_length
    throughput = tokens_per_iter / total_time
    
    return BenchmarkResult(
        backend=backend,
        seq_length=seq_length,
        batch_size=batch_size,
        num_heads=num_heads,
        head_dim=head_dim,
        num_gpus=world_size,
        forward_ms=forward_time * 1000,
        backward_ms=backward_time * 1000,
        memory_mb=memory_mb,
        throughput_tokens_per_sec=throughput,
    )


def run_benchmarks(
    seq_lengths: List[int],
    backends: List[str],
    batch_size: int = 2,
    embed_dim: int = 128,
    num_heads: int = 8,
) -> List[BenchmarkResult]:
    """Run benchmarks for all configurations."""
    results = []
    
    for backend in backends:
        for seq_len in seq_lengths:
            try:
                result = benchmark_attention(
                    backend=backend,
                    seq_length=seq_len,
                    batch_size=batch_size,
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                )
                results.append(result)
                
                # Print progress
                if not dist.is_initialized() or dist.get_rank() == 0:
                    print(f"{backend:8} | seq={seq_len:>8} | "
                          f"fwd={result.forward_ms:>8.2f}ms | "
                          f"bwd={result.backward_ms:>8.2f}ms | "
                          f"mem={result.memory_mb:>8.0f}MB | "
                          f"tok/s={result.throughput_tokens_per_sec:>10.0f}")
            
            except Exception as e:
                if not dist.is_initialized() or dist.get_rank() == 0:
                    print(f"{backend:8} | seq={seq_len:>8} | ERROR: {e}")
    
    return results


def print_summary(results: List[BenchmarkResult]):
    """Print summary table."""
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    
    # Group by backend
    by_backend: Dict[str, List[BenchmarkResult]] = {}
    for r in results:
        by_backend.setdefault(r.backend, []).append(r)
    
    print(f"{'Backend':<10} {'SeqLen':>10} {'GPUs':>6} {'Fwd(ms)':>10} {'Bwd(ms)':>10} {'Mem(MB)':>10} {'Tok/s':>12}")
    print("-" * 80)
    
    for backend, backend_results in by_backend.items():
        for r in sorted(backend_results, key=lambda x: x.seq_length):
            print(f"{r.backend:<10} {r.seq_length:>10} {r.num_gpus:>6} "
                  f"{r.forward_ms:>10.2f} {r.backward_ms:>10.2f} "
                  f"{r.memory_mb:>10.0f} {r.throughput_tokens_per_sec:>12.0f}")


def main():
    parser = argparse.ArgumentParser(description="Mosaic Benchmark Suite")
    parser.add_argument(
        "--seq_lengths", type=str, default="1k,10k",
        help="Comma-separated sequence lengths (e.g., '1k,10k,100k')"
    )
    parser.add_argument(
        "--backends", type=str, default="local",
        help="Comma-separated backends (e.g., 'local,ring,mesh2d')"
    )
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    
    args = parser.parse_args()
    
    # Parse inputs
    seq_lengths = [parse_size(s) for s in args.seq_lengths.split(",")]
    backends = [b.strip() for b in args.backends.split(",")]
    
    # Initialize distributed if available
    if torch.cuda.is_available():
        try:
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl")
            mosaic.init(sp_size=dist.get_world_size())
        except:
            pass  # Single GPU mode
    
    # Print header
    is_main = not dist.is_initialized() or dist.get_rank() == 0
    if is_main:
        print("=" * 80)
        print("MOSAIC BENCHMARK")
        print("=" * 80)
        print(f"Sequence lengths: {seq_lengths}")
        print(f"Backends: {backends}")
        print(f"Batch size: {args.batch_size}")
        print(f"Embed dim: {args.embed_dim}")
        print(f"Num heads: {args.num_heads}")
        if dist.is_initialized():
            print(f"GPUs: {dist.get_world_size()}")
        print("=" * 80 + "\n")
    
    # Run benchmarks
    results = run_benchmarks(
        seq_lengths=seq_lengths,
        backends=backends,
        batch_size=args.batch_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
    )
    
    # Print summary
    if is_main:
        print_summary(results)
        
        # Save to JSON
        if args.output:
            with open(args.output, "w") as f:
                json.dump([asdict(r) for r in results], f, indent=2)
            print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()


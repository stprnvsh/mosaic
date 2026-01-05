# Mosaic on Google Cloud

Deploy multi-node GPU training with Mosaic on Google Cloud Vertex AI.

## Quick Start

### 1. Prerequisites

```bash
# Install gcloud CLI
# https://cloud.google.com/sdk/docs/install

# Authenticate
gcloud auth login
gcloud auth application-default login

# Set project
gcloud config set project YOUR_PROJECT_ID
```

### 2. Setup (One Time)

```bash
# Run setup script
chmod +x gcp/setup.sh
./gcp/setup.sh YOUR_PROJECT_ID us-central1
```

This will:
- Enable required APIs
- Create Artifact Registry repository
- Build and push Docker image
- Create GCS bucket for outputs

### 3. Submit Training Job

```bash
python gcp/submit_job.py \
    --project=YOUR_PROJECT_ID \
    --region=us-central1 \
    --image=us-central1-docker.pkg.dev/YOUR_PROJECT_ID/mosaic/mosaic:latest \
    --bucket=gs://YOUR_PROJECT_ID-mosaic-output \
    --num_nodes=2 \
    --seq_len=100000
```

### 4. Monitor Job

```bash
# Via CLI
gcloud ai custom-jobs list --project=YOUR_PROJECT_ID --region=us-central1

# Or via Console
# https://console.cloud.google.com/vertex-ai/training/custom-jobs
```

## Configuration Options

### Machine Types

| Machine Type | GPUs | GPU Memory | Use Case |
|--------------|------|------------|----------|
| `a2-highgpu-1g` | 1× A100 | 40GB | Testing |
| `a2-highgpu-8g` | 8× A100 | 320GB | Production |
| `a2-ultragpu-8g` | 8× A100 80GB | 640GB | Large models |
| `a3-highgpu-8g` | 8× H100 | 640GB | Maximum perf |

### Scaling

```bash
# 2 nodes × 8 GPUs = 16 GPUs (default)
python gcp/submit_job.py --num_nodes=2 --seq_len=100000

# 4 nodes × 8 GPUs = 32 GPUs
python gcp/submit_job.py --num_nodes=4 --seq_len=500000

# 8 nodes × 8 GPUs = 64 GPUs
python gcp/submit_job.py --num_nodes=8 --seq_len=1000000
```

## Costs

Approximate costs (us-central1, on-demand):

| Configuration | GPUs | $/hour |
|---------------|------|--------|
| 1× a2-highgpu-8g | 8× A100 | ~$30 |
| 2× a2-highgpu-8g | 16× A100 | ~$60 |
| 4× a2-highgpu-8g | 32× A100 | ~$120 |

Use spot/preemptible instances for ~70% savings.

## Troubleshooting

### Check logs
```bash
gcloud ai custom-jobs describe JOB_ID \
    --project=YOUR_PROJECT_ID \
    --region=us-central1
```

### Common issues

1. **Quota exceeded**: Request GPU quota increase in Cloud Console
2. **NCCL timeout**: Check network configuration, ensure all nodes can communicate
3. **OOM**: Reduce batch size or sequence length, enable checkpointing

## Files

```
gcp/
├── README.md          # This file
├── setup.sh           # One-time setup script
├── train_gcp.py       # Training script
└── submit_job.py      # Job submission script
```


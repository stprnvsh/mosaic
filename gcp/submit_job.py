"""
Submit Mosaic training job to Google Cloud Vertex AI.

Prerequisites:
    1. gcloud CLI installed and authenticated
    2. Docker image pushed to Artifact Registry
    3. GCS bucket for output

Usage:
    python submit_job.py \
        --project=your-project-id \
        --region=us-central1 \
        --image=us-central1-docker.pkg.dev/your-project/mosaic/mosaic:latest \
        --bucket=gs://your-bucket/mosaic-output
"""

import argparse
from google.cloud import aiplatform


def submit_training_job(
    project: str,
    region: str,
    image_uri: str,
    output_bucket: str,
    machine_type: str = "a2-highgpu-8g",  # 8x A100
    num_nodes: int = 2,
    seq_len: int = 100000,
    steps: int = 100,
):
    """Submit a multi-node training job to Vertex AI."""
    
    aiplatform.init(project=project, location=region)
    
    # Training arguments
    args = [
        f"--seq_len={seq_len}",
        f"--steps={steps}",
        f"--batch_size=4",
        f"--output_dir=/gcs/output",
    ]
    
    # Create custom job
    job = aiplatform.CustomJob(
        display_name="mosaic-multinode-training",
        worker_pool_specs=[
            {
                # Master node
                "machine_spec": {
                    "machine_type": machine_type,
                    "accelerator_type": "NVIDIA_TESLA_A100",
                    "accelerator_count": 8,
                },
                "replica_count": 1,
                "container_spec": {
                    "image_uri": image_uri,
                    "command": ["torchrun"],
                    "args": [
                        f"--nnodes={num_nodes}",
                        "--nproc_per_node=8",
                        "--node_rank=0",
                        "--master_addr=$MASTER_ADDR",
                        "--master_port=29500",
                        "/app/mosaic/gcp/train_gcp.py",
                    ] + args,
                },
            },
        ] + [
            {
                # Worker nodes
                "machine_spec": {
                    "machine_type": machine_type,
                    "accelerator_type": "NVIDIA_TESLA_A100",
                    "accelerator_count": 8,
                },
                "replica_count": num_nodes - 1,
                "container_spec": {
                    "image_uri": image_uri,
                    "command": ["torchrun"],
                    "args": [
                        f"--nnodes={num_nodes}",
                        "--nproc_per_node=8",
                        "--node_rank=$RANK",
                        "--master_addr=$MASTER_ADDR",
                        "--master_port=29500",
                        "/app/mosaic/gcp/train_gcp.py",
                    ] + args,
                },
            }
        ] if num_nodes > 1 else [],
        staging_bucket=output_bucket,
    )
    
    print(f"Submitting job with {num_nodes} nodes × 8 GPUs = {num_nodes * 8} total GPUs")
    print(f"Sequence length: {seq_len}")
    
    job.run(sync=False)
    
    print(f"Job submitted: {job.resource_name}")
    print(f"Monitor at: https://console.cloud.google.com/vertex-ai/training/custom-jobs?project={project}")
    
    return job


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, help="GCP project ID")
    parser.add_argument("--region", default="us-central1", help="GCP region")
    parser.add_argument("--image", required=True, help="Docker image URI")
    parser.add_argument("--bucket", required=True, help="GCS bucket for output")
    parser.add_argument("--machine_type", default="a2-highgpu-8g")
    parser.add_argument("--num_nodes", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=100000)
    parser.add_argument("--steps", type=int, default=100)
    
    args = parser.parse_args()
    
    submit_training_job(
        project=args.project,
        region=args.region,
        image_uri=args.image,
        output_bucket=args.bucket,
        machine_type=args.machine_type,
        num_nodes=args.num_nodes,
        seq_len=args.seq_len,
        steps=args.steps,
    )


if __name__ == "__main__":
    main()


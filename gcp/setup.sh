#!/bin/bash
# Setup script for deploying Mosaic to Google Cloud
# 
# Usage:
#   ./setup.sh <PROJECT_ID> <REGION>
#
# Example:
#   ./setup.sh my-project us-central1

set -e

PROJECT_ID=${1:-$(gcloud config get-value project)}
REGION=${2:-us-central1}
REPO_NAME="mosaic"
IMAGE_NAME="mosaic"
IMAGE_TAG="latest"

echo "=== Mosaic GCP Setup ==="
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo ""

# 1. Enable required APIs
echo "Enabling APIs..."
gcloud services enable \
    artifactregistry.googleapis.com \
    aiplatform.googleapis.com \
    storage.googleapis.com \
    --project=$PROJECT_ID

# 2. Create Artifact Registry repository
echo "Creating Artifact Registry repository..."
gcloud artifacts repositories create $REPO_NAME \
    --repository-format=docker \
    --location=$REGION \
    --project=$PROJECT_ID \
    2>/dev/null || echo "Repository already exists"

# 3. Configure Docker authentication
echo "Configuring Docker authentication..."
gcloud auth configure-docker ${REGION}-docker.pkg.dev --quiet

# 4. Build and push Docker image
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:${IMAGE_TAG}"
echo "Building Docker image: $IMAGE_URI"

cd "$(dirname "$0")/.."
docker build -t $IMAGE_URI .
docker push $IMAGE_URI

# 5. Create GCS bucket for outputs
BUCKET_NAME="gs://${PROJECT_ID}-mosaic-output"
echo "Creating GCS bucket: $BUCKET_NAME"
gcloud storage buckets create $BUCKET_NAME \
    --project=$PROJECT_ID \
    --location=$REGION \
    2>/dev/null || echo "Bucket already exists"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Image URI: $IMAGE_URI"
echo "Output Bucket: $BUCKET_NAME"
echo ""
echo "To submit a training job:"
echo ""
echo "  python gcp/submit_job.py \\"
echo "      --project=$PROJECT_ID \\"
echo "      --region=$REGION \\"
echo "      --image=$IMAGE_URI \\"
echo "      --bucket=$BUCKET_NAME \\"
echo "      --num_nodes=2 \\"
echo "      --seq_len=100000"
echo ""


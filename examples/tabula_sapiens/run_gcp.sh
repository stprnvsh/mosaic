#!/bin/bash
# Deploy Tabula Sapiens training to GCP Vertex AI
#
# Usage:
#   ./run_gcp.sh

set -e

PROJECT_ID=${PROJECT_ID:-mosaic-gpu}
REGION="europe-central2"
REPO_NAME="mosaic"
IMAGE_NAME="tabula-sapiens"
IMAGE_TAG="latest"
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:${IMAGE_TAG}"
DATA_BUCKET="gs://${PROJECT_ID}-tabula-sapiens"

echo "=== Tabula Sapiens Training Setup ==="
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo "Image: $IMAGE_URI"
echo "Data: $DATA_BUCKET"
echo ""

# 1. Enable APIs
echo "1. Enabling APIs..."
gcloud services enable \
    aiplatform.googleapis.com \
    artifactregistry.googleapis.com \
    storage.googleapis.com \
    --project=$PROJECT_ID

# 2. Create Artifact Registry repo
echo "2. Creating Artifact Registry..."
gcloud artifacts repositories create $REPO_NAME \
    --repository-format=docker \
    --location=$REGION \
    --project=$PROJECT_ID 2>/dev/null || true

# 3. Configure Docker auth
echo "3. Configuring Docker auth..."
gcloud auth configure-docker ${REGION}-docker.pkg.dev --quiet

# 4. Build and push Docker image
echo "4. Building Docker image..."
cd "$(dirname "$0")"
docker build -t $IMAGE_URI .
docker push $IMAGE_URI

# 5. Check if data exists, if not run setup
echo "5. Checking data..."
if ! gcloud storage ls ${DATA_BUCKET}/X.npy &>/dev/null; then
    echo "   Data not found, running setup on Cloud Shell..."
    echo "   This downloads Tabula Sapiens (~2GB) and uploads to GCS."
    echo ""
    echo "   Run this command in Cloud Shell:"
    echo "   pip install cellxgene-census scanpy && python setup_data.py"
    echo ""
    echo "   Or run a small VM:"
    gcloud compute instances create tabula-setup \
        --zone=${REGION}-a \
        --machine-type=e2-standard-4 \
        --image-family=debian-11 \
        --image-project=debian-cloud \
        --metadata=startup-script="#!/bin/bash
apt-get update && apt-get install -y python3-pip
pip3 install cellxgene-census scanpy google-cloud-storage
cat > /tmp/setup.py << 'EOF'
$(cat setup_data.py)
EOF
python3 /tmp/setup.py
poweroff" \
        --scopes=cloud-platform \
        --project=$PROJECT_ID || true
    
    echo "   Waiting for data setup VM..."
    sleep 60
fi

# 6. Submit training job
echo "6. Submitting training job..."
gcloud ai custom-jobs create \
    --region=$REGION \
    --display-name=tabula-sapiens-hierarchical \
    --worker-pool-spec=machine-type=n1-standard-32,accelerator-type=NVIDIA_TESLA_T4,accelerator-count=8,replica-count=1,container-image-uri=$IMAGE_URI \
    --args="--nproc_per_node=8","/app/train.py" \
    --env-vars="DATA_BUCKET=${DATA_BUCKET}" \
    --project=$PROJECT_ID

echo ""
echo "=== Job Submitted ==="
echo "Monitor at: https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=$PROJECT_ID"


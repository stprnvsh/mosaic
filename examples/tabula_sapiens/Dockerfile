# Tabula Sapiens training container
FROM us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-2.py310:latest

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    mosaic-attention \
    cellxgene-census \
    scanpy \
    anndata

# Copy training scripts
COPY setup_data.py /app/
COPY train.py /app/

# Default: run training
CMD ["torchrun", "--nproc_per_node=8", "/app/train.py"]


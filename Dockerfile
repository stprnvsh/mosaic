# Mosaic Multi-Node GPU Training Container
# For Google Cloud Vertex AI / GKE

FROM nvcr.io/nvidia/pytorch:24.01-py3

WORKDIR /app

# Install mosaic from PyPI
RUN pip install --no-cache-dir mosaic-attention

# Install optional dependencies
RUN pip install --no-cache-dir flash-attn --no-build-isolation || true
RUN pip install --no-cache-dir google-cloud-storage google-cloud-aiplatform

# Copy training scripts
COPY gcp/ /app/gcp/

# Environment for distributed training
ENV NCCL_DEBUG=INFO
ENV NCCL_IB_DISABLE=0
ENV NCCL_NET_GDR_LEVEL=2

CMD ["python", "-c", "import mosaic; print(f'Mosaic {mosaic.__version__} ready')"]


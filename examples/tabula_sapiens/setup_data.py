"""
Download and prepare Tabula Sapiens for GCS.

Run this on a GCP VM or Cloud Shell to download data directly to GCS.
"""

import os
import subprocess

# Config
PROJECT_ID = os.environ.get("PROJECT_ID", "mosaic-gpu")
BUCKET = f"gs://{PROJECT_ID}-tabula-sapiens"
REGION = "europe-central2"


def run(cmd):
    print(f"$ {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def main():
    print("=== Tabula Sapiens Setup ===")
    
    # Create bucket
    print("\n1. Creating GCS bucket...")
    run(f"gcloud storage buckets create {BUCKET} --location={REGION} --project={PROJECT_ID} 2>/dev/null || true")
    
    # Install dependencies
    print("\n2. Installing dependencies...")
    run("pip install -q cellxgene-census scanpy anndata pyarrow")
    
    # Download and process data
    print("\n3. Downloading Tabula Sapiens via CellxGene Census...")
    
    import cellxgene_census
    import scanpy as sc
    import numpy as np
    import pandas as pd
    
    # Open census
    census = cellxgene_census.open_soma()
    
    # Query Tabula Sapiens (human, all tissues)
    # This is ~500k cells × 20k genes
    print("   Querying Tabula Sapiens...")
    adata = cellxgene_census.get_anndata(
        census,
        organism="Homo sapiens",
        obs_value_filter="dataset_id == '53d208b0-2cfd-4366-9866-c3c6114081bc'",  # Tabula Sapiens
        var_value_filter="feature_biotype == 'gene'",
    )
    
    print(f"   Downloaded: {adata.shape[0]} cells × {adata.shape[1]} genes")
    
    # Preprocess
    print("\n4. Preprocessing...")
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    # Select highly variable genes (reduce to manageable size)
    sc.pp.highly_variable_genes(adata, n_top_genes=5000)
    adata = adata[:, adata.var.highly_variable]
    
    print(f"   After filtering: {adata.shape[0]} cells × {adata.shape[1]} genes")
    
    # Extract arrays
    X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    cell_types = adata.obs['cell_type'].values
    
    # Encode cell types
    unique_types = np.unique(cell_types)
    type_to_idx = {t: i for i, t in enumerate(unique_types)}
    y = np.array([type_to_idx[t] for t in cell_types])
    
    print(f"   {len(unique_types)} cell types")
    
    # Save locally first
    print("\n5. Saving to local files...")
    os.makedirs("/tmp/tabula_sapiens", exist_ok=True)
    np.save("/tmp/tabula_sapiens/X.npy", X.astype(np.float32))
    np.save("/tmp/tabula_sapiens/y.npy", y.astype(np.int64))
    np.save("/tmp/tabula_sapiens/cell_types.npy", unique_types)
    
    # Save metadata
    metadata = {
        "n_cells": X.shape[0],
        "n_genes": X.shape[1],
        "n_cell_types": len(unique_types),
        "cell_types": list(unique_types),
    }
    import json
    with open("/tmp/tabula_sapiens/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    
    # Upload to GCS
    print("\n6. Uploading to GCS...")
    run(f"gcloud storage cp -r /tmp/tabula_sapiens/* {BUCKET}/")
    
    print(f"\n=== Done! Data available at {BUCKET}/ ===")
    print(f"   X.npy: {X.shape}")
    print(f"   y.npy: {y.shape}")
    print(f"   cell_types: {len(unique_types)}")
    
    census.close()


if __name__ == "__main__":
    main()


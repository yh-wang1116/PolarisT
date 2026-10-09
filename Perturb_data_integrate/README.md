# Perturbation Data Integration

This directory contains the supplementary integration code used to generate a
low-dimensional representation of a large single-cell AnnData object for the
PolarisT analysis.

The code is provided for transparency and method reproduction. It is not
required for the core perturbation-ranking functions and is not
executed when importing `polarist`.

## Main script

```text
Integrate_data.py
```

The script trains a conditional variational autoencoder with:

- dataset-conditioned encoding and decoding;
- negative-binomial reconstruction;
- triplet-loss supervision using immune-cell population labels;
- a 30-dimensional latent representation;
- optional UMAP visualization of the integrated representation.

## Input data

The input must be an AnnData file in `.h5ad` format containing:

```python
adata.layers["counts"]
adata.obs["Dataset"]
adata.obs["Immune_type"]
```

`adata.layers["counts"]` should contain raw count data. `Dataset` is used as
the batch covariate, and `Immune_type` is used to construct the triplet-loss
labels.

The complete AnnData input is not distributed with this repository because of
its large size.

## Running the analysis

```bash
python Integrate_data.py \
    --input /path/to/input.h5ad \
    --output_model /path/to/model.pth \
    --output_h5ad /path/to/output.h5ad
```

The analysis requires substantial memory and is best run with a CUDA-enabled
GPU. Depending on the input size and hardware, training may take several
hours or longer.

## Output

The integrated latent representation is stored in:

```python
adata.obsm["New_embedding"]
```

The resulting AnnData object can be used for downstream neighbor-graph,
embedding, and cell-state analyses. The trained PyTorch model is also saved as
a checkpoint.

## Important notes

This is supplementary research code rather than a lightweight end-user
workflow. Reproducing the analysis requires a compatible AnnData input and
the corresponding computational resources.

The results may depend on the input data, software versions, random state,
GPU configuration, and available memory. The script currently contains
analysis-specific output conventions; users should check the output paths in
the script before running it in a new environment.

For routine CD8T-cell perturbation ranking, use the core package APIs instead:

```python
from polarist import rank_seen_drivers
from polarist import rank_unseen_drivers
```

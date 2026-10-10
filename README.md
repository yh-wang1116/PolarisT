<p align="center">
  <a href="https://github.com/org-PolarisT/PolarisT/blob/main/PolarisT_icon.png">
    <img width="150" alt="PolarisT" src="./PolarisT_icon.png" />
  </a>
</p>

<h1 align="center">
  Navigating T cell transcriptomic reprogramming by an<br>
  inverse AI virtual T-cell model
</h1>

<p align="center">
  <img src="https://img.shields.io/badge/version-1.0.0-62518C" alt="Version 1.0.0" />
  <img src="https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?logo=python&logoColor=white" alt="Python 3.10, 3.11 or 3.12" />
</p>

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Installation](#-installation)
  - [Package installation](#1-package-installation)
  - [Data preparation](#2-data-preparation)
- [Usage](#-usage)
  - [Atlas-profiled perturbation ranking](#atlas-profiled-perturbation-ranking)
  - [Atlas-unprofiled gene ranking](#atlas-unprofiled-gene-ranking)
  - [Custom phenotype gene sets](#custom-phenotype-gene-sets)
  - [Perturbation data integration](#perturbation-data-integration)
- [Tutorial](#-tutorial)
- [Repository Structure](#-repository-structure)
- [Citation](#-citation)

---

## 💡 Overview

PolarisT is a perturbation-centric AI virtual T-cell model for navigating CD8⁺ T cell transcriptomic reprogramming.

PolarisT provides a framework for learning from immune perturbation atlases, navigating among their genetic interventions and extrapolating beyond their gene coverage, linking perturbation atlases to experimentally testable strategies for CD8⁺ T cell reprogramming.

It contains three related components:

- **Perturbation data integration:** integrates large-scale single-cell perturbation data with a batch-conditioned variational autoencoder and metric-learning supervision.
- **Atlas-profiled perturbations (seen):** ranks perturbations measured in the perturbation atlas according to a user-defined T cell phenotype.
- **Atlas-unprofiled perturbations (unseen):** uses a pretrained graph-informed model to rank candidate genes that were not directly profiled in the atlas.

---

## 📦 Installation

### 1. Package installation

PolarisT requires Python 3.10, 3.11 or 3.12. Create a conda environment and install the package as follows:

```bash
conda create -n polarist python=3.10
conda activate polarist
git clone https://github.com/org-PolarisT/PolarisT.git
cd PolarisT
pip install .
```

To install the optional tutorial and development dependencies:

```bash
pip install ".[demo,dev]"
```

The package includes pretrained models and compact inference resources. Complete the data preparation step below before running the ranking workflows or tutorial notebooks.

### 2. Data preparation

Download the CD8⁺ T-cell AnnData file:

[Download Anndata_cd8_raw.h5ad](https://figshare.com/ndownloader/files/66953270)

The dataset is documented in the associated [Figshare record](https://doi.org/10.6084/m9.figshare.32934569).

Save `Anndata_cd8_raw.h5ad` in a local data directory, for example:

```text
/path/to/polarist_data/Anndata_cd8_raw.h5ad
```

Use this directory as `resource_dir` in both ranking workflows and tutorial notebooks. Replace `/path/to/polarist_data` in the examples with your actual directory.

**Note:** `resource_dir` must point to the directory containing `Anndata_cd8_raw.h5ad`, not to the file itself.

---

## 🚀 Usage

### Atlas-profiled perturbation ranking

`rank_seen_drivers()` defines a phenotype from positive and negative gene signatures, selects the corresponding extreme cells, and ranks perturbations already represented in the atlas.

```python
from polarist import rank_seen_drivers

result = rank_seen_drivers(
    positive_genes=[
        "TCF7", "LEF1", "SLAMF6", "SELL", "BCL2",
        "BCL6", "CXCR5", "CCNE1", "CCNE2",
    ],
    negative_genes=["TOX", "HAVCR2", "ENTPD1", "CD101", "CD244"],
    phenotype_name="stemness",
    extreme_fraction=0.05,
    refinement_weight=0.9,
    tf_only=True,
    resource_dir="/path/to/polarist_data",
)

ranking = result.ranking
print(ranking.head())
```

### Atlas-unprofiled gene ranking

`rank_unseen_drivers()` ranks genes that were not directly profiled as perturbations in the released atlas.

```python
from polarist import rank_unseen_drivers

result = rank_unseen_drivers(
    positive_genes=[
        "TCF7", "LEF1", "SLAMF6", "SELL", "BCL2",
        "BCL6", "CXCR5", "CCNE1", "CCNE2",
    ],
    negative_genes=["TOX", "HAVCR2", "ENTPD1", "CD101", "CD244"],
    phenotype_name="stemness",
    extreme_fraction=0.05,
    n_genes=1000,
    tf_only=True,
    resource_dir="/path/to/polarist_data",
)

ranking = result.ranking
print(ranking.head())
```

### Custom phenotype gene sets

Users can define custom CD8⁺ T-cell phenotypes by providing their own positive and negative gene sets. Replace the gene sets in either ranking workflow and set `phenotype_name` to a descriptive label. Use the same `resource_dir` configured during data preparation.

For example:

```python
from polarist import rank_seen_drivers

result = rank_seen_drivers(
    positive_genes=["NKG7", "GZMB", "IFNG"],
    negative_genes=["TOX", "PDCD1"],
    phenotype_name="cytotoxicity",
    extreme_fraction=0.05,
    refinement_weight=0.9,
    tf_only=True,
    resource_dir="/path/to/polarist_data",
)

ranking = result.ranking
print(ranking.head())
```

Custom gene sets can also be used with `rank_unseen_drivers()`.

### Perturbation data integration

The integration component uses a batch-conditioned variational autoencoder with metric-learning supervision to generate a low-dimensional representation of large-scale single-cell perturbation data. The implementation is provided in `Perturb_data_integrate/Integrate_data.py`.

The script requires a compatible AnnData object containing raw counts and the metadata fields `Dataset` and `Immune_type`. Because the input data are large, the complete AnnData object is not included. This analysis requires substantial memory and is best run with a CUDA-enabled GPU.

See [`Perturb_data_integrate/README.md`](Perturb_data_integrate/README.md) for the input requirements and command-line usage.

---

## 📖 Tutorial

The [`Tutorial/`](Tutorial/) directory contains runnable notebooks for both ranking workflows:

- [Atlas-profiled perturbation ranking](Tutorial/atlas_profiled_driver_demo.ipynb)
- [Atlas-unprofiled gene ranking](Tutorial/atlas_unprofiled_driver_demo.ipynb)

Before running either notebook, complete [Data preparation](#2-data-preparation) and set `resource_dir` in the notebook to your local data directory.

The accompanying CSV files contain example ranking outputs.

---

## 📂 Repository Structure

```text
src/polarist/
├── __init__.py       # Public package API and version
├── phenotype.py      # Gene-signature scoring and phenotype definition
├── seen.py           # Atlas-profiled perturbation ranking
├── unseen.py         # Atlas-unprofiled gene ranking
├── workflow.py       # High-level seen workflow
├── models.py         # Neural-network components for unseen inference
└── resources/        # Bundled embeddings, graph and pretrained weights

Perturb_data_integrate/
└── Integrate_data.py # Large-scale integration and metric-learning analysis

Tutorial/            # Example notebooks and ranking outputs
tests/               # Package tests
```

---

## 📝 Citation

If you find PolarisT useful for your research, please consider citing our paper:

```bibtex
@article{polarist_submitted,
  title   = {Navigating T cell transcriptomic reprogramming by an inverse AI virtual T-cell model},
  journal = {Submitted},
  year    = {2026}
}
```

The citation will be updated with the preprint or publication DOI once available.

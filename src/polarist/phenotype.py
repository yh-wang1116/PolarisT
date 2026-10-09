"""Phenotype scoring, cell selection, and direction-vector construction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import mannwhitneyu


@dataclass(frozen=True)
class SignatureScoreResult:
    """Per-cell signature scores and dataset-local extreme-cell assignments."""

    scores: pd.DataFrame
    phases: pd.DataFrame
    stats: pd.DataFrame
    gene_coverage: pd.DataFrame


@dataclass(frozen=True)
class PhenotypeVectorResult:
    """Desired-minus-undesired phenotype direction in cell-embedding space."""

    vector: pd.Series
    desired_mean: pd.Series
    undesired_mean: pd.Series
    n_desired: int
    n_undesired: int


def _validate_fractions(top_fraction: float, bottom_fraction: float) -> None:
    if not 0 < top_fraction <= 1:
        raise ValueError("top_fraction must be in (0, 1]")
    if not 0 < bottom_fraction <= 1:
        raise ValueError("bottom_fraction must be in (0, 1]")
    if top_fraction + bottom_fraction > 1:
        raise ValueError("top_fraction + bottom_fraction must not exceed 1")


def _align_metadata(
    metadata: pd.DataFrame,
    cell_ids: pd.Index,
    cell_id_key: str | None,
) -> pd.DataFrame:
    aligned = metadata.copy()
    if cell_id_key is not None:
        if cell_id_key not in aligned.columns:
            raise KeyError(f"metadata does not contain cell ID column {cell_id_key!r}")
        aligned.index = aligned[cell_id_key].astype(str)
    else:
        aligned.index = aligned.index.astype(str)

    if aligned.index.has_duplicates:
        duplicates = aligned.index[aligned.index.duplicated()].unique().tolist()[:5]
        raise ValueError(f"metadata cell IDs must be unique; duplicates include {duplicates}")

    missing = cell_ids.difference(aligned.index)
    if len(missing):
        raise ValueError(f"metadata is missing {len(missing)} expression-matrix cells")
    return aligned.loc[cell_ids]


def _gene_set_scores(
    expression,
    row_positions: np.ndarray,
    gene_positions: np.ndarray,
    scale_genes: bool,
) -> np.ndarray:
    subset = expression[:, gene_positions][row_positions]
    if scale_genes:
        dense = subset.toarray() if sparse.issparse(subset) else np.asarray(subset, dtype=float)
        means = dense.mean(axis=0)
        stds = dense.std(axis=0, ddof=1)
        if np.any(stds == 0) or np.any(~np.isfinite(stds)):
            raise ValueError("cannot z-score a gene with zero or undefined variance")
        return ((dense - means) / stds).mean(axis=1)

    if sparse.issparse(subset):
        return np.asarray(subset.mean(axis=1)).ravel()
    return np.asarray(subset, dtype=float).mean(axis=1)


def score_gene_sets(
    expression,
    cell_ids: Sequence[str],
    gene_names: Sequence[str],
    metadata: pd.DataFrame,
    gene_sets: Mapping[str, Sequence[str]],
    *,
    dataset_key: str = "Dataset",
    cell_type_key: str = "Immune_type",
    cell_type: str = "CD8T",
    cell_id_key: str | None = "Unique_cellid",
    top_fraction: float = 0.05,
    bottom_fraction: float = 0.05,
    scale_genes: bool = False,
    min_genes: int = 1,
    compute_diagnostics: bool = True,
    cell_order: Sequence[str] | None = None,
) -> SignatureScoreResult:
    """Score gene sets and assign top/bottom cells within each dataset.

    The expression matrix must be cells by genes. Extreme cells are selected
    separately within each dataset, matching the original R implementation.
    """

    _validate_fractions(top_fraction, bottom_fraction)
    if min_genes < 1:
        raise ValueError("min_genes must be at least 1")
    if not gene_sets:
        raise ValueError("gene_sets must not be empty")

    cell_index = pd.Index([str(cell_id) for cell_id in cell_ids], name="cell_id")
    gene_index = pd.Index([str(gene) for gene in gene_names], name="gene")
    if cell_index.has_duplicates:
        raise ValueError("cell_ids must be unique")
    if gene_index.has_duplicates:
        raise ValueError("gene_names must be unique")
    if expression.shape != (len(cell_index), len(gene_index)):
        raise ValueError(
            "expression shape must equal (len(cell_ids), len(gene_names)); "
            f"got {expression.shape} versus {(len(cell_index), len(gene_index))}"
        )

    aligned_meta = _align_metadata(metadata, cell_index, cell_id_key)
    for key in (dataset_key, cell_type_key):
        if key not in aligned_meta.columns:
            raise KeyError(f"metadata does not contain required column {key!r}")

    target_mask = aligned_meta[cell_type_key].astype(str).eq(str(cell_type)).to_numpy()
    target_positions = np.flatnonzero(target_mask)
    if not len(target_positions):
        raise ValueError(f"no cells matched {cell_type_key}={cell_type!r}")

    target_ids = cell_index[target_positions]
    if cell_order is None:
        target_order = np.arange(len(target_ids), dtype=np.int64)
    else:
        ordered_ids = pd.Index([str(cell_id) for cell_id in cell_order], name="cell_id")
        if ordered_ids.has_duplicates:
            raise ValueError("cell_order must contain unique cell IDs")
        order_by_id = pd.Series(
            np.arange(len(ordered_ids), dtype=np.int64),
            index=ordered_ids,
        )
        missing_order = target_ids.difference(order_by_id.index)
        if len(missing_order):
            raise ValueError(f"cell_order is missing {len(missing_order)} target cells")
        target_order = order_by_id.loc[target_ids].to_numpy(dtype=np.int64)

    scores = pd.DataFrame(index=target_ids, columns=list(gene_sets), dtype=float)
    phases = pd.DataFrame(index=target_ids, columns=list(gene_sets), dtype=object)
    gene_to_position = {gene: idx for idx, gene in enumerate(gene_index)}

    stats_rows: list[dict] = []
    coverage_rows: list[dict] = []
    target_meta = aligned_meta.iloc[target_positions]

    for dataset in pd.unique(target_meta[dataset_key]):
        dataset_mask = target_meta[dataset_key].eq(dataset).to_numpy()
        dataset_target_positions = np.flatnonzero(dataset_mask)
        dataset_target_positions = dataset_target_positions[
            np.argsort(target_order[dataset_target_positions], kind="stable")
        ]
        dataset_positions = target_positions[dataset_target_positions]
        dataset_ids = cell_index[dataset_positions]
        n_cells = len(dataset_positions)

        for gene_set_name, requested_genes in gene_sets.items():
            requested = list(dict.fromkeys(str(gene) for gene in requested_genes))
            present = [gene for gene in requested if gene in gene_to_position]
            missing = [gene for gene in requested if gene not in gene_to_position]
            coverage_rows.append(
                {
                    "dataset": dataset,
                    "gene_set": gene_set_name,
                    "n_requested": len(requested),
                    "n_present": len(present),
                    "coverage": len(present) / max(len(requested), 1),
                    "missing_genes": "|".join(missing),
                }
            )
            if len(present) < min_genes:
                continue

            gene_positions = np.asarray([gene_to_position[gene] for gene in present])
            values = _gene_set_scores(
                expression,
                dataset_positions,
                gene_positions,
                scale_genes,
            )
            if np.any(~np.isfinite(values)):
                raise ValueError(
                    f"non-finite scores for dataset={dataset!r}, gene_set={gene_set_name!r}"
                )

            scores.loc[dataset_ids, gene_set_name] = values
            phases.loc[dataset_ids, gene_set_name] = "mid"

            order = np.argsort(-values, kind="stable")
            n_top = max(1, int(np.floor(top_fraction * n_cells)))
            n_bottom = max(1, int(np.floor(bottom_fraction * n_cells)))
            top_local = order[:n_top]
            bottom_local = order[n_cells - n_bottom :]
            top_ids = dataset_ids[top_local]
            bottom_ids = dataset_ids[bottom_local]
            phases.loc[top_ids, gene_set_name] = "top"
            phases.loc[bottom_ids, gene_set_name] = "bottom"

            top_scores = values[top_local]
            bottom_scores = values[bottom_local]
            p_value = np.nan
            if compute_diagnostics:
                p_value = float(
                    mannwhitneyu(top_scores, bottom_scores, alternative="two-sided").pvalue
                )
            stats_rows.append(
                {
                    "dataset": dataset,
                    "gene_set": gene_set_name,
                    "n_genes": len(present),
                    "n_top_cells": n_top,
                    "n_bottom_cells": n_bottom,
                    "top_mean": float(top_scores.mean()),
                    "bottom_mean": float(bottom_scores.mean()),
                    "diff_mean": float(top_scores.mean() - bottom_scores.mean()),
                    "p_value": p_value,
                }
            )

    return SignatureScoreResult(
        scores=scores,
        phases=phases,
        stats=pd.DataFrame(stats_rows),
        gene_coverage=pd.DataFrame(coverage_rows),
    )


def score_gene_sets_anndata(
    adata,
    gene_sets: Mapping[str, Sequence[str]],
    *,
    layer: str | None = None,
    **kwargs,
) -> SignatureScoreResult:
    """AnnData convenience wrapper around :func:`score_gene_sets`."""

    expression = adata.layers[layer] if layer is not None else adata.X
    return score_gene_sets(
        expression=expression,
        cell_ids=adata.obs_names,
        gene_names=adata.var_names,
        metadata=adata.obs,
        gene_sets=gene_sets,
        cell_id_key=None,
        **kwargs,
    )


def assign_combined_phenotype(
    phases: pd.DataFrame,
    desired_conditions: Mapping[str, str],
    undesired_conditions: Mapping[str, str],
    *,
    desired_label: str = "top",
    undesired_label: str = "bottom",
) -> pd.Series:
    """Combine per-signature phases into desired and undesired cell groups."""

    if not desired_conditions or not undesired_conditions:
        raise ValueError("desired_conditions and undesired_conditions must not be empty")

    required = set(desired_conditions) | set(undesired_conditions)
    missing = required - set(phases.columns)
    if missing:
        raise KeyError(f"phase table is missing gene sets: {sorted(missing)}")

    desired_mask = pd.Series(True, index=phases.index)
    for gene_set, expected_phase in desired_conditions.items():
        desired_mask &= phases[gene_set].eq(expected_phase)

    undesired_mask = pd.Series(True, index=phases.index)
    for gene_set, expected_phase in undesired_conditions.items():
        undesired_mask &= phases[gene_set].eq(expected_phase)

    overlap = desired_mask & undesired_mask
    if overlap.any():
        raise ValueError("desired and undesired phenotype rules selected overlapping cells")

    labels = pd.Series("mid", index=phases.index, name="phenotype_group", dtype=object)
    labels.loc[undesired_mask] = undesired_label
    labels.loc[desired_mask] = desired_label

    if not desired_mask.any():
        raise ValueError("desired phenotype rules selected no cells")
    if not undesired_mask.any():
        raise ValueError("undesired phenotype rules selected no cells")
    return labels


def compute_phenotype_vector(
    cell_embeddings: pd.DataFrame,
    phenotype_labels: pd.Series,
    *,
    desired_label: str = "top",
    undesired_label: str = "bottom",
) -> PhenotypeVectorResult:
    """Compute the desired-minus-undesired centroid difference."""

    embeddings = cell_embeddings.copy()
    embeddings.index = embeddings.index.astype(str)
    if embeddings.index.has_duplicates:
        raise ValueError("cell embedding index must be unique")
    labels = phenotype_labels.copy()
    labels.index = labels.index.astype(str)

    selected = labels[labels.isin([desired_label, undesired_label])]
    missing = selected.index.difference(embeddings.index)
    if len(missing):
        raise ValueError(f"cell embeddings are missing {len(missing)} selected phenotype cells")

    desired_ids = selected.index[selected.eq(desired_label)]
    undesired_ids = selected.index[selected.eq(undesired_label)]
    if not len(desired_ids) or not len(undesired_ids):
        raise ValueError("both desired and undesired cells are required")

    numeric = embeddings.astype(float)
    desired_mean = numeric.loc[desired_ids].mean(axis=0)
    undesired_mean = numeric.loc[undesired_ids].mean(axis=0)
    vector = desired_mean - undesired_mean
    vector.name = "phenotype_vector"
    if not np.isfinite(vector.to_numpy()).all() or np.allclose(vector.to_numpy(), 0):
        raise ValueError("phenotype vector is zero or contains non-finite values")

    return PhenotypeVectorResult(
        vector=vector,
        desired_mean=desired_mean,
        undesired_mean=undesired_mean,
        n_desired=len(desired_ids),
        n_undesired=len(undesired_ids),
    )

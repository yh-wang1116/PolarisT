"""Ranking of atlas-profiled perturbations against a phenotype direction."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .phenotype import (
    PhenotypeVectorResult,
    assign_combined_phenotype,
    compute_phenotype_vector,
)


@dataclass(frozen=True)
class SeenRankingResult:
    """Auditable outputs from the atlas-profiled ranking track."""

    phenotype_labels: pd.Series
    phenotype_vector: PhenotypeVectorResult
    raw_ranking: pd.DataFrame
    refined_ranking: pd.DataFrame | None


def _rowwise_pearson(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    centered_matrix = matrix - matrix.mean(axis=1, keepdims=True)
    centered_vector = vector - vector.mean()
    numerator = centered_matrix @ centered_vector
    denominator = np.linalg.norm(centered_matrix, axis=1) * np.linalg.norm(centered_vector)
    return np.divide(
        numerator,
        denominator,
        out=np.full(matrix.shape[0], np.nan, dtype=float),
        where=denominator != 0,
    )


def _rowwise_spearman(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    ranked_vector = rankdata(vector, method="average")
    ranked_matrix = np.apply_along_axis(rankdata, 1, matrix, method="average")
    return _rowwise_pearson(ranked_matrix, ranked_vector)


def _perturbation_direction(name: str) -> str:
    lowered = name.lower()
    if lowered.endswith(("_gain", "-gain")):
        return "gain"
    if lowered.endswith(("_loss", "-loss")):
        return "loss"
    return "unknown"


def rank_seen_perturbations(
    perturbation_embeddings: pd.DataFrame,
    phenotype_vector: pd.Series,
    *,
    sort_metric: str = "pearson_correlation",
) -> pd.DataFrame:
    """Score measured perturbation embeddings against a phenotype vector."""

    embeddings = perturbation_embeddings.copy()
    embeddings.index = embeddings.index.astype(str)
    if embeddings.index.has_duplicates:
        raise ValueError("perturbation embedding index must be unique")
    if not embeddings.columns.equals(phenotype_vector.index):
        missing = phenotype_vector.index.difference(embeddings.columns)
        extra = embeddings.columns.difference(phenotype_vector.index)
        if len(missing) or len(extra):
            raise ValueError(
                "perturbation and phenotype embedding dimensions differ; "
                f"missing={missing.tolist()[:5]}, extra={extra.tolist()[:5]}"
            )
        embeddings = embeddings.loc[:, phenotype_vector.index]

    matrix = embeddings.to_numpy(dtype=float)
    vector = phenotype_vector.to_numpy(dtype=float)
    if matrix.shape[1] != vector.size:
        raise ValueError("perturbation and phenotype embeddings must have the same dimension")
    if not np.isfinite(matrix).all() or not np.isfinite(vector).all():
        raise ValueError("embeddings must contain only finite values")

    result = pd.DataFrame(index=embeddings.index)
    result.index.name = "perturbation"
    result["dot_product"] = matrix @ vector
    result["pearson_correlation"] = _rowwise_pearson(matrix, vector)
    result["spearman_correlation"] = _rowwise_spearman(matrix, vector)
    result["perturbation_direction"] = [_perturbation_direction(name) for name in result.index]

    if sort_metric not in result.columns:
        raise ValueError(f"unknown sort metric {sort_metric!r}")
    result["raw_rank"] = result[sort_metric].rank(method="average", ascending=False)
    return result.sort_values(sort_metric, ascending=False, kind="stable", na_position="last")


def refine_seen_ranking(
    raw_ranking: pd.DataFrame,
    cluster_groups: pd.Series,
    *,
    score_column: str = "pearson_correlation",
    weight: float = 0.8,
) -> pd.DataFrame:
    """Smooth perturbation scores toward their cluster mean."""

    if not 0 <= weight <= 1:
        raise ValueError("weight must be in [0, 1]")
    if score_column not in raw_ranking.columns:
        raise KeyError(f"raw ranking does not contain {score_column!r}")

    ranking = raw_ranking.copy()
    ranking.index = ranking.index.astype(str)
    groups = cluster_groups.copy()
    groups.index = groups.index.astype(str)
    missing = ranking.index.difference(groups.index)
    if len(missing):
        raise ValueError(f"cluster labels are missing {len(missing)} perturbations")

    ranking["cluster"] = groups.loc[ranking.index].to_numpy()
    ranking["cluster_mean_score"] = ranking.groupby("cluster", sort=False)[score_column].transform(
        "mean"
    )
    ranking["refined_score"] = (
        weight * ranking[score_column] + (1 - weight) * ranking["cluster_mean_score"]
    )
    ranking["refined_rank"] = ranking["refined_score"].rank(method="average", ascending=False)
    return ranking.sort_values("refined_score", ascending=False, kind="stable", na_position="last")


def rank_seen(
    phases: pd.DataFrame,
    cell_embeddings: pd.DataFrame,
    perturbation_embeddings: pd.DataFrame,
    desired_conditions: Mapping[str, str],
    undesired_conditions: Mapping[str, str],
    *,
    cluster_groups: pd.Series | None = None,
    refinement_weight: float = 0.8,
    sort_metric: str = "pearson_correlation",
) -> SeenRankingResult:
    """Run phenotype-cell selection, direction construction, and seen ranking."""

    labels = assign_combined_phenotype(
        phases,
        desired_conditions=desired_conditions,
        undesired_conditions=undesired_conditions,
    )
    vector_result = compute_phenotype_vector(cell_embeddings, labels)
    raw = rank_seen_perturbations(
        perturbation_embeddings,
        vector_result.vector,
        sort_metric=sort_metric,
    )
    refined = None
    if cluster_groups is not None:
        refined = refine_seen_ranking(
            raw,
            cluster_groups,
            score_column=sort_metric,
            weight=refinement_weight,
        )
    return SeenRankingResult(
        phenotype_labels=labels,
        phenotype_vector=vector_result,
        raw_ranking=raw,
        refined_ranking=refined,
    )

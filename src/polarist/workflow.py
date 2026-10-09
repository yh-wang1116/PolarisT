"""High-level workflow for atlas-profiled PolarisT driver ranking."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .phenotype import (
    assign_combined_phenotype,
    compute_phenotype_vector,
    score_gene_sets,
)
from .seen import rank_seen_perturbations, refine_seen_ranking

BUNDLED_RESOURCE_DIR = Path(__file__).with_name("resources")
DEFAULT_RESOURCE_DIR = BUNDLED_RESOURCE_DIR
RESOURCE_DIR_ENV = "POLARIST_RESOURCE_DIR"


@dataclass(frozen=True)
class SeenResourcePaths:
    """Files required by the atlas-profiled ranking workflow."""

    resource_dir: Path

    @property
    def adata(self) -> Path:
        return self._large_resource("Anndata_cd8_raw.h5ad")

    @property
    def cell_order(self) -> Path:
        return self._bundled_resource("Seen_cell_order.csv")

    @property
    def cell_embeddings(self) -> Path:
        for filename in ("Seen_cd8_cell_embedding.npz", "Seen_cell_embedding.csv"):
            direct = self.resource_dir / filename
            legacy = self.resource_dir / "seen_rank" / filename
            bundled = BUNDLED_RESOURCE_DIR / filename
            for candidate in (direct, legacy, bundled):
                if candidate.is_file():
                    return candidate
        return BUNDLED_RESOURCE_DIR / "Seen_cd8_cell_embedding.npz"

    @property
    def perturbation_embeddings(self) -> Path:
        return self._bundled_resource("Seen_perturbation_vector.csv")

    @property
    def perturbation_groups(self) -> Path:
        return self._bundled_resource("Seen_perturbation_groups.csv")

    @property
    def transcription_factors(self) -> Path:
        return self._bundled_resource("Human_tf_list.txt")

    def _large_resource(self, filename: str) -> Path:
        direct = self.resource_dir / filename
        legacy = self.resource_dir / "seen_rank" / filename
        return legacy if not direct.is_file() and legacy.is_file() else direct

    def _bundled_resource(self, filename: str) -> Path:
        direct = self.resource_dir / filename
        legacy = self.resource_dir / "seen_rank" / filename
        for candidate in (direct, legacy, BUNDLED_RESOURCE_DIR / filename):
            if candidate.is_file():
                return candidate
        return BUNDLED_RESOURCE_DIR / filename


@dataclass(frozen=True)
class SeenDriverResult:
    """Final ranking and auditable intermediate outputs."""

    phenotype_name: str
    positive_genes: tuple[str, ...]
    negative_genes: tuple[str, ...]
    extreme_fraction: float
    refinement_weight: float
    phenotype_cells: pd.DataFrame
    phenotype_vector: pd.Series
    raw_ranking: pd.DataFrame
    refined_ranking: pd.DataFrame
    full_ranking: pd.DataFrame
    tf_ranking: pd.DataFrame | None
    diagnostics: Mapping[str, pd.DataFrame]
    tf_only: bool

    @property
    def ranking(self) -> pd.DataFrame:
        """Return the requested final output ranking."""

        if self.tf_only:
            if self.tf_ranking is None:
                raise RuntimeError("TF-only ranking was requested but not generated")
            return self.tf_ranking
        return self.full_ranking

    def save(self, output_dir: str | Path) -> dict[str, Path]:
        """Save final and intermediate results as CSV files."""

        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        prefix = _filename_component(self.phenotype_name)

        paths = {
            "ranking": destination / f"{prefix}_seen_ranking.csv",
            "phenotype_cells": destination / f"{prefix}_phenotype_cells.csv",
            "phenotype_vector": destination / f"{prefix}_phenotype_vector.csv",
            "raw_ranking": destination / f"{prefix}_raw_ranking.csv",
            "refined_ranking": destination / f"{prefix}_refined_ranking.csv",
        }
        self.ranking.to_csv(paths["ranking"], index=True)
        self.phenotype_cells.to_csv(paths["phenotype_cells"], index=False)
        self.phenotype_vector.rename("value").to_csv(paths["phenotype_vector"], index=True)
        self.raw_ranking.to_csv(paths["raw_ranking"], index=True)
        self.refined_ranking.to_csv(paths["refined_ranking"], index=True)

        for name, frame in self.diagnostics.items():
            path = destination / f"{prefix}_{_filename_component(name)}.csv"
            frame.to_csv(path, index=False)
            paths[name] = path
        return paths


@dataclass(frozen=True)
class _PhenotypeDefinition:
    cells: pd.DataFrame
    labels: pd.Series
    signature_stats: pd.DataFrame
    gene_coverage: pd.DataFrame


class SeenRankingWorkflow:
    """Reusable workflow with lazy loading and caching of fixed resources."""

    def __init__(
        self,
        resource_dir: str | Path | None = None,
        *,
        adata: Any | None = None,
        cell_order: Sequence[str] | None = None,
        cell_embeddings: pd.DataFrame | None = None,
        perturbation_embeddings: pd.DataFrame | None = None,
        perturbation_groups: pd.Series | None = None,
        transcription_factors: Sequence[str] | None = None,
    ) -> None:
        base_dir = Path(resource_dir) if resource_dir is not None else _default_resource_dir()
        self.paths = SeenResourcePaths(base_dir)
        self._adata = adata
        self._cell_order = (
            tuple(str(cell_id) for cell_id in cell_order) if cell_order is not None else None
        )
        self._cell_embeddings = cell_embeddings.copy() if cell_embeddings is not None else None
        self._perturbation_embeddings = (
            perturbation_embeddings.copy() if perturbation_embeddings is not None else None
        )
        self._perturbation_groups = (
            perturbation_groups.copy() if perturbation_groups is not None else None
        )
        self._transcription_factors = (
            frozenset(str(gene).upper() for gene in transcription_factors)
            if transcription_factors is not None
            else None
        )

    def rank(
        self,
        positive_genes: Sequence[str] | None = None,
        negative_genes: Sequence[str] | None = None,
        *,
        phenotype_name: str = "phenotype",
        extreme_fraction: float = 0.05,
        refinement_weight: float = 0.9,
        tf_only: bool = False,
        verbose: bool = True,
        output_dir: str | Path | None = None,
    ) -> SeenDriverResult:
        """Rank atlas-profiled perturbations toward a user-defined phenotype.

        Desired cells have high positive-signature scores and/or low
        negative-signature scores. Undesired cells have the inverse pattern.
        """

        positive, negative = _validate_phenotype_inputs(
            positive_genes,
            negative_genes,
            extreme_fraction,
        )
        if not 0 <= refinement_weight <= 1:
            raise ValueError("refinement_weight must be in [0, 1]")
        _filename_component(phenotype_name)

        total_steps = 5 if tf_only else 4
        _report_progress(
            verbose,
            1,
            total_steps,
            "Scoring phenotype signatures and selecting "
            f"the top {extreme_fraction:.1%} and bottom {extreme_fraction:.1%} "
            "cells within GEX...",
        )

        phenotype_definition = self._define_phenotype(
            positive,
            negative,
            extreme_fraction,
        )

        vector_result = compute_phenotype_vector(
            self.cell_embeddings,
            phenotype_definition.labels,
        )
        _report_progress(
            verbose,
            2,
            total_steps,
            "Phenotype cells selected; phenotype direction vector computed.",
        )
        perturbation_embeddings = self.perturbation_embeddings.copy()
        if perturbation_embeddings.shape[1] != len(vector_result.vector):
            raise ValueError(
                "cell and perturbation embeddings have different dimensions: "
                f"{len(vector_result.vector)} versus {perturbation_embeddings.shape[1]}"
            )
        perturbation_embeddings.columns = vector_result.vector.index

        raw_ranking = rank_seen_perturbations(
            perturbation_embeddings,
            vector_result.vector,
            sort_metric="pearson_correlation",
        )
        _report_progress(
            verbose,
            3,
            total_steps,
            "Initial perturbation-phenotype alignment completed.",
        )
        perturbation_groups = self.perturbation_groups
        missing_groups = raw_ranking.index.difference(perturbation_groups.index)
        grouped_raw_ranking = raw_ranking.drop(index=missing_groups)
        if grouped_raw_ranking.empty:
            raise ValueError("no ranked perturbations have refinement groups")
        refined_ranking = refine_seen_ranking(
            grouped_raw_ranking,
            perturbation_groups,
            score_column="pearson_correlation",
            weight=refinement_weight,
        )
        full_ranking = _add_gene_columns(refined_ranking).drop(columns="cluster")
        _report_progress(
            verbose,
            4,
            total_steps,
            "Final perturbation ranking completed.",
        )

        tf_ranking = None
        if tf_only:
            transcription_factors = self.transcription_factors
            tf_ranking = full_ranking.loc[
                full_ranking["gene"].str.upper().isin(transcription_factors)
            ].copy()
            _report_progress(
                verbose,
                5,
                total_steps,
                f"Transcription-factor ranking completed ({len(tf_ranking)} TFs).",
            )

        counts = (
            phenotype_definition.labels.value_counts()
            .rename_axis("phenotype_group")
            .rename("n_cells")
            .reset_index()
        )
        diagnostics = {
            "signature_stats": phenotype_definition.signature_stats,
            "gene_coverage": phenotype_definition.gene_coverage,
            "phenotype_counts": counts,
            "unrefined_perturbations": _add_gene_columns(raw_ranking.loc[missing_groups])
            .rename_axis("perturbation")
            .reset_index(),
        }
        result = SeenDriverResult(
            phenotype_name=phenotype_name,
            positive_genes=positive,
            negative_genes=negative,
            extreme_fraction=extreme_fraction,
            refinement_weight=refinement_weight,
            phenotype_cells=phenotype_definition.cells,
            phenotype_vector=vector_result.vector,
            raw_ranking=_add_gene_columns(raw_ranking),
            refined_ranking=full_ranking,
            full_ranking=full_ranking,
            tf_ranking=tf_ranking,
            diagnostics=diagnostics,
            tf_only=tf_only,
        )
        if output_dir is not None:
            result.save(output_dir)
        return result

    def define_phenotype_cells(
        self,
        positive_genes: Sequence[str] | None = None,
        negative_genes: Sequence[str] | None = None,
        *,
        extreme_fraction: float = 0.05,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """Define desired and undesired CD8 T-cell states from gene signatures."""

        positive, negative = _validate_phenotype_inputs(
            positive_genes,
            negative_genes,
            extreme_fraction,
        )
        definition = self._define_phenotype(positive, negative, extreme_fraction)
        if verbose:
            counts = definition.cells["phenotype_group"].value_counts()
            print(
                "Phenotype cells selected: "
                f"{int(counts.get('top', 0))} desired (top) and "
                f"{int(counts.get('bottom', 0))} undesired (bottom) cells.",
                flush=True,
            )
        return definition.cells.copy()

    def _define_phenotype(
        self,
        positive: tuple[str, ...],
        negative: tuple[str, ...],
        extreme_fraction: float,
    ) -> _PhenotypeDefinition:
        adata = self.adata
        required_obs = {"Unique_cellid", "Dataset", "Immune_type"}
        missing_obs = required_obs.difference(adata.obs.columns)
        if missing_obs:
            raise KeyError(f"AnnData obs is missing columns: {sorted(missing_obs)}")
        if "logNor" not in adata.layers:
            raise KeyError("AnnData is missing the required 'logNor' expression layer")

        gene_sets = {}
        desired_conditions = {}
        undesired_conditions = {}
        if positive:
            gene_sets["positive"] = positive
            desired_conditions["positive"] = "top"
            undesired_conditions["positive"] = "bottom"
        if negative:
            gene_sets["negative"] = negative
            desired_conditions["negative"] = "bottom"
            undesired_conditions["negative"] = "top"

        signature_result = score_gene_sets(
            expression=adata.layers["logNor"],
            cell_ids=adata.obs["Unique_cellid"].astype(str).tolist(),
            gene_names=adata.var_names.astype(str),
            metadata=adata.obs,
            gene_sets=gene_sets,
            dataset_key="Dataset",
            cell_type_key="Immune_type",
            cell_type="CD8T",
            cell_id_key="Unique_cellid",
            top_fraction=extreme_fraction,
            bottom_fraction=extreme_fraction,
            scale_genes=False,
            min_genes=1,
            compute_diagnostics=True,
            cell_order=self.cell_order,
        )
        labels = assign_combined_phenotype(
            signature_result.phases,
            desired_conditions=desired_conditions,
            undesired_conditions=undesired_conditions,
        )
        cells = _build_phenotype_cells(
            adata.obs,
            signature_result.scores,
            signature_result.phases,
            labels,
        )
        return _PhenotypeDefinition(
            cells=cells,
            labels=labels,
            signature_stats=signature_result.stats,
            gene_coverage=signature_result.gene_coverage,
        )

    @property
    def adata(self) -> Any:
        if self._adata is None:
            try:
                import anndata as ad
            except ImportError as exc:
                raise ImportError(
                    "Reading Anndata_cd8_raw.h5ad requires the 'anndata' dependency"
                ) from exc
            _require_file(self.paths.adata)
            self._adata = ad.read_h5ad(self.paths.adata)
        return self._adata

    @property
    def cell_order(self) -> tuple[str, ...]:
        if self._cell_order is None:
            frame = _read_csv(self.paths.cell_order)
            if "Unique_cellid" not in frame.columns:
                raise KeyError("cell-order file must contain 'Unique_cellid'")
            ids = frame["Unique_cellid"].astype(str)
            if ids.duplicated().any():
                raise ValueError("cell-order file contains duplicate cell IDs")
            self._cell_order = tuple(ids)
        return self._cell_order

    @property
    def cell_embeddings(self) -> pd.DataFrame:
        if self._cell_embeddings is None:
            path = self.paths.cell_embeddings
            _require_file(path)
            if path.suffix.lower() == ".npz":
                with np.load(path, allow_pickle=False) as archive:
                    ids = pd.Index(archive["cell_ids"].astype(str), name="Unique_cellid")
                    columns = archive["columns"].astype(str)
                    numeric = pd.DataFrame(archive["embeddings"], index=ids, columns=columns)
            else:
                frame = _read_csv(path)
                if "Unique_cellid" not in frame.columns:
                    raise KeyError("cell-embedding file must contain 'Unique_cellid'")
                frame["Unique_cellid"] = frame["Unique_cellid"].astype(str)
                numeric = frame.drop(columns="Unique_cellid").select_dtypes(include=[np.number])
                numeric.index = frame["Unique_cellid"]
            if numeric.index.has_duplicates:
                raise ValueError("cell-embedding file contains duplicate cell IDs")
            if numeric.empty:
                raise ValueError("cell-embedding file contains no numeric dimensions")
            self._cell_embeddings = numeric.astype(float)
        else:
            self._cell_embeddings.index = self._cell_embeddings.index.astype(str)
        return self._cell_embeddings

    @property
    def perturbation_embeddings(self) -> pd.DataFrame:
        if self._perturbation_embeddings is None:
            _require_file(self.paths.perturbation_embeddings)
            frame = pd.read_csv(self.paths.perturbation_embeddings, index_col=0)
            frame = _drop_unnamed_columns(frame).select_dtypes(include=[np.number])
            if frame.empty:
                raise ValueError("perturbation-embedding file contains no numeric dimensions")
            frame.index = frame.index.astype(str)
            self._perturbation_embeddings = frame.astype(float)
        else:
            self._perturbation_embeddings.index = self._perturbation_embeddings.index.astype(str)
        return self._perturbation_embeddings

    @property
    def perturbation_groups(self) -> pd.Series:
        if self._perturbation_groups is None:
            frame = _read_csv(self.paths.perturbation_groups)
            required = {"Perturbation", "Group"}
            missing = required.difference(frame.columns)
            if missing:
                raise KeyError(f"perturbation-group file is missing columns: {sorted(missing)}")
            frame = frame.loc[:, ["Perturbation", "Group"]].copy()
            frame["Perturbation"] = frame["Perturbation"].astype(str)
            if frame["Perturbation"].duplicated().any():
                raise ValueError("perturbation-group file contains duplicate perturbations")
            self._perturbation_groups = frame.set_index("Perturbation")["Group"]
        else:
            self._perturbation_groups.index = self._perturbation_groups.index.astype(str)
        return self._perturbation_groups

    @property
    def transcription_factors(self) -> frozenset[str]:
        if self._transcription_factors is None:
            _require_file(self.paths.transcription_factors)
            genes = []
            for line in self.paths.transcription_factors.read_text(encoding="utf-8").splitlines():
                gene = line.strip().split()[0] if line.strip() else ""
                if gene and not gene.startswith("#"):
                    genes.append(gene.upper())
            if not genes:
                raise ValueError("TF list is empty")
            self._transcription_factors = frozenset(genes)
        return self._transcription_factors


def rank_seen_drivers(
    positive_genes: Sequence[str] | None = None,
    negative_genes: Sequence[str] | None = None,
    *,
    phenotype_name: str = "phenotype",
    extreme_fraction: float = 0.05,
    refinement_weight: float = 0.9,
    tf_only: bool = False,
    verbose: bool = True,
    output_dir: str | Path | None = None,
    resource_dir: str | Path | None = None,
) -> SeenDriverResult:
    """Run the atlas-profiled PolarisT workflow."""

    resolved_resource_dir = (
        Path(resource_dir).expanduser() if resource_dir is not None else _default_resource_dir()
    )
    return _cached_default_workflow(str(resolved_resource_dir)).rank(
        positive_genes,
        negative_genes,
        phenotype_name=phenotype_name,
        extreme_fraction=extreme_fraction,
        refinement_weight=refinement_weight,
        tf_only=tf_only,
        verbose=verbose,
        output_dir=output_dir,
    )


def define_phenotype_cells(
    positive_genes: Sequence[str] | None = None,
    negative_genes: Sequence[str] | None = None,
    *,
    extreme_fraction: float = 0.05,
    verbose: bool = True,
    resource_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Select desired (top) and undesired (bottom) CD8 T-cell states."""

    resolved_resource_dir = (
        Path(resource_dir).expanduser() if resource_dir is not None else _default_resource_dir()
    )
    return _cached_default_workflow(str(resolved_resource_dir)).define_phenotype_cells(
        positive_genes,
        negative_genes,
        extreme_fraction=extreme_fraction,
        verbose=verbose,
    )


@lru_cache(maxsize=4)
def _cached_default_workflow(resource_dir: str) -> SeenRankingWorkflow:
    return SeenRankingWorkflow(resource_dir)


def _default_resource_dir() -> Path:
    return Path(os.environ.get(RESOURCE_DIR_ENV, DEFAULT_RESOURCE_DIR)).expanduser()


def _report_progress(
    verbose: bool,
    step: int,
    total_steps: int,
    message: str,
) -> None:
    if verbose:
        print(f"[{step}/{total_steps}] {message}", flush=True)


def _normalize_genes(
    genes: Sequence[str] | None,
) -> tuple[str, ...]:
    if genes is None:
        return ()
    normalized = tuple(dict.fromkeys(str(gene).strip() for gene in genes if str(gene).strip()))
    return normalized


def _validate_phenotype_inputs(
    positive_genes: Sequence[str] | None,
    negative_genes: Sequence[str] | None,
    extreme_fraction: float,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    positive = _normalize_genes(positive_genes)
    negative = _normalize_genes(negative_genes)
    if not positive and not negative:
        raise ValueError("at least one of positive_genes or negative_genes is required")
    if not 0 < extreme_fraction <= 0.5:
        raise ValueError("extreme_fraction must be in (0, 0.5]")
    overlap = set(positive) & set(negative)
    if overlap:
        raise ValueError(
            f"positive_genes and negative_genes must not overlap; overlap={sorted(overlap)}"
        )
    return positive, negative


def _build_phenotype_cells(
    metadata: pd.DataFrame,
    scores: pd.DataFrame,
    phases: pd.DataFrame,
    labels: pd.Series,
) -> pd.DataFrame:
    selected_labels = labels.loc[labels.ne("mid")]
    metadata_by_id = metadata.copy()
    metadata_by_id.index = metadata_by_id["Unique_cellid"].astype(str)
    if metadata_by_id.index.has_duplicates:
        raise ValueError("AnnData obs contains duplicate Unique_cellid values")

    result_data = {
        "Unique_cellid": selected_labels.index,
        "Dataset": metadata_by_id.loc[selected_labels.index, "Dataset"].astype(str).to_numpy(),
    }
    for signature_name in scores.columns:
        result_data[f"{signature_name}_score"] = scores.loc[
            selected_labels.index, signature_name
        ].to_numpy()
        result_data[f"{signature_name}_phase"] = phases.loc[
            selected_labels.index, signature_name
        ].to_numpy()
    result_data["phenotype_group"] = selected_labels.to_numpy()
    return pd.DataFrame(result_data)


def _add_gene_columns(ranking: pd.DataFrame) -> pd.DataFrame:
    result = ranking.copy()
    result.insert(0, "gene", [_base_gene(name) for name in result.index])
    return result


def _base_gene(perturbation: str) -> str:
    return re.sub(r"(?:_|-)(?:gain|loss)$", "", str(perturbation), flags=re.IGNORECASE)


def _filename_component(value: str) -> str:
    component = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip()).strip("_.-")
    if not component:
        raise ValueError("phenotype_name must contain a filename-safe character")
    return component


def _read_csv(path: Path) -> pd.DataFrame:
    _require_file(path)
    return _drop_unnamed_columns(pd.read_csv(path))


def _drop_unnamed_columns(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[:, ~frame.columns.astype(str).str.startswith("Unnamed:")]


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"required PolarisT resource not found: {path}")

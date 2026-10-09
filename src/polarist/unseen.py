"""Frozen-model ranking for genes that were unseen during training."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .models import PPIGCN, FusionCellSetClassifier
try:
    import torch
except ImportError:  # pragma: no cover - gives a useful error at call time
    torch = None  # type: ignore[assignment]


PACKAGE_RESOURCE_DIR = Path(__file__).with_name("resources")
RESOURCE_DIR_ENV = "POLARIST_UNSEEN_RESOURCE_DIR"
LEGACY_RESOURCE_DIR_ENV = "POLARIST_RESOURCE_DIR"
CHECKPOINT_SHA256 = "1f84c27a92a9dc15c3e7df0ed4b21042bf48f78caf825c6c5f55f60fefc87503"
EXPECTED_GRAPH_NODES = 16_073
EXPECTED_GRAPH_EDGES = 795_004
EXPECTED_CANDIDATES = 15_963
EXPECTED_FEATURE_DIM = 34
HUB_BETA = 2.0
TF_NUMERICAL_TIE_TOLERANCE = 5e-7
FORMAL_STEMNESS_POSITIVE_GENES = frozenset(
    {"TCF7", "LEF1", "SLAMF6", "SELL", "BCL2", "BCL6", "CXCR5", "CCNE1", "CCNE2"}
)
FORMAL_STEMNESS_NEGATIVE_GENES = frozenset(
    {"TOX", "HAVCR2", "ENTPD1", "CD101", "CD244"}
)
FORMAL_STEMNESS_EXTREME_FRACTION = 0.05


@dataclass(frozen=True)
class UnseenResourcePaths:
    """Resolved resource roots used by the unseen-driver workflow."""

    resource_dir: Path | None = None

    @property
    def roots(self) -> tuple[Path, ...]:
        roots: list[Path] = []
        if self.resource_dir is not None:
            roots.append(Path(self.resource_dir).expanduser())
        for variable in (RESOURCE_DIR_ENV, LEGACY_RESOURCE_DIR_ENV):
            value = os.environ.get(variable)
            if value:
                roots.append(Path(value).expanduser())
        cache_root = os.environ.get("XDG_CACHE_HOME")
        if cache_root:
            roots.append(Path(cache_root).expanduser() / "polarist" / "unseen_v1")
        else:
            roots.append(Path.home() / ".cache" / "polarist" / "unseen_v1")
        roots.append(PACKAGE_RESOURCE_DIR)

        unique: list[Path] = []
        for root in roots:
            if root not in unique:
                unique.append(root)
        return tuple(unique)

    def find(self, *relative_paths: str) -> Path | None:
        for root in self.roots:
            for relative in relative_paths:
                candidate = root / relative
                if candidate.is_file():
                    return candidate
        return None

    @property
    def checkpoint(self) -> Path:
        path = self.find(
            "models/baseline1_step216_reference_state.pt",
            "baseline1_step216_reference_state.pt",
        )
        if path is None:
            raise FileNotFoundError(
                "step216 checkpoint was not found; provide resource_dir or install the LFS asset"
            )
        return path

    @property
    def graph(self) -> Path:
        path = self.find("unseen_graph_v1.npz", "graph/unseen_graph_v1.npz")
        if path is None:
            raise FileNotFoundError(
                "unseen graph resource was not found; install package resources or provide resource_dir"
            )
        return path

    @property
    def inference_bundle(self) -> Path:
        path = self.find(
            "inference_features.npz",
            "inference/inference_features.npz",
            "polarist_unseen_v1.npz",
            "inference_assets/polarist_unseen_v1/inference_features.npz",
        )
        if path is None:
            raise FileNotFoundError(
                "inference feature bundle was not found; reinstall package resources "
                "or provide resource_dir"
            )
        return path

    @property
    def manifest(self) -> Path | None:
        bundle = self.find(
            "inference_features.npz",
            "inference/inference_features.npz",
            "polarist_unseen_v1.npz",
            "inference_assets/polarist_unseen_v1/inference_features.npz",
        )
        if bundle is not None:
            for filename in ("manifest.json", "polarist_unseen_manifest.json"):
                candidate = bundle.parent / filename
                if candidate.is_file():
                    return candidate
            return None
        return self.find(
            "manifest.json",
            "inference/manifest.json",
            "polarist_unseen_manifest.json",
        )

    @property
    def transcription_factors(self) -> Path:
        path = self.find("Human_tf_list.txt")
        if path is None:
            raise FileNotFoundError("Human_tf_list.txt was not found")
        return path

@dataclass(frozen=True)
class _GraphResources:
    genes: np.ndarray
    edge_index: np.ndarray
    candidate_mask: np.ndarray
    frequency_penalty: np.ndarray


@dataclass(frozen=True)
class _InferenceBundle:
    cell_ids: np.ndarray
    features: np.ndarray
    feature_names: np.ndarray
    control_mask: np.ndarray
    stemness_context_ids: np.ndarray

    @property
    def control_features(self) -> np.ndarray:
        return self.features[self.control_mask]


class PretrainedUnseenModel:
    """Loaded step216 model plus its frozen graph and candidate universe."""

    def __init__(
        self,
        classifier: FusionCellSetClassifier,
        gcn: PPIGCN,
        edge_index,
        graph: _GraphResources,
        paths: UnseenResourcePaths,
        device,
        checkpoint_path: Path,
        tf_genes: frozenset[str],
    ) -> None:
        self.classifier = classifier
        self.gcn = gcn
        self.edge_index = edge_index
        self.genes = tuple(str(gene) for gene in graph.genes)
        self.candidate_mask = graph.candidate_mask.copy()
        self.frequency_penalty = graph.frequency_penalty.copy()
        self.paths = paths
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.checkpoint_sha256 = _sha256(checkpoint_path)
        self.transcription_factors = tf_genes
        self.perturbation_type = "Gain"
        self._gene_embeddings = None

    @property
    def candidate_genes(self) -> tuple[str, ...]:
        return tuple(gene for gene, keep in zip(self.genes, self.candidate_mask) if keep)

    @property
    def gene_embeddings(self):
        if self._gene_embeddings is None:
            with torch.no_grad():
                self._gene_embeddings = self.gcn(self.edge_index)
        return self._gene_embeddings

    def score(self, control_features: np.ndarray, context_features: np.ndarray) -> np.ndarray:
        """Return one raw model score for every graph gene."""

        control = _validate_features(control_features, "control_features")
        context = _validate_features(context_features, "context_features")
        if not len(control) or not len(context):
            raise ValueError("control_features and context_features must both contain cells")
        x_ctrl = torch.as_tensor(control, dtype=torch.float32, device=self.device).unsqueeze(0)
        x_pert = torch.as_tensor(context, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.classifier.eval()
        self.gcn.eval()
        with torch.no_grad():
            scores = self.classifier(x_ctrl, x_pert, self.gene_embeddings).squeeze(0)
        values = scores.detach().cpu().numpy().astype(np.float64, copy=False)
        if values.shape != (len(self.genes),) or not np.isfinite(values).all():
            raise ValueError("model returned an invalid score vector")
        return values

    def load_inference_bundle(self) -> _InferenceBundle:
        path = self.paths.inference_bundle
        _validate_inference_manifest(self.paths.manifest, path)
        return _read_inference_bundle(path)


@dataclass(frozen=True)
class UnseenDriverResult:
    """Ranking outputs and provenance for one unseen-driver prediction."""

    phenotype_name: str
    context_mode: str
    full_ranking: pd.DataFrame
    tf_ranking: pd.DataFrame | None
    candidate_count: int
    context_cell_count: int
    desired_cell_count: int
    undesired_cell_count: int
    positive_genes: tuple[str, ...]
    negative_genes: tuple[str, ...]
    n_genes: int | None
    tf_only: bool
    perturbation_type: str
    diagnostics: Mapping[str, pd.DataFrame]

    @property
    def ranking(self) -> pd.DataFrame:
        if self.tf_only:
            if self.tf_ranking is None:
                raise RuntimeError("TF ranking was requested but not generated")
            return self.tf_ranking
        return self.full_ranking

    def save(self, output_dir: str | Path) -> dict[str, Path]:
        """Write only the requested final ranking tables."""

        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        prefix = _filename_component(self.phenotype_name)
        paths = {"full_ranking": destination / f"{prefix}_unseen_ranking.csv"}
        self.full_ranking.to_csv(paths["full_ranking"], index=False)
        if self.tf_ranking is not None:
            paths["tf_ranking"] = destination / f"{prefix}_unseen_tf_ranking.csv"
            self.tf_ranking.to_csv(paths["tf_ranking"], index=False)
        return paths


def load_pretrained_unseen_model(
    resource_dir: str | Path | None = None,
    device: str = "cpu",
) -> PretrainedUnseenModel:
    """Load and strictly validate the formal step216 unseen-driver model."""

    if torch is None:  # pragma: no cover
        raise ImportError("unseen-driver inference requires PyTorch and torch-geometric")
    resolved_device = torch.device(device)
    paths = UnseenResourcePaths(Path(resource_dir).expanduser() if resource_dir is not None else None)
    checkpoint_path = paths.checkpoint
    checkpoint_hash = _sha256(checkpoint_path)
    if checkpoint_hash.lower() != CHECKPOINT_SHA256:
        raise ValueError(
            "checkpoint SHA-256 mismatch: "
            f"expected {CHECKPOINT_SHA256}, got {checkpoint_hash.lower()}"
        )

    graph = _read_graph_resources(paths.graph)
    if len(graph.genes) != EXPECTED_GRAPH_NODES:
        raise ValueError(f"expected {EXPECTED_GRAPH_NODES} graph genes, got {len(graph.genes)}")
    if graph.edge_index.shape != (2, EXPECTED_GRAPH_EDGES):
        raise ValueError(
            f"expected edge_index shape (2, {EXPECTED_GRAPH_EDGES}), got {graph.edge_index.shape}"
        )
    if int(graph.candidate_mask.sum()) != EXPECTED_CANDIDATES:
        raise ValueError(
            f"expected {EXPECTED_CANDIDATES} official-unseen candidates, "
            f"got {int(graph.candidate_mask.sum())}"
        )

    checkpoint = _torch_load(checkpoint_path, resolved_device)
    expected_metadata = {
        "baseline": "baseline1_longfit",
        "locked_checkpoint_step": 216,
        "phase": "train",
    }
    if not isinstance(checkpoint, dict) or any(
        checkpoint.get(key) != expected for key, expected in expected_metadata.items()
    ):
        observed = (
            {key: checkpoint.get(key) for key in expected_metadata}
            if isinstance(checkpoint, dict)
            else {}
        )
        raise ValueError(f"checkpoint metadata mismatch: {observed}")
    clf_state = checkpoint.get("clf_state") if isinstance(checkpoint, dict) else None
    gcn_state = checkpoint.get("gcn_state") if isinstance(checkpoint, dict) else None
    if not isinstance(clf_state, Mapping) or not isinstance(gcn_state, Mapping):
        raise TypeError("checkpoint must contain clf_state and gcn_state mappings")
    gene_emb_state = clf_state.get("gene_emb")
    if gene_emb_state is None or tuple(gene_emb_state.shape) != (EXPECTED_GRAPH_NODES, 256):
        raise ValueError("checkpoint gene_emb does not match the formal 16,073 x 256 shape")

    zeros = torch.zeros((EXPECTED_GRAPH_NODES, 256), dtype=torch.float32)
    classifier = FusionCellSetClassifier(
        d_in=EXPECTED_FEATURE_DIM,
        n_classes=EXPECTED_GRAPH_NODES,
        emb_dim=256,
        initial_gene_embeddings=zeros,
    )
    gcn = PPIGCN(zeros, hidden_dim=64)
    try:
        classifier.load_state_dict(clf_state, strict=True)
        gcn.load_state_dict(gcn_state, strict=True)
    except RuntimeError as exc:
        raise ValueError(f"checkpoint architecture mismatch: {exc}") from exc

    classifier.lambda_delta = 0.65
    classifier.fusion_mode = "late"
    classifier.to(resolved_device).eval()
    gcn.to(resolved_device).eval()
    edge_index = torch.as_tensor(graph.edge_index, dtype=torch.long, device=resolved_device)
    tf_genes = _read_tf_genes(paths.transcription_factors)
    model = PretrainedUnseenModel(
        classifier=classifier,
        gcn=gcn,
        edge_index=edge_index,
        graph=graph,
        paths=paths,
        device=resolved_device,
        checkpoint_path=checkpoint_path,
        tf_genes=tf_genes,
    )
    _validate_manifest(paths, model)
    return model


def rank_unseen_drivers(
    desired_cells: Sequence[str] | pd.Series | pd.DataFrame | str | Path | None = None,
    *,
    positive_genes: Sequence[str] | None = None,
    negative_genes: Sequence[str] | None = None,
    reference_context: str | None = None,
    phenotype_name: str = "phenotype",
    extreme_fraction: float = 0.05,
    n_genes: int | None = 1000,
    tf_only: bool = False,
    model: PretrainedUnseenModel | None = None,
    output_dir: str | Path | None = None,
    resource_dir: str | Path | None = None,
    verbose: bool = True,
) -> UnseenDriverResult:
    """Rank official-unseen genes toward a user-defined phenotype.

    By default, provide positive and/or negative signature genes. An exact
    match to the released stemness signature automatically reuses the bundled
    formal context; other signatures use the shared phenotype-definition
    workflow and retain only the desired ``top`` cells. Alternatively,
    ``desired_cells`` can directly provide those target-state cells as IDs, a
    Series, a DataFrame or a CSV path. Set ``reference_context="stemness"``
    explicitly when reproducing the formal frozen result. These three input
    modes cannot be combined.
    """

    positive = _normalize_genes(positive_genes)
    negative = _normalize_genes(negative_genes)
    signature_mode = bool(positive or negative)
    desired_mode = desired_cells is not None
    frozen_mode = reference_context is not None
    if sum((signature_mode, desired_mode, frozen_mode)) != 1:
        raise ValueError(
            "provide exactly one input mode: signature genes, desired_cells, "
            "or reference_context"
        )
    if frozen_mode and str(reference_context).lower() != "stemness":
        raise ValueError("the only bundled reference_context is 'stemness'")
    _validate_n_genes(n_genes)
    _filename_component(phenotype_name)
    bundled_stemness_mode = signature_mode and _matches_formal_stemness(
        positive,
        negative,
        extreme_fraction,
    )

    resolved_resource_dir = (
        str(Path(resource_dir).expanduser()) if resource_dir is not None else None
    )
    total_steps = 4 if tf_only else 3

    loaded_model = model or _cached_pretrained_unseen_model(
        resolved_resource_dir,
        "cpu",
        os.environ.get(RESOURCE_DIR_ENV),
        os.environ.get(LEGACY_RESOURCE_DIR_ENV),
    )
    bundle = loaded_model.load_inference_bundle()
    if bundle.features.shape[1] != EXPECTED_FEATURE_DIM:
        raise ValueError(f"expected {EXPECTED_FEATURE_DIM} inference features, got {bundle.features.shape[1]}")
    control_features = bundle.control_features
    if control_features.shape[0] != 8_149:
        raise ValueError(f"expected 8,149 Gain control cells, got {control_features.shape[0]}")

    if verbose:
        print(f"[1/{total_steps}] Pretrained model loaded.", flush=True)

    diagnostics: dict[str, pd.DataFrame] = {}
    if frozen_mode or bundled_stemness_mode:
        context_ids = _resolve_stemness_ids(bundle)
        context_features = _features_for_ids(bundle, context_ids)
        desired_count = len(context_ids)
        undesired_count = 0
        context_mode = "frozen_context" if frozen_mode else "bundled_reference"
    elif desired_mode:
        context_ids = _read_desired_cell_ids(desired_cells)
        context_features = _features_for_ids(bundle, context_ids)
        desired_count = len(context_ids)
        undesired_count = 0
        context_mode = "desired_cells"
    else:
        from .workflow import define_phenotype_cells

        phenotype_cells = define_phenotype_cells(
            positive_genes=positive,
            negative_genes=negative,
            extreme_fraction=extreme_fraction,
            verbose=False,
            resource_dir=resource_dir,
        )
        desired_frame = phenotype_cells.loc[
            phenotype_cells["phenotype_group"].eq("top")
        ].copy()
        context_ids = _read_desired_cell_ids(desired_frame)
        context_features = _features_for_ids(bundle, context_ids)
        desired_count = len(context_ids)
        undesired_count = 0
        context_mode = "gene_signature"
        diagnostics["phenotype_cells"] = phenotype_cells

    if verbose:
        print(
            f"[2/{total_steps}] Defining phenotype from gene signatures.",
            flush=True,
        )

    raw_scores = loaded_model.score(control_features, context_features)
    candidate_indices = np.flatnonzero(loaded_model.candidate_mask)
    ranking = _make_ranking(loaded_model, raw_scores, candidate_indices)
    full_ranking = _truncate(ranking, n_genes)
    if verbose:
        print(
            f"[3/{total_steps}] Perturbation-phenotype alignment and ranking completed.",
            flush=True,
        )
    tf_ranking = None
    if tf_only:
        tf_ranking = _make_tf_ranking(ranking)
        tf_ranking = _truncate(tf_ranking, n_genes)
        if verbose:
            print(
                f"[4/{total_steps}] Transcription-factor ranking completed "
                f"({len(tf_ranking)} TFs).",
                flush=True,
            )

    result = UnseenDriverResult(
        phenotype_name=phenotype_name,
        context_mode=context_mode,
        full_ranking=full_ranking,
        tf_ranking=tf_ranking,
        candidate_count=len(candidate_indices),
        context_cell_count=len(context_ids),
        desired_cell_count=desired_count,
        undesired_cell_count=undesired_count,
        positive_genes=positive,
        negative_genes=negative,
        n_genes=n_genes,
        tf_only=tf_only,
        perturbation_type=loaded_model.perturbation_type,
        diagnostics=diagnostics,
    )
    if output_dir is not None:
        result.save(output_dir)
    return result


def _make_ranking(
    model: PretrainedUnseenModel,
    raw_scores: np.ndarray,
    candidate_indices: np.ndarray,
) -> pd.DataFrame:
    genes = np.asarray(model.genes, dtype=str)
    penalties = model.frequency_penalty
    rows = pd.DataFrame(
        {
            "gene_idx": candidate_indices.astype(int),
            "gene": genes[candidate_indices],
            "raw_score": raw_scores[candidate_indices],
            "hub_penalty": penalties[candidate_indices],
        }
    )
    rows["score"] = rows["raw_score"] - HUB_BETA * rows["hub_penalty"]
    rows["is_transcription_factor"] = rows["gene"].str.upper().isin(model.transcription_factors)
    rows = rows.sort_values(["score", "gene_idx"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
    rows.insert(0, "rank", np.arange(1, len(rows) + 1, dtype=int))
    return rows[
        ["rank", "gene", "score", "raw_score", "hub_penalty", "is_transcription_factor", "gene_idx"]
    ]


def _truncate(frame: pd.DataFrame, n_genes: int | None) -> pd.DataFrame:
    result = frame if n_genes is None else frame.head(n_genes)
    return result.drop(columns="gene_idx").reset_index(drop=True)


def _make_tf_ranking(ranking: pd.DataFrame) -> pd.DataFrame:
    tf_rows = ranking.loc[ranking["is_transcription_factor"]].copy().reset_index(drop=True)
    scores = tf_rows["score"].to_numpy()
    groups: list[pd.DataFrame] = []
    start = 0
    for stop in range(1, len(tf_rows) + 1):
        if stop == len(tf_rows) or scores[start] - scores[stop] > TF_NUMERICAL_TIE_TOLERANCE:
            groups.append(tf_rows.iloc[start:stop].sort_values("gene_idx", kind="stable"))
            start = stop
    result = pd.concat(groups, ignore_index=True) if groups else tf_rows
    result.insert(0, "tf_rank", np.arange(1, len(result) + 1, dtype=int))
    return result


def _read_desired_cell_ids(
    source: Sequence[str] | pd.Series | pd.DataFrame | str | Path | None,
) -> np.ndarray:
    if source is None:
        raise ValueError("desired_cells is required")
    if isinstance(source, pd.DataFrame):
        if "Unique_cellid" not in source.columns:
            raise KeyError("desired_cells DataFrame must contain 'Unique_cellid'")
        _validate_desired_labels(source)
        values = source["Unique_cellid"]
    elif isinstance(source, pd.Series):
        values = source
    elif isinstance(source, (str, Path)):
        path = Path(source).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"desired phenotype cell file not found: {path}")
        frame = pd.read_csv(path)
        if "Unique_cellid" not in frame.columns:
            raise KeyError("desired phenotype cell CSV must contain 'Unique_cellid'")
        _validate_desired_labels(frame)
        values = frame["Unique_cellid"]
    else:
        values = pd.Series(list(source), dtype="object")

    if values.isna().any():
        raise ValueError("desired_cells contains missing cell IDs")
    ids = values.astype(str).str.strip()
    if ids.empty or ids.eq("").any():
        raise ValueError("desired_cells must contain at least one non-empty cell ID")
    duplicates = ids[ids.duplicated()].unique().tolist()
    if duplicates:
        raise ValueError(f"desired_cells contains duplicate cell IDs: {duplicates[:5]}")
    return ids.to_numpy(dtype=str)


@lru_cache(maxsize=4)
def _cached_pretrained_unseen_model(
    resource_dir: str | None,
    device: str,
    _unseen_resource_env: str | None,
    _legacy_resource_env: str | None,
) -> PretrainedUnseenModel:
    return load_pretrained_unseen_model(resource_dir=resource_dir, device=device)


def _validate_desired_labels(frame: pd.DataFrame) -> None:
    if "phenotype_group" not in frame.columns:
        return
    labels = frame["phenotype_group"].astype(str).str.strip().str.lower()
    if not labels.eq("top").all():
        raise ValueError(
            "desired_cells must contain only phenotype_group == 'top' cells; "
            "filter out bottom cells before unseen ranking"
        )


def _resolve_stemness_ids(bundle: _InferenceBundle) -> np.ndarray:
    ids = np.asarray(bundle.stemness_context_ids, dtype=str)
    if len(ids) != 4_048 or len(np.unique(ids)) != 4_048:
        raise ValueError(f"formal stemness context must contain 4,048 unique cells, got {len(ids)}")
    bundle_ids = set(bundle.cell_ids.astype(str))
    missing = sorted(set(ids) - bundle_ids)
    if missing:
        raise ValueError(f"inference bundle is missing {len(missing)} stemness context cells")
    requested = set(ids)
    return bundle.cell_ids[np.asarray([cell_id in requested for cell_id in bundle.cell_ids])]


def _features_for_ids(bundle: _InferenceBundle, ids: Sequence[str]) -> np.ndarray:
    requested = pd.Index([str(value) for value in ids])
    positions = pd.Series(np.arange(len(bundle.cell_ids)), index=bundle.cell_ids.astype(str))
    missing = requested.difference(positions.index)
    if len(missing):
        raise ValueError(f"inference bundle is missing {len(missing)} requested cells")
    return bundle.features[positions.loc[requested].to_numpy(dtype=int)]


def _read_graph_resources(path: Path) -> _GraphResources:
    with np.load(path, allow_pickle=False) as archive:
        required = {"genes", "edge_index", "candidate_mask", "frequency_penalty"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"graph resource is missing arrays: {sorted(missing)}")
        genes = archive["genes"].astype(str)
        edge_index = np.asarray(archive["edge_index"], dtype=np.int64)
        candidate_mask = np.asarray(archive["candidate_mask"], dtype=bool)
        frequency_penalty = np.asarray(archive["frequency_penalty"], dtype=np.float64)
    if len(genes) != len(set(genes.tolist())):
        raise ValueError("graph gene universe contains duplicates")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("graph edge_index must have shape [2, E]")
    if candidate_mask.shape != (len(genes),) or frequency_penalty.shape != (len(genes),):
        raise ValueError("graph arrays do not share the gene-universe length")
    if not np.isfinite(frequency_penalty).all() or np.any(frequency_penalty < 0):
        raise ValueError("frequency_penalty must be finite and non-negative")
    return _GraphResources(genes, edge_index, candidate_mask, frequency_penalty)


def _read_inference_bundle(path: Path) -> _InferenceBundle:
    with np.load(path, allow_pickle=False) as archive:
        required = {"cell_ids", "features", "feature_names", "control_mask", "stemness_context_ids"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"inference bundle is missing arrays: {sorted(missing)}")
        cell_ids = archive["cell_ids"].astype(str)
        features = np.asarray(archive["features"], dtype=np.float32)
        feature_names = archive["feature_names"].astype(str)
        control_mask = np.asarray(archive["control_mask"], dtype=bool)
        stemness_ids = archive["stemness_context_ids"].astype(str)
    if features.ndim != 2 or features.shape[0] != len(cell_ids):
        raise ValueError("inference features and cell IDs have inconsistent shapes")
    if features.shape[1] != len(feature_names) or features.shape[1] != EXPECTED_FEATURE_DIM:
        raise ValueError("inference feature names do not match the 34-dimensional model input")
    if len(set(cell_ids.tolist())) != len(cell_ids):
        raise ValueError("inference bundle contains duplicate cell IDs")
    if control_mask.shape != (len(cell_ids),) or not control_mask.any():
        raise ValueError("inference bundle has an invalid control mask")
    if not np.isfinite(features).all():
        raise ValueError("inference features contain non-finite values")
    return _InferenceBundle(cell_ids, features, feature_names, control_mask, stemness_ids)


def _validate_manifest(paths: UnseenResourcePaths, model: PretrainedUnseenModel) -> None:
    manifest_path = paths.manifest
    if manifest_path is None:
        return
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read resource manifest: {manifest_path}") from exc
    expected_hash = str(manifest.get("checkpoint_sha256", "")).lower()
    if expected_hash and expected_hash != model.checkpoint_sha256.lower():
        raise ValueError("resource manifest checkpoint checksum does not match the loaded checkpoint")
    for key, expected in {
        "graph_nodes": EXPECTED_GRAPH_NODES,
        "graph_edges": EXPECTED_GRAPH_EDGES,
        "official_unseen_candidates": EXPECTED_CANDIDATES,
    }.items():
        if key in manifest and int(manifest[key]) != expected:
            raise ValueError(f"resource manifest has unexpected {key}: {manifest[key]}")


def _validate_inference_manifest(manifest_path: Path | None, bundle_path: Path) -> None:
    if manifest_path is None:
        return
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read resource manifest: {manifest_path}") from exc
    expected_hash = str(manifest.get("inference_bundle_sha256", "")).lower()
    if expected_hash and expected_hash != _sha256(bundle_path).lower():
        raise ValueError("inference bundle SHA-256 does not match the resource manifest")
    for key, expected in {
        "n_cells": 147_887,
        "n_control_cells": 8_149,
        "n_features": EXPECTED_FEATURE_DIM,
        "stemness_context_cells": 4_048,
    }.items():
        if key in manifest and int(manifest[key]) != expected:
            raise ValueError(f"resource manifest has unexpected {key}: {manifest[key]}")


def _torch_load(path: Path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # torch < 2.0
        return torch.load(path, map_location=device)


def _read_tf_genes(path: Path) -> frozenset[str]:
    genes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip().split()[0] if line.strip() else ""
        if value and not value.startswith("#"):
            genes.append(value.upper())
    # The archived text file has 1,639 newline-delimited entries (the historical
    # release metadata counted 1,638 lines because the final line has no newline).
    if len(set(genes)) not in {1_638, 1_639}:
        raise ValueError(f"unexpected formal TF list size: {len(set(genes))}")
    return frozenset(genes)


def _validate_features(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != EXPECTED_FEATURE_DIM:
        raise ValueError(f"{name} must have shape [cells, {EXPECTED_FEATURE_DIM}]")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return np.ascontiguousarray(array)


def _normalize_genes(genes: Sequence[str] | None) -> tuple[str, ...]:
    if genes is None:
        return ()
    values = tuple(dict.fromkeys(str(gene).strip() for gene in genes if str(gene).strip()))
    if any("|" in gene for gene in values):
        raise ValueError("gene names must be provided as individual strings")
    return values


def _matches_formal_stemness(
    positive: tuple[str, ...],
    negative: tuple[str, ...],
    extreme_fraction: float,
) -> bool:
    return (
        frozenset(positive) == FORMAL_STEMNESS_POSITIVE_GENES
        and frozenset(negative) == FORMAL_STEMNESS_NEGATIVE_GENES
        and extreme_fraction == FORMAL_STEMNESS_EXTREME_FRACTION
    )


def _validate_n_genes(n_genes: int | None) -> None:
    if n_genes is not None and (isinstance(n_genes, bool) or n_genes not in {100, 1000}):
        raise ValueError("n_genes must be 100, 1000, or None")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _filename_component(value: str) -> str:
    component = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip()).strip("_.-")
    if not component:
        raise ValueError("phenotype_name must contain a filename-safe character")
    return component

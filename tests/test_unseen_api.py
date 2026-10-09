from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from polarist import rank_unseen_drivers
from polarist.models import PPIGCN, FusionCellSetClassifier
from polarist.unseen import (
    PACKAGE_RESOURCE_DIR,
    UnseenResourcePaths,
    _validate_inference_manifest,
)


@dataclass
class FakeBundle:
    cell_ids: np.ndarray
    features: np.ndarray
    feature_names: np.ndarray
    control_mask: np.ndarray
    stemness_context_ids: np.ndarray

    @property
    def control_features(self) -> np.ndarray:
        return self.features[self.control_mask]


class FakeModel:
    genes = ("GENE_A", "GENE_B", "GENE_C", "GENE_D")
    candidate_mask = np.array([True, True, True, False])
    frequency_penalty = np.array([0.0, 0.5, 0.25, 0.0])
    transcription_factors = frozenset({"GENE_A", "GENE_C"})
    perturbation_type = "Gain"
    paths = SimpleNamespace()

    def __init__(self) -> None:
        control_ids = np.asarray([f"ctrl_{i}" for i in range(8149)])
        context_ids = np.asarray([f"stem_{i}" for i in range(4048)])
        self._bundle = FakeBundle(
            cell_ids=np.concatenate([control_ids, context_ids]),
            features=np.zeros((8149 + 4048, 34), dtype=np.float32),
            feature_names=np.asarray([f"f{i}" for i in range(34)]),
            control_mask=np.concatenate(
                [np.ones(8149, dtype=bool), np.zeros(4048, dtype=bool)]
            ),
            stemness_context_ids=context_ids[::-1],
        )

    def load_inference_bundle(self) -> FakeBundle:
        return self._bundle

    def score(self, control_features: np.ndarray, context_features: np.ndarray) -> np.ndarray:
        assert control_features.shape == (8149, 34)
        assert context_features.ndim == 2
        assert context_features.shape[1] == 34
        return np.asarray([1.0, 2.0, 1.5, -1.0])


def test_frozen_context_ranking_uses_candidate_denominator_and_hub2(tmp_path):
    result = rank_unseen_drivers(
        reference_context="stemness",
        phenotype_name="stemness",
        n_genes=100,
        tf_only=True,
        model=FakeModel(),
        output_dir=tmp_path,
        verbose=False,
    )

    assert result.candidate_count == 3
    assert result.context_cell_count == 4048
    assert result.undesired_cell_count == 0
    assert result.full_ranking.columns.tolist() == [
        "rank",
        "gene",
        "score",
        "raw_score",
        "hub_penalty",
        "is_transcription_factor",
    ]
    assert result.full_ranking["gene"].tolist() == ["GENE_A", "GENE_B", "GENE_C"]
    assert result.full_ranking["score"].tolist() == pytest.approx([1.0, 1.0, 1.0])
    assert result.tf_ranking is not None
    assert result.tf_ranking.columns.tolist()[0:2] == ["tf_rank", "rank"]
    assert result.tf_ranking["gene"].tolist() == ["GENE_A", "GENE_C"]
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "stemness_unseen_ranking.csv",
        "stemness_unseen_tf_ranking.csv",
    ]


def test_unseen_input_modes_are_mutually_exclusive():
    with pytest.raises(ValueError, match="exactly one input mode"):
        rank_unseen_drivers(
            desired_cells=["stem_0"],
            reference_context="stemness",
            model=FakeModel(),
            verbose=False,
        )
    with pytest.raises(ValueError, match="exactly one input mode"):
        rank_unseen_drivers(
            desired_cells=["stem_0"],
            positive_genes=["TCF7"],
            model=FakeModel(),
            verbose=False,
        )


def test_signature_mode_defines_phenotype_and_uses_only_top_cells(monkeypatch):
    phenotype_cells = pd.DataFrame(
        {
            "Unique_cellid": ["stem_0", "stem_1", "stem_2"],
            "phenotype_group": ["top", "bottom", "top"],
        }
    )

    def fake_define_phenotype_cells(**kwargs):
        assert kwargs["positive_genes"] == ("TCF7",)
        assert kwargs["negative_genes"] == ("TOX",)
        assert kwargs["extreme_fraction"] == 0.1
        return phenotype_cells

    monkeypatch.setattr(
        "polarist.workflow.define_phenotype_cells",
        fake_define_phenotype_cells,
    )
    result = rank_unseen_drivers(
        positive_genes=["TCF7"],
        negative_genes=["TOX"],
        extreme_fraction=0.1,
        n_genes=100,
        model=FakeModel(),
        verbose=False,
    )

    assert result.context_mode == "gene_signature"
    assert result.desired_cell_count == 2
    assert result.undesired_cell_count == 0
    assert result.diagnostics["phenotype_cells"].equals(phenotype_cells)


def test_formal_stemness_signature_reuses_bundled_context():
    result = rank_unseen_drivers(
        positive_genes=[
            "TCF7", "LEF1", "SLAMF6", "SELL", "BCL2",
            "BCL6", "CXCR5", "CCNE1", "CCNE2",
        ],
        negative_genes=["TOX", "HAVCR2", "ENTPD1", "CD101", "CD244"],
        phenotype_name="stemness",
        n_genes=100,
        model=FakeModel(),
        verbose=False,
    )

    assert result.context_mode == "bundled_reference"
    assert result.context_cell_count == 4048
    assert result.desired_cell_count == 4048
    assert result.undesired_cell_count == 0
    assert "phenotype_cells" not in result.diagnostics


def test_user_defined_desired_cells_are_used_without_undesired_cells(tmp_path):
    desired_cells = pd.DataFrame(
        {"Unique_cellid": ["stem_3", "stem_1", "stem_2"]}
    )

    result = rank_unseen_drivers(
        desired_cells=desired_cells,
        phenotype_name="custom_state",
        n_genes=100,
        model=FakeModel(),
        output_dir=tmp_path,
        verbose=False,
    )

    assert result.context_mode == "desired_cells"
    assert result.context_cell_count == 3
    assert result.desired_cell_count == 3
    assert result.undesired_cell_count == 0
    assert (tmp_path / "custom_state_unseen_ranking.csv").is_file()


def test_desired_cells_can_be_read_from_csv(tmp_path):
    path = tmp_path / "desired_cells.csv"
    pd.DataFrame({"Unique_cellid": ["stem_0", "stem_1"]}).to_csv(path, index=False)

    result = rank_unseen_drivers(
        desired_cells=path,
        n_genes=100,
        model=FakeModel(),
        verbose=False,
    )

    assert result.desired_cell_count == 2


def test_desired_cells_reject_bottom_or_other_metadata_without_cell_ids():
    with pytest.raises(KeyError, match="Unique_cellid"):
        rank_unseen_drivers(
            desired_cells=pd.DataFrame({"phenotype_group": ["top"]}),
            model=FakeModel(),
            verbose=False,
        )

    with pytest.raises(ValueError, match="only phenotype_group == 'top'"):
        rank_unseen_drivers(
            desired_cells=pd.DataFrame(
                {
                    "Unique_cellid": ["stem_0", "stem_1"],
                    "phenotype_group": ["top", "bottom"],
                }
            ),
            model=FakeModel(),
            verbose=False,
        )


def test_mode_and_n_genes_validation():
    with pytest.raises(ValueError, match="exactly one input mode"):
        rank_unseen_drivers(model=FakeModel(), verbose=False)
    with pytest.raises(ValueError, match="n_genes"):
        rank_unseen_drivers(reference_context="stemness", n_genes=50, model=FakeModel(), verbose=False)
    with pytest.raises(ValueError, match="only bundled"):
        rank_unseen_drivers(reference_context="aging", model=FakeModel(), verbose=False)


@pytest.mark.parametrize("n_genes", [100, 1000, None])
def test_supported_n_genes_values(n_genes):
    result = rank_unseen_drivers(
        reference_context="stemness",
        n_genes=n_genes,
        model=FakeModel(),
        verbose=False,
    )
    assert len(result.full_ranking) == 3


def test_synthetic_ppigcn_and_fusion_forward():
    import torch

    torch.manual_seed(7)
    initial = torch.randn(5, 8)
    gcn = PPIGCN(initial, hidden_dim=4)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    embeddings = gcn(edge_index)
    assert embeddings.shape == (5, 8)

    classifier = FusionCellSetClassifier(
        d_in=3,
        n_classes=5,
        emb_dim=8,
        initial_gene_embeddings=initial,
    ).eval()
    x_ctrl = torch.randn(1, 4, 3)
    x_pert = torch.randn(1, 5, 3)
    with torch.no_grad():
        scores = classifier(x_ctrl, x_pert, embeddings)
    assert scores.shape == (1, 5)
    assert torch.isfinite(scores).all()

    incomplete_state = classifier.state_dict()
    incomplete_state.pop("encoder.net.0.weight")
    with pytest.raises(RuntimeError):
        classifier.load_state_dict(incomplete_state, strict=True)


def test_inference_manifest_checksum_error(tmp_path):
    bundle = tmp_path / "inference_features.npz"
    bundle.write_bytes(b"not-the-declared-bundle")
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"inference_bundle_sha256": "' + "0" * 64 + '"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        _validate_inference_manifest(manifest, bundle)


def test_missing_inference_resource_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(UnseenResourcePaths, "roots", property(lambda self: (tmp_path,)))
    with pytest.raises(FileNotFoundError, match="inference feature bundle"):
        _ = UnseenResourcePaths(tmp_path).inference_bundle


def test_bundled_inference_resource_is_the_zero_config_default(tmp_path, monkeypatch):
    monkeypatch.delenv("POLARIST_UNSEEN_RESOURCE_DIR", raising=False)
    monkeypatch.delenv("POLARIST_RESOURCE_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

    paths = UnseenResourcePaths()
    expected_bundle = PACKAGE_RESOURCE_DIR / "inference" / "inference_features.npz"
    expected_manifest = PACKAGE_RESOURCE_DIR / "inference" / "manifest.json"
    assert paths.inference_bundle == expected_bundle
    assert paths.manifest == expected_manifest
    _validate_inference_manifest(paths.manifest, paths.inference_bundle)

    with np.load(paths.inference_bundle, allow_pickle=False) as archive:
        assert archive["features"].shape == (147_887, 34)
        assert int(archive["control_mask"].sum()) == 8_149
        assert len(archive["stemness_context_ids"]) == 4_048


def test_external_bundle_does_not_borrow_the_packaged_manifest(tmp_path):
    external_bundle = tmp_path / "inference_features.npz"
    external_bundle.write_bytes(b"external-bundle-placeholder")
    paths = UnseenResourcePaths(tmp_path)
    assert paths.inference_bundle == external_bundle
    assert paths.manifest is None

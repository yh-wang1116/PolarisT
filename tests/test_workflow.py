from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from polarist.workflow import BUNDLED_RESOURCE_DIR, SeenRankingWorkflow, SeenResourcePaths


def _workflow() -> SeenRankingWorkflow:
    cell_ids = ["c1", "c2", "c3", "c4", "c5", "c6"]
    adata = SimpleNamespace(
        layers={
            "logNor": np.asarray(
                [
                    [10.0, 0.0],
                    [0.0, 10.0],
                    [5.0, 5.0],
                    [4.0, 4.0],
                    [3.0, 3.0],
                    [2.0, 2.0],
                ]
            )
        },
        obs=pd.DataFrame(
            {
                "Unique_cellid": cell_ids,
                "Dataset": ["D1"] * 6,
                "Immune_type": ["CD8T"] * 6,
            }
        ),
        var_names=pd.Index(["POS", "NEG"]),
    )
    cell_embeddings = pd.DataFrame(
        [
            [1.0, 0.0, -1.0],
            [-1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        index=cell_ids,
        columns=["d1", "d2", "d3"],
    )
    perturbation_embeddings = pd.DataFrame(
        [
            [2.0, 0.0, -2.0],
            [-2.0, 0.0, 2.0],
            [1.0, 0.0, -1.0],
        ],
        index=["TF1_Gain", "GENE2_Loss", "TF10_Gain"],
        columns=["x1", "x2", "x3"],
    )
    groups = pd.Series({"TF1_Gain": "A", "GENE2_Loss": "B", "TF10_Gain": "C"})
    return SeenRankingWorkflow(
        adata=adata,
        cell_order=cell_ids,
        cell_embeddings=cell_embeddings,
        perturbation_embeddings=perturbation_embeddings,
        perturbation_groups=groups,
        transcription_factors=["TF1"],
    )


def test_workflow_ranks_positive_high_negative_low_and_filters_tfs():
    result = _workflow().rank(
        positive_genes=["POS"],
        negative_genes=["NEG"],
        phenotype_name="test_state",
        tf_only=True,
    )

    assert result.phenotype_cells.set_index("Unique_cellid")["phenotype_group"].to_dict() == {
        "c1": "top",
        "c2": "bottom",
    }
    np.testing.assert_allclose(result.phenotype_vector.to_numpy(), [2.0, 0.0, -2.0])
    assert result.full_ranking.index[0] == "TF1_Gain"
    assert result.ranking.index.tolist() == ["TF1_Gain"]
    assert result.full_ranking.loc["TF1_Gain", "gene"] == "TF1"
    assert result.full_ranking.loc["TF10_Gain", "gene"] == "TF10"
    assert "cluster" not in result.full_ranking.columns
    assert "cluster" not in result.tf_ranking.columns
    assert result.extreme_fraction == 0.05
    assert result.refinement_weight == 0.9


def test_phenotype_definition_can_be_used_without_running_seen_ranking(capsys):
    cells = _workflow().define_phenotype_cells(
        positive_genes=["POS"],
        negative_genes=["NEG"],
    )

    assert cells.set_index("Unique_cellid")["phenotype_group"].to_dict() == {
        "c1": "top",
        "c2": "bottom",
    }
    assert "1 desired (top) and 1 undesired (bottom) cells" in capsys.readouterr().out


def test_workflow_saves_requested_tf_ranking_and_audit_outputs(tmp_path):
    result = _workflow().rank(
        positive_genes=["POS"],
        negative_genes=["NEG"],
        phenotype_name="test state",
        tf_only=True,
    )

    paths = result.save(tmp_path)
    saved_ranking = pd.read_csv(paths["ranking"], index_col=0)

    assert paths["ranking"].name == "test_state_seen_ranking.csv"
    assert saved_ranking.index.tolist() == ["TF1_Gain"]
    assert paths["phenotype_cells"].is_file()
    assert paths["phenotype_vector"].is_file()
    assert paths["signature_stats"].is_file()


def test_workflow_supports_positive_genes_without_negative_genes():
    result = _workflow().rank(
        positive_genes=["POS"],
        phenotype_name="positive_only",
    )

    assert result.phenotype_cells.set_index("Unique_cellid")["phenotype_group"].to_dict() == {
        "c1": "top",
        "c2": "bottom",
    }
    np.testing.assert_allclose(result.phenotype_vector.to_numpy(), [2.0, 0.0, -2.0])
    assert "positive_score" in result.phenotype_cells.columns
    assert "negative_score" not in result.phenotype_cells.columns
    assert result.negative_genes == ()


def test_workflow_requires_at_least_one_gene_signature():
    with pytest.raises(ValueError, match="at least one"):
        _workflow().rank(positive_genes=[], negative_genes=[])


def test_workflow_accepts_custom_fraction_and_refinement_weight():
    result = _workflow().rank(
        positive_genes=["POS"],
        extreme_fraction=0.2,
        refinement_weight=0.4,
    )

    assert result.extreme_fraction == 0.2
    assert result.refinement_weight == 0.4


@pytest.mark.parametrize(
    ("parameter", "value", "message"),
    [
        ("extreme_fraction", 0.6, "extreme_fraction"),
        ("refinement_weight", 1.1, "refinement_weight"),
    ],
)
def test_workflow_validates_configurable_parameters(parameter, value, message):
    with pytest.raises(ValueError, match=message):
        _workflow().rank(positive_genes=["POS"], **{parameter: value})


def test_workflow_records_perturbations_without_refinement_groups():
    workflow = _workflow()
    workflow._perturbation_groups = workflow.perturbation_groups.drop("TF10_Gain")

    result = workflow.rank(positive_genes=["POS"])

    assert "TF10_Gain" in result.raw_ranking.index
    assert "TF10_Gain" not in result.full_ranking.index
    assert result.diagnostics["unrefined_perturbations"]["perturbation"].tolist() == ["TF10_Gain"]


def test_workflow_reports_stage_progress(capsys):
    _workflow().rank(
        positive_genes=["POS"],
        negative_genes=["NEG"],
        extreme_fraction=0.2,
        tf_only=True,
    )

    output = capsys.readouterr().out
    assert "[1/5]" in output
    assert "top 20.0% and bottom 20.0% cells within GEX" in output
    assert "[2/5] Phenotype cells selected; phenotype direction vector computed." in output
    assert "[3/5] Initial perturbation-phenotype alignment completed." in output
    assert "[4/5] Final perturbation ranking completed." in output
    assert "[5/5] Transcription-factor ranking completed (1 TFs)." in output


def test_workflow_can_disable_stage_progress(capsys):
    _workflow().rank(positive_genes=["POS"], verbose=False)

    assert capsys.readouterr().out == ""


def test_resource_paths_use_bundled_small_resources(tmp_path):
    paths = SeenResourcePaths(tmp_path)

    assert paths.adata == tmp_path / "Anndata_cd8_raw.h5ad"
    assert paths.cell_embeddings == BUNDLED_RESOURCE_DIR / "Seen_cd8_cell_embedding.npz"
    assert paths.cell_order == BUNDLED_RESOURCE_DIR / "Seen_cell_order.csv"
    assert paths.perturbation_embeddings == (BUNDLED_RESOURCE_DIR / "Seen_perturbation_vector.csv")
    assert paths.perturbation_groups == (BUNDLED_RESOURCE_DIR / "Seen_perturbation_groups.csv")
    assert paths.transcription_factors == BUNDLED_RESOURCE_DIR / "Human_tf_list.txt"


def test_workflow_loads_npz_cell_embeddings(tmp_path):
    path = tmp_path / "Seen_cd8_cell_embedding.npz"
    np.savez_compressed(
        path,
        cell_ids=np.asarray(["c1", "c2"]),
        columns=np.asarray(["d1", "d2"]),
        embeddings=np.asarray([[1.0, 2.0], [3.0, 4.0]]),
    )

    loaded = SeenRankingWorkflow(resource_dir=tmp_path).cell_embeddings

    assert loaded.index.tolist() == ["c1", "c2"]
    assert loaded.columns.tolist() == ["d1", "d2"]
    np.testing.assert_allclose(loaded.to_numpy(), [[1.0, 2.0], [3.0, 4.0]])

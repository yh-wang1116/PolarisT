import numpy as np
import pandas as pd

from polarist.seen import rank_seen, rank_seen_perturbations, refine_seen_ranking


def test_rank_seen_perturbations_orders_by_pearson():
    vector = pd.Series([1.0, 0.0, -1.0], index=["d1", "d2", "d3"])
    perturbations = pd.DataFrame(
        {
            "d1": [2.0, 1.0, -1.0],
            "d2": [0.0, 1.0, 0.0],
            "d3": [-2.0, -1.0, 1.0],
        },
        index=["P1_gain", "P2_gain", "P3_loss"],
    )

    result = rank_seen_perturbations(perturbations, vector)

    assert result.index.tolist() == ["P1_gain", "P2_gain", "P3_loss"]
    np.testing.assert_allclose(result.loc["P1_gain", "pearson_correlation"], 1.0)
    np.testing.assert_allclose(result.loc["P3_loss", "pearson_correlation"], -1.0)
    assert result.loc["P1_gain", "dot_product"] == 4.0
    assert result.loc["P3_loss", "perturbation_direction"] == "loss"


def test_refinement_uses_cluster_mean_and_configurable_weight():
    raw = pd.DataFrame(
        {"pearson_correlation": [1.0, 0.2, -0.5]},
        index=["P1_gain", "P2_gain", "P3_loss"],
    )
    groups = pd.Series(
        {"P1_gain": "A", "P2_gain": "A", "P3_loss": "B"},
        name="cluster",
    )

    result = refine_seen_ranking(raw, groups, weight=0.8)

    np.testing.assert_allclose(result.loc["P1_gain", "cluster_mean_score"], 0.6)
    np.testing.assert_allclose(result.loc["P1_gain", "refined_score"], 0.92)
    np.testing.assert_allclose(result.loc["P2_gain", "refined_score"], 0.28)


def test_rank_seen_returns_auditable_intermediate_results():
    phases = pd.DataFrame(
        {
            "effector": ["top", "mid", "bottom"],
            "terminal_bad": ["bottom", "mid", "top"],
        },
        index=["c1", "c2", "c3"],
    )
    cell_embeddings = pd.DataFrame(
        [[2.0, 0.0, -2.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 1.0]],
        index=phases.index,
        columns=["d1", "d2", "d3"],
    )
    perturbation_embeddings = pd.DataFrame(
        [[2.0, 0.0, -2.0], [-1.0, 0.0, 1.0]],
        index=["P1_gain", "P2_loss"],
        columns=cell_embeddings.columns,
    )
    groups = pd.Series({"P1_gain": "A", "P2_loss": "B"})

    result = rank_seen(
        phases,
        cell_embeddings,
        perturbation_embeddings,
        desired_conditions={"effector": "top", "terminal_bad": "bottom"},
        undesired_conditions={"effector": "bottom", "terminal_bad": "top"},
        cluster_groups=groups,
    )

    np.testing.assert_allclose(result.phenotype_vector.vector, [3.0, 0.0, -3.0])
    assert result.raw_ranking.index[0] == "P1_gain"
    assert result.refined_ranking is not None

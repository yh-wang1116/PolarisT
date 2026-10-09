import numpy as np
import pandas as pd

from polarist.phenotype import (
    assign_combined_phenotype,
    compute_phenotype_vector,
    score_gene_sets,
)


def test_score_gene_sets_selects_extremes_within_each_dataset():
    cell_ids = [f"c{i}" for i in range(8)]
    gene_names = ["EFF", "BAD"]
    expression = np.asarray(
        [
            [1, 8],
            [2, 7],
            [3, 6],
            [9, 1],
            [10, 1],
            [8, 2],
            [7, 3],
            [1, 9],
        ],
        dtype=float,
    )
    metadata = pd.DataFrame(
        {
            "Unique_cellid": cell_ids,
            "Dataset": ["A"] * 4 + ["B"] * 4,
            "Immune_type": ["CD8T"] * 8,
        }
    )

    result = score_gene_sets(
        expression,
        cell_ids,
        gene_names,
        metadata,
        {"effector": ["EFF"], "terminal_bad": ["BAD"]},
        top_fraction=0.25,
        bottom_fraction=0.25,
    )

    assert result.phases.loc["c3", "effector"] == "top"
    assert result.phases.loc["c0", "effector"] == "bottom"
    assert result.phases.loc["c4", "effector"] == "top"
    assert result.phases.loc["c7", "effector"] == "bottom"
    assert result.phases.loc["c0", "terminal_bad"] == "top"
    assert result.phases.loc["c3", "terminal_bad"] == "bottom"
    assert result.phases.loc["c7", "terminal_bad"] == "top"
    assert result.phases.loc["c4", "terminal_bad"] == "bottom"


def test_combined_phenotype_and_vector_use_explicit_top_minus_bottom():
    phases = pd.DataFrame(
        {
            "effector": ["top", "mid", "mid", "bottom"],
            "terminal_bad": ["bottom", "mid", "mid", "top"],
        },
        index=["c1", "c2", "c3", "c4"],
    )
    labels = assign_combined_phenotype(
        phases,
        desired_conditions={"effector": "top", "terminal_bad": "bottom"},
        undesired_conditions={"effector": "bottom", "terminal_bad": "top"},
    )
    embeddings = pd.DataFrame(
        [[3.0, 2.0], [0.0, 0.0], [1.0, 1.0], [1.0, 4.0]],
        index=phases.index,
        columns=["dim1", "dim2"],
    )
    result = compute_phenotype_vector(embeddings, labels)

    assert labels.to_dict() == {"c1": "top", "c2": "mid", "c3": "mid", "c4": "bottom"}
    np.testing.assert_allclose(result.vector.to_numpy(), [2.0, -2.0])
    assert result.n_desired == 1
    assert result.n_undesired == 1


def test_score_gene_sets_uses_fixed_cell_order_to_break_ties():
    cell_ids = ["c1", "c2", "c3", "c4"]
    metadata = pd.DataFrame(
        {
            "Unique_cellid": cell_ids,
            "Dataset": ["A"] * 4,
            "Immune_type": ["CD8T"] * 4,
        }
    )

    result = score_gene_sets(
        expression=np.asarray([[0.0], [0.0], [2.0], [1.0]]),
        cell_ids=cell_ids,
        gene_names=["SIGNATURE"],
        metadata=metadata,
        gene_sets={"positive": ["SIGNATURE"]},
        top_fraction=0.25,
        bottom_fraction=0.25,
        cell_order=["c2", "c1", "c3", "c4"],
    )

    assert result.phases.loc["c3", "positive"] == "top"
    assert result.phases.loc["c1", "positive"] == "bottom"


def test_score_gene_sets_reports_requested_and_missing_genes():
    result = score_gene_sets(
        expression=np.asarray([[0.0], [1.0], [2.0], [3.0]]),
        cell_ids=["c1", "c2", "c3", "c4"],
        gene_names=["PRESENT"],
        metadata=pd.DataFrame(
            {
                "Unique_cellid": ["c1", "c2", "c3", "c4"],
                "Dataset": ["A"] * 4,
                "Immune_type": ["CD8T"] * 4,
            }
        ),
        gene_sets={"positive": ["PRESENT", "MISSING"]},
        top_fraction=0.25,
        bottom_fraction=0.25,
    )

    coverage = result.gene_coverage.iloc[0]
    assert coverage["n_requested"] == 2
    assert coverage["n_present"] == 1
    assert coverage["coverage"] == 0.5
    assert coverage["missing_genes"] == "MISSING"

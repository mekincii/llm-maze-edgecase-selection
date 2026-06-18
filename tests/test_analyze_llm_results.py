from pathlib import Path

import pandas as pd
import pytest

from src.experiments.analyze_llm_results import (
    compute_class_prediction_counts,
    compute_group_summary,
    compute_overall_summary,
    compute_recommendation_counts,
    load_llm_evaluation_results,
    save_summary_tables,
)


def make_evaluation_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model_name": "qwen3:1.7b",
                "true_maze_family": "OPEN",
                "representation_mode": "features",
                "predicted_edge_case_class": "UNKNOWN",
                "recommended_solver": "BFS",
                "classification_correct": False,
                "empirical_solver_selection_correct": False,
                "guarantee_aware_solver_selection_correct": False,
                "selected_solver_shortest_path": True,
                "quality_failure": False,
                "guarantee_aware_policy_violation": False,
                "empirical_expansion_regret": 196,
                "guarantee_aware_expansion_delta": 196,
            },
            {
                "model_name": "qwen3:8b",
                "true_maze_family": "OPEN",
                "representation_mode": "features",
                "predicted_edge_case_class": "OPEN",
                "recommended_solver": "A*",
                "classification_correct": True,
                "empirical_solver_selection_correct": True,
                "guarantee_aware_solver_selection_correct": True,
                "selected_solver_shortest_path": True,
                "quality_failure": False,
                "guarantee_aware_policy_violation": False,
                "empirical_expansion_regret": 0,
                "guarantee_aware_expansion_delta": 0,
            },
            {
                "model_name": "qwen3:8b",
                "true_maze_family": "ASTAR_TRAP",
                "representation_mode": "ascii",
                "predicted_edge_case_class": "OPEN",
                "recommended_solver": "A*",
                "classification_correct": False,
                "empirical_solver_selection_correct": False,
                "guarantee_aware_solver_selection_correct": True,
                "selected_solver_shortest_path": True,
                "quality_failure": False,
                "guarantee_aware_policy_violation": False,
                "empirical_expansion_regret": 54,
                "guarantee_aware_expansion_delta": 0,
            },
        ]
    )


def test_compute_overall_summary() -> None:
    evaluation_df = make_evaluation_df()

    summary = compute_overall_summary(evaluation_df)

    assert len(summary) == 1
    assert summary.iloc[0]["n"] == 3
    assert summary.iloc[0]["classification_accuracy"] == pytest.approx(1 / 3)
    assert summary.iloc[0]["shortest_path_rate"] == 1.0
    assert summary.iloc[0]["quality_failure_rate"] == 0.0


def test_compute_group_summary_by_model() -> None:
    evaluation_df = make_evaluation_df()

    summary = compute_group_summary(evaluation_df, ["model_name"])

    assert set(summary["model_name"]) == {"qwen3:1.7b", "qwen3:8b"}

    qwen8 = summary[summary["model_name"] == "qwen3:8b"].iloc[0]

    assert qwen8["n"] == 2
    assert qwen8["guarantee_aware_solver_selection_accuracy"] == 1.0


def test_compute_group_summary_rejects_empty_df() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        compute_group_summary(pd.DataFrame(), ["model_name"])


def test_compute_group_summary_rejects_missing_group_column() -> None:
    evaluation_df = make_evaluation_df()

    with pytest.raises(ValueError, match="Missing group columns"):
        compute_group_summary(evaluation_df, ["missing_column"])


def test_compute_recommendation_counts() -> None:
    evaluation_df = make_evaluation_df()

    counts = compute_recommendation_counts(evaluation_df)

    qwen8_counts = counts[counts["model_name"] == "qwen3:8b"]

    assert set(qwen8_counts["recommended_solver"]) == {"A*"}


def test_compute_class_prediction_counts() -> None:
    evaluation_df = make_evaluation_df()

    counts = compute_class_prediction_counts(evaluation_df)

    assert "predicted_edge_case_class" in counts.columns
    assert counts["count"].sum() == 3


def test_save_summary_tables(tmp_path: Path) -> None:
    evaluation_df = make_evaluation_df()

    output_paths = save_summary_tables(evaluation_df, tmp_path)

    assert "overall" in output_paths
    assert "by_model" in output_paths
    assert output_paths["overall"].exists()
    assert output_paths["by_model"].exists()


def test_load_llm_evaluation_results_rejects_missing_columns(tmp_path: Path) -> None:
    bad_path = tmp_path / "bad_eval.csv"

    pd.DataFrame(
        [
            {
                "model_name": "qwen3:1.7b",
                "true_maze_family": "OPEN",
            }
        ]
    ).to_csv(bad_path, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        load_llm_evaluation_results(bad_path)
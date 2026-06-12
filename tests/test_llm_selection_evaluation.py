import pandas as pd
import pytest

from src.experiments.evaluate_llm_selection import (
    evaluate_llm_response_rows,
    evaluate_single_llm_response,
    get_oracle_solvers,
    get_solver_row,
    summarize_llm_evaluation,
)


def make_test_benchmark_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "maze_family": "GREEDY_TRAP",
                "solver_name": "A*",
                "success": True,
                "path_length": 17,
                "shortest_path_length": 17,
                "expanded_nodes": 29,
                "is_shortest_path": True,
                "is_best_optimal_expansion": True,
            },
            {
                "maze_family": "GREEDY_TRAP",
                "solver_name": "Greedy Best-First",
                "success": True,
                "path_length": 17,
                "shortest_path_length": 17,
                "expanded_nodes": 46,
                "is_shortest_path": True,
                "is_best_optimal_expansion": False,
            },
            {
                "maze_family": "GREEDY_TRAP",
                "solver_name": "DFS",
                "success": True,
                "path_length": 25,
                "shortest_path_length": 17,
                "expanded_nodes": 20,
                "is_shortest_path": False,
                "is_best_optimal_expansion": False,
            },
        ]
    )


def test_get_oracle_solvers() -> None:
    benchmark_df = make_test_benchmark_df()

    oracle_solvers = get_oracle_solvers(
        benchmark_df=benchmark_df,
        maze_family="GREEDY_TRAP",
    )

    assert oracle_solvers == {"A*"}


def test_get_oracle_solvers_rejects_unknown_family() -> None:
    benchmark_df = make_test_benchmark_df()

    with pytest.raises(ValueError, match="No benchmark rows"):
        get_oracle_solvers(
            benchmark_df=benchmark_df,
            maze_family="UNKNOWN_FAMILY",
        )


def test_get_solver_row() -> None:
    benchmark_df = make_test_benchmark_df()

    row = get_solver_row(
        benchmark_df=benchmark_df,
        maze_family="GREEDY_TRAP",
        solver_name="A*",
    )

    assert row["expanded_nodes"] == 29


def test_evaluate_single_correct_oracle_selection() -> None:
    benchmark_df = make_test_benchmark_df()

    raw_response = """
    {
      "edge_case_class": "GREEDY_TRAP",
      "recommended_solver": "A*",
      "confidence": 0.9,
      "reason": "A* is reliable and efficient."
    }
    """

    result = evaluate_single_llm_response(
        benchmark_df=benchmark_df,
        true_maze_family="GREEDY_TRAP",
        representation_mode="features_ascii",
        raw_response=raw_response,
        model_name="mock",
    )

    assert result["classification_correct"] is True
    assert result["solver_selection_correct"] is True
    assert result["selected_solver_shortest_path"] is True
    assert result["quality_failure"] is False
    assert result["expansion_regret"] == 0


def test_evaluate_single_shortest_but_not_oracle_selection() -> None:
    benchmark_df = make_test_benchmark_df()

    raw_response = """
    {
      "edge_case_class": "GREEDY_TRAP",
      "recommended_solver": "Greedy Best-First",
      "confidence": 0.7,
      "reason": "Greedy is heuristic-directed."
    }
    """

    result = evaluate_single_llm_response(
        benchmark_df=benchmark_df,
        true_maze_family="GREEDY_TRAP",
        representation_mode="features_ascii",
        raw_response=raw_response,
        model_name="mock",
    )

    assert result["classification_correct"] is True
    assert result["solver_selection_correct"] is False
    assert result["selected_solver_shortest_path"] is True
    assert result["quality_failure"] is False
    assert result["expansion_regret"] == 17


def test_evaluate_single_quality_failure() -> None:
    benchmark_df = make_test_benchmark_df()

    raw_response = """
    {
      "edge_case_class": "GREEDY_TRAP",
      "recommended_solver": "DFS",
      "confidence": 0.4,
      "reason": "DFS may be fast."
    }
    """

    result = evaluate_single_llm_response(
        benchmark_df=benchmark_df,
        true_maze_family="GREEDY_TRAP",
        representation_mode="features_ascii",
        raw_response=raw_response,
        model_name="mock",
    )

    assert result["classification_correct"] is True
    assert result["solver_selection_correct"] is False
    assert result["selected_solver_shortest_path"] is False
    assert result["quality_failure"] is True
    assert result["expansion_regret"] == -9


def test_evaluate_llm_response_rows() -> None:
    benchmark_df = make_test_benchmark_df()

    response_rows = [
        {
            "model_name": "mock",
            "true_maze_family": "GREEDY_TRAP",
            "representation_mode": "features_ascii",
            "raw_response": """
            {
              "edge_case_class": "GREEDY_TRAP",
              "recommended_solver": "A*",
              "confidence": 0.9,
              "reason": "A* is reliable and efficient."
            }
            """,
        }
    ]

    evaluation_df = evaluate_llm_response_rows(
        benchmark_df=benchmark_df,
        response_rows=response_rows,
    )

    assert len(evaluation_df) == 1
    assert evaluation_df.iloc[0]["recommended_solver"] == "A*"


def test_summarize_llm_evaluation() -> None:
    evaluation_df = pd.DataFrame(
        [
            {
                "classification_correct": True,
                "solver_selection_correct": True,
                "selected_solver_shortest_path": True,
                "quality_failure": False,
                "expansion_regret": 0,
            },
            {
                "classification_correct": False,
                "solver_selection_correct": False,
                "selected_solver_shortest_path": True,
                "quality_failure": False,
                "expansion_regret": 10,
            },
        ]
    )

    summary = summarize_llm_evaluation(evaluation_df)

    assert summary["n"] == 2.0
    assert summary["classification_accuracy"] == 0.5
    assert summary["solver_selection_accuracy"] == 0.5
    assert summary["shortest_path_rate"] == 1.0
    assert summary["quality_failure_rate"] == 0.0
    assert summary["average_expansion_regret"] == 5.0


def test_summarize_llm_evaluation_rejects_empty_df() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        summarize_llm_evaluation(pd.DataFrame())
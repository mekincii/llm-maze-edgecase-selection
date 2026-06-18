from pathlib import Path
from typing import Any

import pandas as pd

from src.llm.response_parser import parse_llm_selection_response


BENCHMARK_PATH = Path("data/results/classical_benchmark.csv")

GUARANTEED_SHORTEST_PATH_SOLVERS = {
    "BFS",
    "Dijkstra",
    "A*",
}


def load_classical_benchmark(path: Path = BENCHMARK_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Benchmark file not found: {path}. "
            "Run `python -m src.experiments.run_classical_benchmark` first."
        )

    return pd.read_csv(path)


def get_oracle_solvers(
    benchmark_df: pd.DataFrame,
    maze_family: str,
) -> set[str]:
    """
    Return empirical oracle solvers for the given maze family.

    Empirical oracle definition:
    Among all solvers that find a shortest path, choose those with the
    lowest expanded-node count on this exact instance.

    This oracle may select non-guaranteed solvers such as DFS if they happen
    to find a shortest path with fewer expansions on a specific maze.
    """
    family_df = benchmark_df[benchmark_df["maze_family"] == maze_family]

    if family_df.empty:
        raise ValueError(f"No benchmark rows found for maze family: {maze_family}")

    oracle_df = family_df[family_df["is_best_optimal_expansion"] == True]

    if oracle_df.empty:
        raise ValueError(f"No oracle solver found for maze family: {maze_family}")

    return set(oracle_df["solver_name"].tolist())


def get_guarantee_aware_oracle_solvers(
    benchmark_df: pd.DataFrame,
    maze_family: str,
) -> set[str]:
    """
    Return guarantee-aware oracle solvers for the given maze family.

    Guarantee-aware oracle definition:
    Among solvers with shortest-path guarantees, choose those that found a
    shortest path with the lowest expanded-node count.

    Guaranteed shortest-path solvers in this project:
    - BFS
    - Dijkstra
    - A*
    """
    family_df = benchmark_df[benchmark_df["maze_family"] == maze_family]

    if family_df.empty:
        raise ValueError(f"No benchmark rows found for maze family: {maze_family}")

    guaranteed_df = family_df[
        family_df["solver_name"].isin(GUARANTEED_SHORTEST_PATH_SOLVERS)
        & (family_df["success"] == True)
        & (family_df["is_shortest_path"] == True)
    ]

    if guaranteed_df.empty:
        raise ValueError(
            f"No guarantee-aware oracle solver found for maze family: {maze_family}"
        )

    best_expanded = guaranteed_df["expanded_nodes"].min()

    oracle_df = guaranteed_df[
        guaranteed_df["expanded_nodes"] == best_expanded
    ]

    return set(oracle_df["solver_name"].tolist())


def get_solver_row(
    benchmark_df: pd.DataFrame,
    maze_family: str,
    solver_name: str,
) -> pd.Series:
    family_solver_df = benchmark_df[
        (benchmark_df["maze_family"] == maze_family)
        & (benchmark_df["solver_name"] == solver_name)
    ]

    if family_solver_df.empty:
        raise ValueError(
            f"No benchmark row found for maze_family={maze_family}, "
            f"solver_name={solver_name}"
        )

    if len(family_solver_df) > 1:
        raise ValueError(
            f"Expected one benchmark row for maze_family={maze_family}, "
            f"solver_name={solver_name}, found {len(family_solver_df)}."
        )

    return family_solver_df.iloc[0]


def evaluate_single_llm_response(
    benchmark_df: pd.DataFrame,
    true_maze_family: str,
    representation_mode: str,
    raw_response: str,
    model_name: str = "mock",
) -> dict[str, Any]:
    """
    Evaluate one raw LLM response against the classical benchmark.

    Returns a row-like dictionary suitable for DataFrame construction.
    """
    parsed = parse_llm_selection_response(raw_response)

    empirical_oracle_solvers = get_oracle_solvers(
        benchmark_df=benchmark_df,
        maze_family=true_maze_family,
    )

    guarantee_aware_oracle_solvers = get_guarantee_aware_oracle_solvers(
        benchmark_df=benchmark_df,
        maze_family=true_maze_family,
    )

    selected_solver_row = get_solver_row(
        benchmark_df=benchmark_df,
        maze_family=true_maze_family,
        solver_name=parsed.recommended_solver,
    )

    empirical_oracle_expanded_nodes = min(
        get_solver_row(benchmark_df, true_maze_family, solver_name)[
            "expanded_nodes"
        ]
        for solver_name in empirical_oracle_solvers
    )

    guarantee_aware_oracle_expanded_nodes = min(
        get_solver_row(benchmark_df, true_maze_family, solver_name)[
            "expanded_nodes"
        ]
        for solver_name in guarantee_aware_oracle_solvers
    )

    selected_expanded_nodes = int(selected_solver_row["expanded_nodes"])
    selected_path_length = int(selected_solver_row["path_length"])
    shortest_path_length = int(selected_solver_row["shortest_path_length"])

    classification_correct = parsed.edge_case_class == true_maze_family

    empirical_solver_selection_correct = (
        parsed.recommended_solver in empirical_oracle_solvers
    )

    guarantee_aware_solver_selection_correct = (
        parsed.recommended_solver in guarantee_aware_oracle_solvers
    )

    selected_solver_success = bool(selected_solver_row["success"])
    selected_solver_shortest_path = bool(selected_solver_row["is_shortest_path"])

    quality_failure = (
        not selected_solver_success
        or not selected_solver_shortest_path
    )

    empirical_expansion_regret = (
        selected_expanded_nodes - int(empirical_oracle_expanded_nodes)
    )

    guarantee_aware_expansion_regret = (
        selected_expanded_nodes - int(guarantee_aware_oracle_expanded_nodes)
    )

    return {
        "model_name": model_name,
        "true_maze_family": true_maze_family,
        "representation_mode": representation_mode,
        "predicted_edge_case_class": parsed.edge_case_class,
        "recommended_solver": parsed.recommended_solver,
        "confidence": parsed.confidence,
        "reason": parsed.reason,
        "empirical_oracle_solvers": "|".join(sorted(empirical_oracle_solvers)),
        "guarantee_aware_oracle_solvers": "|".join(
            sorted(guarantee_aware_oracle_solvers)
        ),
        "classification_correct": classification_correct,
        "empirical_solver_selection_correct": empirical_solver_selection_correct,
        "guarantee_aware_solver_selection_correct": (
            guarantee_aware_solver_selection_correct
        ),
        "selected_solver_success": selected_solver_success,
        "selected_solver_shortest_path": selected_solver_shortest_path,
        "quality_failure": quality_failure,
        "selected_path_length": selected_path_length,
        "shortest_path_length": shortest_path_length,
        "selected_expanded_nodes": selected_expanded_nodes,
        "empirical_oracle_expanded_nodes": int(empirical_oracle_expanded_nodes),
        "guarantee_aware_oracle_expanded_nodes": int(
            guarantee_aware_oracle_expanded_nodes
        ),
        "empirical_expansion_regret": empirical_expansion_regret,
        "guarantee_aware_expansion_regret": guarantee_aware_expansion_regret,
    }


def evaluate_llm_response_rows(
    benchmark_df: pd.DataFrame,
    response_rows: list[dict[str, str]],
) -> pd.DataFrame:
    """
    Evaluate multiple LLM response rows.

    Each input row must contain:
    - true_maze_family
    - representation_mode
    - raw_response

    Optional:
    - model_name
    """
    evaluated_rows = []

    for row in response_rows:
        evaluated_rows.append(
            evaluate_single_llm_response(
                benchmark_df=benchmark_df,
                true_maze_family=row["true_maze_family"],
                representation_mode=row["representation_mode"],
                raw_response=row["raw_response"],
                model_name=row.get("model_name", "mock"),
            )
        )

    return pd.DataFrame(evaluated_rows)


def summarize_llm_evaluation(evaluation_df: pd.DataFrame) -> dict[str, float]:
    """
    Compute aggregate LLM-selection metrics.
    """
    if evaluation_df.empty:
        raise ValueError("Evaluation DataFrame cannot be empty.")

    return {
        "n": float(len(evaluation_df)),
        "classification_accuracy": float(
            evaluation_df["classification_correct"].mean()
        ),
        "empirical_solver_selection_accuracy": float(
            evaluation_df["empirical_solver_selection_correct"].mean()
        ),
        "guarantee_aware_solver_selection_accuracy": float(
            evaluation_df["guarantee_aware_solver_selection_correct"].mean()
        ),
        "shortest_path_rate": float(
            evaluation_df["selected_solver_shortest_path"].mean()
        ),
        "quality_failure_rate": float(
            evaluation_df["quality_failure"].mean()
        ),
        "average_empirical_expansion_regret": float(
            evaluation_df["empirical_expansion_regret"].mean()
        ),
        "average_guarantee_aware_expansion_regret": float(
            evaluation_df["guarantee_aware_expansion_regret"].mean()
        ),
    }


def get_mock_llm_responses() -> list[dict[str, str]]:
    """
    Small mock response set for testing the evaluator before using a real LLM.

    These are intentionally simple and deterministic.
    """
    return [
        {
            "model_name": "mock",
            "true_maze_family": "OPEN",
            "representation_mode": "features_ascii",
            "raw_response": """
            {
              "edge_case_class": "OPEN",
              "recommended_solver": "A*",
              "confidence": 0.80,
              "reason": "The maze is open, so A* can move directly toward the goal."
            }
            """,
        },
        {
            "model_name": "mock",
            "true_maze_family": "COMB",
            "representation_mode": "features_ascii",
            "raw_response": """
            {
              "edge_case_class": "COMB",
              "recommended_solver": "A*",
              "confidence": 0.75,
              "reason": "The maze has repeated branches, but the goal direction is clear."
            }
            """,
        },
        {
            "model_name": "mock",
            "true_maze_family": "ASTAR_TRAP",
            "representation_mode": "features_ascii",
            "raw_response": """
            {
              "edge_case_class": "ASTAR_TRAP",
              "recommended_solver": "Greedy Best-First",
              "confidence": 0.70,
              "reason": "The structure is heuristic-sensitive, but Greedy may expand fewer nodes here."
            }
            """,
        },
        {
            "model_name": "mock",
            "true_maze_family": "DFS_TRAP",
            "representation_mode": "features_ascii",
            "raw_response": """
            {
              "edge_case_class": "DFS_TRAP",
              "recommended_solver": "A*",
              "confidence": 0.85,
              "reason": "DFS may follow a longer branch, while A* is safer."
            }
            """,
        },
        {
            "model_name": "mock",
            "true_maze_family": "GREEDY_TRAP",
            "representation_mode": "features_ascii",
            "raw_response": """
            {
              "edge_case_class": "GREEDY_TRAP",
              "recommended_solver": "A*",
              "confidence": 0.72,
              "reason": "Greedy can overexpand in heuristic-attractive regions."
            }
            """,
        },
    ]


def main() -> None:
    benchmark_df = load_classical_benchmark()
    response_rows = get_mock_llm_responses()

    evaluation_df = evaluate_llm_response_rows(
        benchmark_df=benchmark_df,
        response_rows=response_rows,
    )

    print("=" * 90)
    print("Mock LLM selection evaluation")
    print("=" * 90)
    print(evaluation_df.to_string(index=False))

    print()
    print("=" * 90)
    print("Summary")
    print("=" * 90)

    summary = summarize_llm_evaluation(evaluation_df)

    for key, value in summary.items():
        print(f"{key}: {value:.3f}")


if __name__ == "__main__":
    main()
from pathlib import Path

import pandas as pd


BENCHMARK_PATH = Path("data/results/classical_benchmark.csv")


def load_benchmark(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Benchmark file not found: {path}. "
            "Run `python -m src.experiments.run_classical_benchmark` first."
        )

    return pd.read_csv(path)


def print_section(title: str) -> None:
    print()
    print("=" * 90)
    print(title)
    print("=" * 90)


def summarize_solver_performance(df: pd.DataFrame) -> None:
    print_section("Average solver performance")

    summary = (
        df.groupby("solver_name")
        .agg(
            success_rate=("success", "mean"),
            avg_path_length=("path_length", "mean"),
            avg_expanded_nodes=("expanded_nodes", "mean"),
            avg_visited_nodes=("visited_nodes", "mean"),
            avg_max_frontier_size=("max_frontier_size", "mean"),
            avg_runtime_ms=("runtime_ms", "mean"),
        )
        .reset_index()
        .sort_values("avg_expanded_nodes")
    )

    print(summary.to_string(index=False))


def summarize_shortest_path_reliability(df: pd.DataFrame) -> None:
    print_section("Shortest-path reliability by solver")

    summary = (
        df.groupby("solver_name")
        .agg(
            total_cases=("solver_name", "count"),
            success_count=("success", "sum"),
            shortest_path_count=("is_shortest_path", "sum"),
            avg_expanded_nodes=("expanded_nodes", "mean"),
        )
        .reset_index()
    )

    summary["success_rate"] = summary["success_count"] / summary["total_cases"]
    summary["shortest_path_rate"] = (
        summary["shortest_path_count"] / summary["total_cases"]
    )

    shortest_only = df[df["is_shortest_path"] == True]

    avg_expansion_shortest = (
        shortest_only.groupby("solver_name")
        .agg(avg_expanded_when_shortest=("expanded_nodes", "mean"))
        .reset_index()
    )

    summary = summary.merge(
        avg_expansion_shortest,
        on="solver_name",
        how="left",
    )

    summary = summary[
        [
            "solver_name",
            "total_cases",
            "success_count",
            "success_rate",
            "shortest_path_count",
            "shortest_path_rate",
            "avg_expanded_nodes",
            "avg_expanded_when_shortest",
        ]
    ].sort_values(
        ["shortest_path_rate", "avg_expanded_when_shortest"],
        ascending=[False, True],
    )

    print(summary.to_string(index=False))


def summarize_by_maze_family(df: pd.DataFrame) -> None:
    print_section("Solver performance by maze family")

    summary = (
        df.groupby(["maze_family", "solver_name"])
        .agg(
            path_length=("path_length", "mean"),
            expanded_nodes=("expanded_nodes", "mean"),
            visited_nodes=("visited_nodes", "mean"),
            runtime_ms=("runtime_ms", "mean"),
            is_shortest_path=("is_shortest_path", "mean"),
            is_best_optimal_expansion=("is_best_optimal_expansion", "mean"),
        )
        .reset_index()
        .sort_values(["maze_family", "expanded_nodes"])
    )

    print(summary.to_string(index=False))


def summarize_best_optimal_solvers(df: pd.DataFrame) -> None:
    print_section("Best solver among shortest-path solvers by maze family")

    best_rows = df[df["is_best_optimal_expansion"] == True].copy()

    if best_rows.empty:
        print("No best optimal solvers found.")
        return

    best_rows = best_rows[
        [
            "maze_family",
            "solver_name",
            "path_length",
            "expanded_nodes",
            "optimal_expansion_regret",
        ]
    ].sort_values(["maze_family", "expanded_nodes", "solver_name"])

    print(best_rows.to_string(index=False))


def summarize_raw_vs_optimal_conflicts(df: pd.DataFrame) -> None:
    print_section("Cases where raw expansion winner is not a shortest-path winner")

    raw_best = df[df["is_best_raw_expansion"] == True].copy()

    conflicts = raw_best[raw_best["is_shortest_path"] == False].copy()

    if conflicts.empty:
        print("No conflicts found.")
        return

    conflicts = conflicts[
        [
            "maze_family",
            "solver_name",
            "path_length",
            "shortest_path_length",
            "expanded_nodes",
            "raw_expansion_regret",
        ]
    ].sort_values(["maze_family", "solver_name"])

    print(conflicts.to_string(index=False))


def summarize_oracle_regret(df: pd.DataFrame) -> None:
    print_section("Average optimal-expansion regret by solver")

    shortest_df = df[df["is_shortest_path"] == True].copy()

    if shortest_df.empty:
        print("No shortest-path solver rows found.")
        return

    summary = (
        shortest_df.groupby("solver_name")
        .agg(
            avg_optimal_expansion_regret=("optimal_expansion_regret", "mean"),
            max_optimal_expansion_regret=("optimal_expansion_regret", "max"),
            shortest_path_count=("is_shortest_path", "sum"),
        )
        .reset_index()
        .sort_values("avg_optimal_expansion_regret")
    )

    print(summary.to_string(index=False))


def main() -> None:
    df = load_benchmark(BENCHMARK_PATH)

    print(f"Loaded {len(df)} rows from {BENCHMARK_PATH}")

    summarize_solver_performance(df)
    summarize_shortest_path_reliability(df)
    summarize_by_maze_family(df)
    summarize_best_optimal_solvers(df)
    summarize_raw_vs_optimal_conflicts(df)
    summarize_oracle_regret(df)


if __name__ == "__main__":
    main()
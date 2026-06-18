from pathlib import Path
from typing import Any

import pandas as pd


EVALUATION_PATH = Path("data/results/llm_response_evaluation.csv")
OUTPUT_DIR = Path("data/results")

REQUIRED_EVALUATION_COLUMNS = {
    "model_name",
    "true_maze_family",
    "representation_mode",
    "predicted_edge_case_class",
    "recommended_solver",
    "classification_correct",
    "empirical_solver_selection_correct",
    "guarantee_aware_solver_selection_correct",
    "selected_solver_shortest_path",
    "quality_failure",
    "guarantee_aware_policy_violation",
    "empirical_expansion_regret",
    "guarantee_aware_expansion_delta",
}


METRIC_COLUMNS = [
    "classification_correct",
    "empirical_solver_selection_correct",
    "guarantee_aware_solver_selection_correct",
    "selected_solver_shortest_path",
    "quality_failure",
    "guarantee_aware_policy_violation",
    "empirical_expansion_regret",
    "guarantee_aware_expansion_delta",
]


RENAMED_METRICS = {
    "classification_correct": "classification_accuracy",
    "empirical_solver_selection_correct": "empirical_solver_selection_accuracy",
    "guarantee_aware_solver_selection_correct": (
        "guarantee_aware_solver_selection_accuracy"
    ),
    "selected_solver_shortest_path": "shortest_path_rate",
    "quality_failure": "quality_failure_rate",
    "guarantee_aware_policy_violation": "guarantee_aware_policy_violation_rate",
    "empirical_expansion_regret": "average_empirical_expansion_regret",
    "guarantee_aware_expansion_delta": "average_guarantee_aware_expansion_delta",
}


def load_llm_evaluation_results(path: Path = EVALUATION_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"LLM evaluation file not found: {path}. "
            "Run `python -m src.experiments.evaluate_llm_response_file "
            "data/results/ollama_responses.csv` first."
        )

    df = pd.read_csv(path)

    missing_columns = REQUIRED_EVALUATION_COLUMNS - set(df.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"LLM evaluation file is missing required columns: {missing}")

    return df


def compute_overall_summary(evaluation_df: pd.DataFrame) -> pd.DataFrame:
    if evaluation_df.empty:
        raise ValueError("Evaluation DataFrame cannot be empty.")

    summary: dict[str, Any] = {"n": len(evaluation_df)}

    for column in METRIC_COLUMNS:
        summary[RENAMED_METRICS[column]] = evaluation_df[column].mean()

    return pd.DataFrame([summary])


def compute_group_summary(
    evaluation_df: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    if evaluation_df.empty:
        raise ValueError("Evaluation DataFrame cannot be empty.")

    missing_group_columns = set(group_columns) - set(evaluation_df.columns)

    if missing_group_columns:
        missing = ", ".join(sorted(missing_group_columns))
        raise ValueError(f"Missing group columns: {missing}")

    grouped = (
        evaluation_df
        .groupby(group_columns, dropna=False)
        .agg(
            n=("model_name", "size"),
            classification_accuracy=("classification_correct", "mean"),
            empirical_solver_selection_accuracy=(
                "empirical_solver_selection_correct",
                "mean",
            ),
            guarantee_aware_solver_selection_accuracy=(
                "guarantee_aware_solver_selection_correct",
                "mean",
            ),
            shortest_path_rate=("selected_solver_shortest_path", "mean"),
            quality_failure_rate=("quality_failure", "mean"),
            guarantee_aware_policy_violation_rate=(
                "guarantee_aware_policy_violation",
                "mean",
            ),
            average_empirical_expansion_regret=(
                "empirical_expansion_regret",
                "mean",
            ),
            average_guarantee_aware_expansion_delta=(
                "guarantee_aware_expansion_delta",
                "mean",
            ),
        )
        .reset_index()
    )

    return grouped


def compute_recommendation_counts(evaluation_df: pd.DataFrame) -> pd.DataFrame:
    if evaluation_df.empty:
        raise ValueError("Evaluation DataFrame cannot be empty.")

    return (
        evaluation_df
        .groupby(["model_name", "recommended_solver"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["model_name", "count"], ascending=[True, False])
    )


def compute_class_prediction_counts(evaluation_df: pd.DataFrame) -> pd.DataFrame:
    if evaluation_df.empty:
        raise ValueError("Evaluation DataFrame cannot be empty.")

    return (
        evaluation_df
        .groupby(["model_name", "predicted_edge_case_class"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["model_name", "count"], ascending=[True, False])
    )


def save_summary_tables(
    evaluation_df: pd.DataFrame,
    output_dir: Path = OUTPUT_DIR,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    tables = {
        "overall": compute_overall_summary(evaluation_df),
        "by_model": compute_group_summary(evaluation_df, ["model_name"]),
        "by_representation": compute_group_summary(
            evaluation_df,
            ["representation_mode"],
        ),
        "by_maze_family": compute_group_summary(
            evaluation_df,
            ["true_maze_family"],
        ),
        "by_model_and_representation": compute_group_summary(
            evaluation_df,
            ["model_name", "representation_mode"],
        ),
        "recommendation_counts": compute_recommendation_counts(evaluation_df),
        "class_prediction_counts": compute_class_prediction_counts(evaluation_df),
    }

    output_paths: dict[str, Path] = {}

    for name, table in tables.items():
        path = output_dir / f"llm_summary_{name}.csv"
        table.to_csv(path, index=False)
        output_paths[name] = path

    return output_paths


def print_table(title: str, table: pd.DataFrame) -> None:
    print()
    print("=" * 90)
    print(title)
    print("=" * 90)
    print(table.to_string(index=False))


def main() -> None:
    evaluation_df = load_llm_evaluation_results(EVALUATION_PATH)

    overall = compute_overall_summary(evaluation_df)
    by_model = compute_group_summary(evaluation_df, ["model_name"])
    by_representation = compute_group_summary(evaluation_df, ["representation_mode"])
    by_maze_family = compute_group_summary(evaluation_df, ["true_maze_family"])
    by_model_and_representation = compute_group_summary(
        evaluation_df,
        ["model_name", "representation_mode"],
    )
    recommendation_counts = compute_recommendation_counts(evaluation_df)
    class_prediction_counts = compute_class_prediction_counts(evaluation_df)

    print_table("Overall LLM evaluation summary", overall)
    print_table("Summary by model", by_model)
    print_table("Summary by representation mode", by_representation)
    print_table("Summary by maze family", by_maze_family)
    print_table("Summary by model and representation mode", by_model_and_representation)
    print_table("Recommended solver counts by model", recommendation_counts)
    print_table("Predicted edge-case class counts by model", class_prediction_counts)

    output_paths = save_summary_tables(evaluation_df, OUTPUT_DIR)

    print()
    print("=" * 90)
    print("Saved summary tables")
    print("=" * 90)

    for name, path in output_paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
import sys
from pathlib import Path

import pandas as pd

from src.experiments.evaluate_llm_selection import (
    evaluate_llm_response_rows,
    load_classical_benchmark,
    summarize_llm_evaluation,
)


RESPONSES_PATH = Path("data/prompts/llm_responses_template.csv")
OUTPUT_PATH = Path("data/results/llm_response_evaluation.csv")

REQUIRED_RESPONSE_COLUMNS = {
    "true_maze_family",
    "representation_mode",
    "model_name",
    "raw_response",
}


def load_completed_response_rows(
    path: Path = RESPONSES_PATH,
) -> list[dict[str, str]]:
    """
    Load completed LLM response rows from CSV.

    Rows with empty raw_response values are skipped.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"LLM response file not found: {path}. "
            "Run `python -m src.experiments.generate_llm_response_template` first."
        )

    df = pd.read_csv(path)

    missing_columns = REQUIRED_RESPONSE_COLUMNS - set(df.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Response file is missing required columns: {missing}")

    completed_df = df[
        df["raw_response"].notna()
        & (df["raw_response"].astype(str).str.strip() != "")
    ].copy()

    rows: list[dict[str, str]] = []

    for _, row in completed_df.iterrows():
        rows.append(
            {
                "true_maze_family": str(row["true_maze_family"]),
                "representation_mode": str(row["representation_mode"]),
                "model_name": str(row["model_name"]),
                "raw_response": str(row["raw_response"]),
            }
        )

    return rows


def save_evaluation_results(
    evaluation_df: pd.DataFrame,
    output_path: Path = OUTPUT_PATH,
) -> None:
    if evaluation_df.empty:
        raise ValueError("No evaluation rows to save.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    evaluation_df.to_csv(output_path, index=False)


def print_summary(summary: dict[str, float]) -> None:
    print()
    print("=" * 90)
    print("Summary")
    print("=" * 90)

    for key, value in summary.items():
        print(f"{key}: {value:.3f}")


def resolve_response_path_from_args() -> Path:
    """
    Resolve response CSV path from command-line arguments.

    Usage:
        python -m src.experiments.evaluate_llm_response_file
        python -m src.experiments.evaluate_llm_response_file data/results/ollama_responses.csv

    If no path is provided, the default response template path is used.
    """
    if len(sys.argv) == 1:
        return RESPONSES_PATH

    if len(sys.argv) == 2:
        return Path(sys.argv[1])

    raise ValueError(
        "Usage: python -m src.experiments.evaluate_llm_response_file "
        "[optional_response_csv_path]"
    )


def main() -> None:
    response_path = resolve_response_path_from_args()

    benchmark_df = load_classical_benchmark()
    response_rows = load_completed_response_rows(response_path)

    if not response_rows:
        print(f"No completed LLM responses found in {response_path}")
        print("Fill the raw_response column with JSON responses first.")
        return

    evaluation_df = evaluate_llm_response_rows(
        benchmark_df=benchmark_df,
        response_rows=response_rows,
    )

    save_evaluation_results(evaluation_df, OUTPUT_PATH)

    print("=" * 90)
    print("LLM response file evaluation")
    print("=" * 90)
    print(f"Response file: {response_path}")
    print()
    print(evaluation_df.to_string(index=False))

    summary = summarize_llm_evaluation(evaluation_df)
    print_summary(summary)

    print()
    print(f"Saved evaluation results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
from pathlib import Path
from typing import Any

import pandas as pd

from src.experiments.evaluate_llm_response_file import load_completed_response_rows
from src.experiments.evaluate_llm_selection import (
    evaluate_llm_response_rows,
    load_classical_benchmark,
    summarize_llm_evaluation,
)


OUTPUT_PATH = Path("data/results/prompt_version_comparison.csv")

PROMPT_VERSION_RESPONSE_FILES = {
    "v1_labels_only": Path("data/results/ollama_responses.csv"),
    "v2_definitions": Path("data/results/ollama_responses_v2_definitions.csv"),
    "v3_operational": Path("data/results/ollama_responses_v3_operational.csv"),
}


def evaluate_prompt_version(
    prompt_version: str,
    response_path: Path,
) -> dict[str, Any]:
    if not response_path.exists():
        raise FileNotFoundError(
            f"Response file for {prompt_version} not found: {response_path}"
        )

    benchmark_df = load_classical_benchmark()
    response_rows = load_completed_response_rows(response_path)

    if not response_rows:
        raise ValueError(f"No completed responses found for {prompt_version}")

    evaluation_df = evaluate_llm_response_rows(
        benchmark_df=benchmark_df,
        response_rows=response_rows,
    )

    summary = summarize_llm_evaluation(evaluation_df)

    return {
        "prompt_version": prompt_version,
        "response_file": str(response_path),
        **summary,
    }


def build_prompt_version_comparison(
    version_files: dict[str, Path] = PROMPT_VERSION_RESPONSE_FILES,
) -> pd.DataFrame:
    rows = []

    for prompt_version, response_path in version_files.items():
        rows.append(
            evaluate_prompt_version(
                prompt_version=prompt_version,
                response_path=response_path,
            )
        )

    return pd.DataFrame(rows)


def save_prompt_version_comparison(
    comparison_df: pd.DataFrame,
    output_path: Path = OUTPUT_PATH,
) -> None:
    if comparison_df.empty:
        raise ValueError("Prompt-version comparison DataFrame cannot be empty.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(output_path, index=False)


def main() -> None:
    comparison_df = build_prompt_version_comparison()
    save_prompt_version_comparison(comparison_df, OUTPUT_PATH)

    print("=" * 90)
    print("Prompt version comparison")
    print("=" * 90)
    print(comparison_df.to_string(index=False))

    print()
    print(f"Saved prompt-version comparison to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
from pathlib import Path
from typing import Any

import pandas as pd


PROMPTS_PATH = Path("data/prompts/llm_prompts.csv")
OUTPUT_PATH = Path("data/prompts/llm_responses_template.csv")

REQUIRED_PROMPT_COLUMNS = {
    "prompt_id",
    "maze_family",
    "representation_mode",
    "prompt",
}


def load_prompt_dataset(path: Path = PROMPTS_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Prompt dataset not found: {path}. "
            "Run `python -m src.experiments.generate_llm_prompts` first."
        )

    df = pd.read_csv(path)

    missing_columns = REQUIRED_PROMPT_COLUMNS - set(df.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Prompt dataset is missing required columns: {missing}")

    return df


def create_response_template_rows(
    prompt_df: pd.DataFrame,
    model_name: str = "manual",
) -> list[dict[str, Any]]:
    """
    Create response-template rows from the prompt dataset.

    The raw_response field is intentionally empty. Users can paste an LLM's
    JSON response there later.
    """
    missing_columns = REQUIRED_PROMPT_COLUMNS - set(prompt_df.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Prompt dataset is missing required columns: {missing}")

    rows: list[dict[str, Any]] = []

    for _, row in prompt_df.iterrows():
        rows.append(
            {
                "prompt_id": row["prompt_id"],
                "true_maze_family": row["maze_family"],
                "representation_mode": row["representation_mode"],
                "model_name": model_name,
                "raw_response": "",
                "prompt": row["prompt"],
            }
        )

    return rows


def save_response_template(
    rows: list[dict[str, Any]],
    output_path: Path = OUTPUT_PATH,
) -> None:
    if not rows:
        raise ValueError("No response-template rows to save.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)


def main() -> None:
    prompt_df = load_prompt_dataset(PROMPTS_PATH)

    rows = create_response_template_rows(
        prompt_df=prompt_df,
        model_name="manual",
    )

    save_response_template(rows, OUTPUT_PATH)

    print(f"Saved {len(rows)} LLM response-template rows to {OUTPUT_PATH}")
    print()
    print("Fill the raw_response column with JSON responses, then run:")
    print("python -m src.experiments.evaluate_llm_response_file")


if __name__ == "__main__":
    main()
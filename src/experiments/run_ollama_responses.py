from pathlib import Path
from typing import Any

import pandas as pd

from src.llm.ollama_client import generate_with_ollama


PROMPTS_PATH = Path("data/prompts/llm_prompts.csv")
OUTPUT_PATH = Path("data/results/ollama_responses.csv")

DEFAULT_MODELS = [
    "qwen3:1.7b",
    "qwen3:4b",
    "qwen3:8b",
]

REQUIRED_PROMPT_COLUMNS = {
    "prompt_id",
    "maze_family",
    "representation_mode",
    "prompt",
}


def load_prompt_rows(path: Path = PROMPTS_PATH) -> pd.DataFrame:
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


def run_ollama_prompt_rows(
    prompt_df: pd.DataFrame,
    model_names: list[str],
    temperature: float = 0.0,
    seed: int = 42,
    num_predict: int = 512,
    limit_prompts: int | None = None,
) -> list[dict[str, Any]]:
    """
    Run Ollama models on prompt rows and return response rows.

    The output schema is compatible with our response-file/evaluation pipeline.
    """
    rows: list[dict[str, Any]] = []

    if limit_prompts is not None:
        prompt_df = prompt_df.head(limit_prompts)

    total_runs = len(prompt_df) * len(model_names)
    run_index = 0

    for model_name in model_names:
        for _, prompt_row in prompt_df.iterrows():
            run_index += 1

            prompt_id = str(prompt_row["prompt_id"])
            true_maze_family = str(prompt_row["maze_family"])
            representation_mode = str(prompt_row["representation_mode"])
            prompt = str(prompt_row["prompt"])

            print(
                f"[{run_index}/{total_runs}] "
                f"model={model_name} prompt_id={prompt_id}"
            )

            result = generate_with_ollama(
                model=model_name,
                prompt=prompt,
                temperature=temperature,
                seed=seed,
                num_predict=num_predict,
            )

            rows.append(
                {
                    "prompt_id": prompt_id,
                    "true_maze_family": true_maze_family,
                    "representation_mode": representation_mode,
                    "model_name": model_name,
                    "raw_response": result.response,
                    "prompt": prompt,
                }
            )

    return rows


def save_ollama_responses(
    rows: list[dict[str, Any]],
    output_path: Path = OUTPUT_PATH,
) -> None:
    if not rows:
        raise ValueError("No Ollama response rows to save.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)


def main() -> None:
    prompt_df = load_prompt_rows(PROMPTS_PATH)

    rows = run_ollama_prompt_rows(
        prompt_df=prompt_df,
        model_names=DEFAULT_MODELS,
        temperature=0.0,
        seed=42,
        num_predict=512,
        limit_prompts=None,
    )

    save_ollama_responses(rows, OUTPUT_PATH)

    print()
    print(f"Saved {len(rows)} Ollama response rows to {OUTPUT_PATH}")
    print()
    print("Evaluate with:")
    print(
        "python -m src.experiments.evaluate_llm_response_file "
        "after copying/renaming this file if needed."
    )


if __name__ == "__main__":
    main()
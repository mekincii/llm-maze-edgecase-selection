import csv
from pathlib import Path
from typing import Any

from src.experiments.run_classical_benchmark import get_maze_cases
from src.llm.prompt_builder import (
    RepresentationMode,
    build_prompt_for_maze,
)


OUTPUT_PATH = Path("data/prompts/llm_prompts.csv")
V2_DEFINITIONS_OUTPUT_PATH = Path("data/prompts/llm_prompts_v2_definitions.csv")

REPRESENTATION_MODES: list[RepresentationMode] = [
    "features",
    "ascii",
    "features_ascii",
]


def generate_prompt_rows(
    width: int = 15,
    height: int = 15,
    include_solution_features: bool = False,
    include_edge_case_definitions: bool = False,
    prompt_version: str = "v1_labels_only",
) -> list[dict[str, Any]]:
    """
    Generate prompt rows for all maze families and representation modes.

    By default, shortest-path-derived features are excluded so that prompts
    represent a pre-search solver-selection setting.
    """
    rows: list[dict[str, Any]] = []

    for maze_family, factory in get_maze_cases():
        maze = factory(width=width, height=height)

        for representation_mode in REPRESENTATION_MODES:
            prompt = build_prompt_for_maze(
                maze=maze,
                representation_mode=representation_mode,
                include_solution_features=include_solution_features,
                include_edge_case_definitions=include_edge_case_definitions,
            )

            rows.append(
                {
                    "prompt_id": f"{maze_family}__{representation_mode}",
                    "prompt_version": prompt_version,
                    "maze_family": maze_family,
                    "width": width,
                    "height": height,
                    "representation_mode": representation_mode,
                    "include_solution_features": include_solution_features,
                    "prompt": prompt,
                }
            )

    return rows


def save_prompt_rows_to_csv(
    rows: list[dict[str, Any]],
    output_path: Path = OUTPUT_PATH,
) -> None:
    if not rows:
        raise ValueError("No prompt rows to save.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows = generate_prompt_rows(
        width=15,
        height=15,
        include_solution_features=False,
        include_edge_case_definitions=True,
        prompt_version="v2_definitions",
    )

    save_prompt_rows_to_csv(rows, V2_DEFINITIONS_OUTPUT_PATH)

    print(f"Saved {len(rows)} LLM prompts to {V2_DEFINITIONS_OUTPUT_PATH}")

    print()
    print("Prompt IDs:")
    for row in rows:
        print(f"- {row['prompt_id']}")


if __name__ == "__main__":
    main()
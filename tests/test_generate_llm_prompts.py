from pathlib import Path

import pandas as pd
import pytest

from src.experiments.generate_llm_prompts import (
    REPRESENTATION_MODES,
    generate_prompt_rows,
    save_prompt_rows_to_csv,
)


def test_generate_prompt_rows_creates_expected_count() -> None:
    rows = generate_prompt_rows(width=15, height=15)

    assert len(rows) == 15


def test_generate_prompt_rows_contains_all_representation_modes() -> None:
    rows = generate_prompt_rows(width=15, height=15)

    modes = {row["representation_mode"] for row in rows}

    assert modes == set(REPRESENTATION_MODES)


def test_generate_prompt_rows_contains_prompt_ids() -> None:
    rows = generate_prompt_rows(width=15, height=15)

    prompt_ids = {row["prompt_id"] for row in rows}

    assert "OPEN__features" in prompt_ids
    assert "OPEN__ascii" in prompt_ids
    assert "OPEN__features_ascii" in prompt_ids
    assert "GREEDY_TRAP__features_ascii" in prompt_ids


def test_generate_prompt_rows_excludes_solution_features_by_default() -> None:
    rows = generate_prompt_rows(width=15, height=15)

    for row in rows:
        assert "Shortest path length:" not in row["prompt"]
        assert "Shortest path / Manhattan ratio:" not in row["prompt"]


def test_generate_prompt_rows_can_include_solution_features() -> None:
    rows = generate_prompt_rows(
        width=15,
        height=15,
        include_solution_features=True,
    )

    feature_rows = [
        row
        for row in rows
        if row["representation_mode"] in {"features", "features_ascii"}
    ]

    assert feature_rows

    for row in feature_rows:
        assert "Shortest path length:" in row["prompt"]
        assert "Shortest path / Manhattan ratio:" in row["prompt"]


def test_save_prompt_rows_to_csv(tmp_path: Path) -> None:
    rows = generate_prompt_rows(width=15, height=15)
    output_path = tmp_path / "llm_prompts.csv"

    save_prompt_rows_to_csv(rows, output_path)

    assert output_path.exists()

    df = pd.read_csv(output_path)

    assert len(df) == 15
    assert set(df["representation_mode"]) == set(REPRESENTATION_MODES)
    assert "prompt" in df.columns


def test_save_prompt_rows_to_csv_rejects_empty_rows(tmp_path: Path) -> None:
    output_path = tmp_path / "empty.csv"

    with pytest.raises(ValueError, match="No prompt rows"):
        save_prompt_rows_to_csv([], output_path)


def test_generate_prompt_rows_can_include_edge_case_definitions() -> None:
    rows = generate_prompt_rows(
        width=15,
        height=15,
        include_edge_case_definitions=True,
        prompt_version="v2_definitions",
    )

    assert rows

    for row in rows:
        assert row["prompt_version"] == "v2_definitions"
        assert "Edge-case class definitions:" in row["prompt"]
        assert "ASTAR_TRAP" in row["prompt"]
        assert "GREEDY_TRAP" in row["prompt"]
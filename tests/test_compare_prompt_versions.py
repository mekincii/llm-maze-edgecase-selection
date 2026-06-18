from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from src.experiments.compare_prompt_versions import (
    build_prompt_version_comparison,
    save_prompt_version_comparison,
)


def test_build_prompt_version_comparison_from_mocked_evaluator() -> None:
    version_files = {
        "v1_labels_only": Path("v1.csv"),
        "v2_definitions": Path("v2.csv"),
    }

    def fake_evaluate_prompt_version(prompt_version: str, response_path: Path):
        return {
            "prompt_version": prompt_version,
            "response_file": str(response_path),
            "n": 45.0,
            "classification_accuracy": 0.5,
            "empirical_solver_selection_accuracy": 0.4,
            "guarantee_aware_solver_selection_accuracy": 0.8,
            "shortest_path_rate": 1.0,
            "quality_failure_rate": 0.0,
            "guarantee_aware_policy_violation_rate": 0.0,
            "average_empirical_expansion_regret": 10.0,
            "average_guarantee_aware_expansion_delta": 5.0,
        }

    with patch(
        "src.experiments.compare_prompt_versions.evaluate_prompt_version",
        side_effect=fake_evaluate_prompt_version,
    ):
        comparison_df = build_prompt_version_comparison(version_files)

    assert len(comparison_df) == 2
    assert set(comparison_df["prompt_version"]) == {
        "v1_labels_only",
        "v2_definitions",
    }


def test_save_prompt_version_comparison(tmp_path: Path) -> None:
    comparison_df = pd.DataFrame(
        [
            {
                "prompt_version": "v3_operational",
                "classification_accuracy": 0.4,
            }
        ]
    )

    output_path = tmp_path / "prompt_version_comparison.csv"

    save_prompt_version_comparison(comparison_df, output_path)

    assert output_path.exists()

    loaded_df = pd.read_csv(output_path)

    assert len(loaded_df) == 1
    assert loaded_df.iloc[0]["prompt_version"] == "v3_operational"


def test_save_prompt_version_comparison_rejects_empty_df(tmp_path: Path) -> None:
    output_path = tmp_path / "empty.csv"

    with pytest.raises(ValueError, match="cannot be empty"):
        save_prompt_version_comparison(pd.DataFrame(), output_path)
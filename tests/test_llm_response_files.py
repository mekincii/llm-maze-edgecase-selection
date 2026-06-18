from pathlib import Path

import pandas as pd
import pytest

from src.experiments.evaluate_llm_response_file import (
    load_completed_response_rows,
    resolve_response_path_from_args,
    save_evaluation_results,
)
from src.experiments.generate_llm_response_template import (
    create_response_template_rows,
    save_response_template,
)


def make_prompt_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "prompt_id": "OPEN__features",
                "maze_family": "OPEN",
                "representation_mode": "features",
                "prompt": "Prompt text here.",
            },
            {
                "prompt_id": "GREEDY_TRAP__features_ascii",
                "maze_family": "GREEDY_TRAP",
                "representation_mode": "features_ascii",
                "prompt": "Another prompt text here.",
            },
        ]
    )


def test_create_response_template_rows() -> None:
    prompt_df = make_prompt_df()

    rows = create_response_template_rows(prompt_df, model_name="manual")

    assert len(rows) == 2
    assert rows[0]["prompt_id"] == "OPEN__features"
    assert rows[0]["true_maze_family"] == "OPEN"
    assert rows[0]["representation_mode"] == "features"
    assert rows[0]["model_name"] == "manual"
    assert rows[0]["raw_response"] == ""
    assert "prompt" in rows[0]


def test_create_response_template_rows_rejects_missing_columns() -> None:
    prompt_df = pd.DataFrame(
        [
            {
                "prompt_id": "OPEN__features",
                "maze_family": "OPEN",
            }
        ]
    )

    with pytest.raises(ValueError, match="missing required columns"):
        create_response_template_rows(prompt_df)


def test_save_response_template(tmp_path: Path) -> None:
    prompt_df = make_prompt_df()
    rows = create_response_template_rows(prompt_df)
    output_path = tmp_path / "llm_responses_template.csv"

    save_response_template(rows, output_path)

    assert output_path.exists()

    df = pd.read_csv(output_path)

    assert len(df) == 2
    assert "raw_response" in df.columns
    assert "prompt" in df.columns


def test_save_response_template_rejects_empty_rows(tmp_path: Path) -> None:
    output_path = tmp_path / "empty.csv"

    with pytest.raises(ValueError, match="No response-template rows"):
        save_response_template([], output_path)


def test_load_completed_response_rows_skips_empty_responses(tmp_path: Path) -> None:
    response_path = tmp_path / "responses.csv"

    df = pd.DataFrame(
        [
            {
                "true_maze_family": "OPEN",
                "representation_mode": "features",
                "model_name": "manual",
                "raw_response": "",
            },
            {
                "true_maze_family": "GREEDY_TRAP",
                "representation_mode": "features_ascii",
                "model_name": "manual",
                "raw_response": """
                {
                  "edge_case_class": "GREEDY_TRAP",
                  "recommended_solver": "A*",
                  "confidence": 0.8,
                  "reason": "A* is reliable."
                }
                """,
            },
        ]
    )

    df.to_csv(response_path, index=False)

    rows = load_completed_response_rows(response_path)

    assert len(rows) == 1
    assert rows[0]["true_maze_family"] == "GREEDY_TRAP"
    assert rows[0]["representation_mode"] == "features_ascii"
    assert rows[0]["model_name"] == "manual"
    assert "recommended_solver" in rows[0]["raw_response"]


def test_load_completed_response_rows_rejects_missing_columns(tmp_path: Path) -> None:
    response_path = tmp_path / "bad_responses.csv"

    df = pd.DataFrame(
        [
            {
                "true_maze_family": "OPEN",
                "raw_response": "{}",
            }
        ]
    )

    df.to_csv(response_path, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        load_completed_response_rows(response_path)


def test_save_evaluation_results(tmp_path: Path) -> None:
    output_path = tmp_path / "evaluation.csv"

    evaluation_df = pd.DataFrame(
        [
            {
                "model_name": "manual",
                "true_maze_family": "OPEN",
                "classification_correct": True,
            }
        ]
    )

    save_evaluation_results(evaluation_df, output_path)

    assert output_path.exists()

    df = pd.read_csv(output_path)

    assert len(df) == 1
    assert df.iloc[0]["model_name"] == "manual"


def test_save_evaluation_results_rejects_empty_df(tmp_path: Path) -> None:
    output_path = tmp_path / "empty_eval.csv"

    with pytest.raises(ValueError, match="No evaluation rows"):
        save_evaluation_results(pd.DataFrame(), output_path)


def test_resolve_response_path_from_args_uses_default(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["python -m src.experiments.evaluate_llm_response_file"],
    )

    path = resolve_response_path_from_args()

    assert str(path) == "data\\prompts\\llm_responses_template.csv" or str(path) == "data/prompts/llm_responses_template.csv"


def test_resolve_response_path_from_args_uses_provided_path(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "python -m src.experiments.evaluate_llm_response_file",
            "data/results/ollama_responses.csv",
        ],
    )

    path = resolve_response_path_from_args()

    assert str(path) == "data\\results\\ollama_responses.csv" or str(path) == "data/results/ollama_responses.csv"


def test_resolve_response_path_from_args_rejects_too_many_args(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "python -m src.experiments.evaluate_llm_response_file",
            "one.csv",
            "two.csv",
        ],
    )

    with pytest.raises(ValueError, match="Usage"):
        resolve_response_path_from_args()
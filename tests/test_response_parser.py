import pytest

from src.llm.response_parser import (
    LLMSelectionResponse,
    parse_llm_selection_response,
    validate_llm_selection_data,
)


def test_parse_valid_llm_selection_response() -> None:
    raw_response = """
    {
      "edge_case_class": "GREEDY_TRAP",
      "recommended_solver": "A*",
      "confidence": 0.82,
      "reason": "A* balances heuristic guidance with accumulated path cost."
    }
    """

    parsed = parse_llm_selection_response(raw_response)

    assert isinstance(parsed, LLMSelectionResponse)
    assert parsed.edge_case_class == "GREEDY_TRAP"
    assert parsed.recommended_solver == "A*"
    assert parsed.confidence == 0.82
    assert parsed.reason == "A* balances heuristic guidance with accumulated path cost."


def test_parse_rejects_invalid_json() -> None:
    raw_response = """
    {
      "edge_case_class": "GREEDY_TRAP",
      "recommended_solver": "A*",
    }
    """

    with pytest.raises(ValueError, match="not valid JSON"):
        parse_llm_selection_response(raw_response)


def test_parse_rejects_non_object_json() -> None:
    raw_response = '["A*", "GREEDY_TRAP"]'

    with pytest.raises(ValueError, match="must be a JSON object"):
        parse_llm_selection_response(raw_response)


def test_validate_rejects_missing_key() -> None:
    data = {
        "edge_case_class": "GREEDY_TRAP",
        "recommended_solver": "A*",
        "confidence": 0.7,
    }

    with pytest.raises(ValueError, match="missing required keys"):
        validate_llm_selection_data(data)


def test_validate_rejects_invalid_edge_case_class() -> None:
    data = {
        "edge_case_class": "BAD_CLASS",
        "recommended_solver": "A*",
        "confidence": 0.7,
        "reason": "Valid-looking reason.",
    }

    with pytest.raises(ValueError, match="Invalid edge_case_class"):
        validate_llm_selection_data(data)


def test_validate_rejects_invalid_solver() -> None:
    data = {
        "edge_case_class": "GREEDY_TRAP",
        "recommended_solver": "Magic Solver",
        "confidence": 0.7,
        "reason": "Valid-looking reason.",
    }

    with pytest.raises(ValueError, match="Invalid recommended_solver"):
        validate_llm_selection_data(data)


def test_validate_rejects_non_numeric_confidence() -> None:
    data = {
        "edge_case_class": "GREEDY_TRAP",
        "recommended_solver": "A*",
        "confidence": "high",
        "reason": "Valid-looking reason.",
    }

    with pytest.raises(ValueError, match="confidence must be a number"):
        validate_llm_selection_data(data)


def test_validate_rejects_confidence_below_zero() -> None:
    data = {
        "edge_case_class": "GREEDY_TRAP",
        "recommended_solver": "A*",
        "confidence": -0.1,
        "reason": "Valid-looking reason.",
    }

    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        validate_llm_selection_data(data)


def test_validate_rejects_confidence_above_one() -> None:
    data = {
        "edge_case_class": "GREEDY_TRAP",
        "recommended_solver": "A*",
        "confidence": 1.1,
        "reason": "Valid-looking reason.",
    }

    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        validate_llm_selection_data(data)


def test_validate_rejects_non_string_reason() -> None:
    data = {
        "edge_case_class": "GREEDY_TRAP",
        "recommended_solver": "A*",
        "confidence": 0.7,
        "reason": 123,
    }

    with pytest.raises(ValueError, match="reason must be a string"):
        validate_llm_selection_data(data)


def test_validate_rejects_empty_reason() -> None:
    data = {
        "edge_case_class": "GREEDY_TRAP",
        "recommended_solver": "A*",
        "confidence": 0.7,
        "reason": "   ",
    }

    with pytest.raises(ValueError, match="reason cannot be empty"):
        validate_llm_selection_data(data)
import json
from dataclasses import dataclass
from typing import Any

from src.llm.prompt_builder import (
    ALLOWED_EDGE_CASE_CLASSES,
    ALLOWED_SOLVERS,
)


@dataclass
class LLMSelectionResponse:
    edge_case_class: str
    recommended_solver: str
    confidence: float
    reason: str


def parse_llm_selection_response(raw_response: str) -> LLMSelectionResponse:
    """
    Parse and validate an LLM solver-selection response.

    Expected JSON schema:
    {
      "edge_case_class": "...",
      "recommended_solver": "...",
      "confidence": 0.0,
      "reason": "..."
    }

    Raises:
        ValueError if the response is not valid JSON or fails validation.
    """
    try:
        data = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise ValueError("LLM response is not valid JSON.") from exc

    if not isinstance(data, dict):
        raise ValueError("LLM response must be a JSON object.")

    return validate_llm_selection_data(data)


def validate_llm_selection_data(data: dict[str, Any]) -> LLMSelectionResponse:
    required_keys = {
        "edge_case_class",
        "recommended_solver",
        "confidence",
        "reason",
    }

    missing_keys = required_keys - set(data.keys())

    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise ValueError(f"LLM response is missing required keys: {missing}")

    edge_case_class = data["edge_case_class"]
    recommended_solver = data["recommended_solver"]
    confidence = data["confidence"]
    reason = data["reason"]

    if edge_case_class not in ALLOWED_EDGE_CASE_CLASSES:
        raise ValueError(
            f"Invalid edge_case_class '{edge_case_class}'. "
            f"Allowed values: {ALLOWED_EDGE_CASE_CLASSES}"
        )

    if recommended_solver not in ALLOWED_SOLVERS:
        raise ValueError(
            f"Invalid recommended_solver '{recommended_solver}'. "
            f"Allowed values: {ALLOWED_SOLVERS}"
        )

    if not isinstance(confidence, int | float):
        raise ValueError("confidence must be a number.")

    confidence_float = float(confidence)

    if confidence_float < 0.0 or confidence_float > 1.0:
        raise ValueError("confidence must be between 0.0 and 1.0.")

    if not isinstance(reason, str):
        raise ValueError("reason must be a string.")

    if not reason.strip():
        raise ValueError("reason cannot be empty.")

    return LLMSelectionResponse(
        edge_case_class=edge_case_class,
        recommended_solver=recommended_solver,
        confidence=confidence_float,
        reason=reason.strip(),
    )
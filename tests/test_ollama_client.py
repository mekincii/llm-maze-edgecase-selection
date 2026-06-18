import json
from unittest.mock import patch

from src.llm.ollama_client import (
    build_ollama_generate_payload,
    generate_with_ollama,
)


class FakeHTTPResponse:
    def __init__(self, payload: dict):
        self.payload = payload

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")

    def __enter__(self) -> "FakeHTTPResponse":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        return None


def test_build_ollama_generate_payload() -> None:
    payload = build_ollama_generate_payload(
        model="qwen3:1.7b",
        prompt="Return JSON only.",
        temperature=0.0,
        seed=42,
        num_predict=128,
    )

    assert payload["model"] == "qwen3:1.7b"
    assert payload["prompt"] == "Return JSON only."
    assert payload["stream"] is False
    assert payload["options"]["temperature"] == 0.0
    assert payload["options"]["seed"] == 42
    assert payload["options"]["num_predict"] == 128
    assert payload["think"] is False
    assert payload["format"] == "json"


def test_generate_with_ollama_parses_response() -> None:
    fake_payload = {
        "model": "qwen3:1.7b",
        "response": '{"edge_case_class":"OPEN","recommended_solver":"A*","confidence":0.8,"reason":"Open grid."}',
        "done": True,
    }

    with patch(
        "src.llm.ollama_client.urlopen",
        return_value=FakeHTTPResponse(fake_payload),
    ):
        result = generate_with_ollama(
            model="qwen3:1.7b",
            prompt="Prompt here.",
            timeout_seconds=1,
        )

    assert result.model == "qwen3:1.7b"
    assert "recommended_solver" in result.response
    assert result.raw["done"] is True
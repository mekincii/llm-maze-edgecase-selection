import json
from dataclasses import dataclass
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen


DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"


@dataclass
class OllamaGenerateResult:
    model: str
    response: str
    raw: dict[str, Any]


def build_ollama_generate_payload(
    model: str,
    prompt: str,
    temperature: float = 0.0,
    seed: int = 42,
    num_predict: int = 512,
) -> dict[str, Any]:
    """
    Build payload for Ollama /api/generate.

    stream=False is important because it returns one complete JSON object,
    which is easier for our experiment pipeline.
    """
    return {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "seed": seed,
            "num_predict": num_predict,
        },
    }


def generate_with_ollama(
    model: str,
    prompt: str,
    base_url: str = DEFAULT_OLLAMA_BASE_URL,
    temperature: float = 0.0,
    seed: int = 42,
    num_predict: int = 512,
    timeout_seconds: int = 180,
) -> OllamaGenerateResult:
    """
    Generate a response from a local Ollama model.

    Raises:
        RuntimeError if Ollama is unreachable or returns malformed output.
    """
    payload = build_ollama_generate_payload(
        model=model,
        prompt=prompt,
        temperature=temperature,
        seed=seed,
        num_predict=num_predict,
    )

    request = Request(
        url=f"{base_url}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            response_body = response.read().decode("utf-8")
    except URLError as exc:
        raise RuntimeError(
            f"Could not connect to Ollama at {base_url}. "
            "Make sure Ollama is running."
        ) from exc

    try:
        data = json.loads(response_body)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Ollama returned invalid JSON.") from exc

    if "response" not in data:
        raise RuntimeError("Ollama response did not contain a 'response' field.")

    return OllamaGenerateResult(
        model=model,
        response=str(data["response"]).strip(),
        raw=data,
    )
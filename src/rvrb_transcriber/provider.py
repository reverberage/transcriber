"""Provider contract for rvrb-transcriber.

Follows satellite-protocol-v2.md:
- ModelProvider Protocol (structural, not ABC)
- get_provider() factory with n3rverberage fallback
- DEFAULT_MODEL and DEFAULT_BASE_URL read from env vars
- Generic fallback provider supports any OpenAI-compatible endpoint

Note: This satellite primarily uses Whisper (OpenAI API or local), not LLM providers.
The ModelProvider Protocol is defined here for future enhancements (e.g., Qwen-VL audio understanding).
"""

from __future__ import annotations

import json
import os
from typing import Any, Protocol

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Protocol definition (structural — no inheritance required)
# ---------------------------------------------------------------------------


class ModelProvider(Protocol):
    """Protocol for LLM providers.

    Any object with these attributes/methods is a valid provider.
    No ABC inheritance — uses duck typing.

    Note: Currently unused by transcriber (uses Whisper API directly).
    Reserved for future multimodal audio understanding features.
    """

    model: str
    base_url: str

    def complete(self, messages: list[dict], **kwargs: Any) -> str: ...
    def complete_structured(
        self,
        messages: list[dict],
        output_type: type[BaseModel],
        **kwargs: Any,
    ) -> BaseModel: ...
    def complete_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        **kwargs: Any,
    ) -> Any: ...


# ---------------------------------------------------------------------------
# Defaults — env-var-driven with Qwen fallback for backward compatibility
# ---------------------------------------------------------------------------

DEFAULT_MODEL: str = os.environ.get(
    "N3RVERBERAGE_DEFAULT_MODEL",
    "qwen3.5-omni-plus",  # Future: multimodal audio understanding
)
DEFAULT_BASE_URL: str = os.environ.get(
    "N3RVERBERAGE_DEFAULT_BASE_URL",
    "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
)

# Whisper models (current implementation)
WHISPER_MODEL: str = "whisper-1"
WHISPER_API_BASE: str = "https://api.openai.com/v1"


# ---------------------------------------------------------------------------
# Provider factory
# ---------------------------------------------------------------------------


def get_provider(
    model: str | None = None,
    provider: str | None = None,
) -> ModelProvider:
    """Resolve a model provider.

    Resolution order:
    1. Try n3rverberage.providers.get_provider() with ``provider:model`` format
    2. Fallback to ``_GenericProvider`` with env-var-driven defaults

    Parameters
    ----------
    model : str | None
        Override the model ID. If None, uses DEFAULT_MODEL.
    provider : str | None
        Provider name (qwen, openai, local).  Overrides
        ``N3RVERBERAGE_PROVIDER`` env var.

    Returns
    -------
    ModelProvider
        A provider instance matching the ModelProvider Protocol.

    Note
    ----
    Currently unused by transcriber engine (uses Whisper API directly).
    Reserved for future multimodal features.
    """
    resolved_model = model or DEFAULT_MODEL
    resolved_provider = provider or os.environ.get("N3RVERBERAGE_PROVIDER") or "qwen"

    # Try n3rverberage first (preferred — has fallback chain + quota detection)
    try:
        from n3rverberage.providers import get_provider as n3rv_get_provider

        return n3rv_get_provider(name=f"{resolved_provider}:{resolved_model}")
    except ImportError:
        pass

    # Fallback: generic OpenAI-compatible provider
    return _build_fallback_provider(resolved_provider, resolved_model)


def _build_fallback_provider(provider_type: str, model: str) -> _GenericProvider:
    """Construct a ``_GenericProvider`` from provider type and model."""
    provider_type = provider_type.strip().lower()
    fallbacks = {
        "qwen": (
            "qwen3.5-omni-plus",
            "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            "DASHSCOPE_API_KEY",
        ),
        "openai": (
            "gpt-4",
            "https://api.openai.com/v1",
            "OPENAI_API_KEY",
        ),
        "local": (
            "qwen2.5",
            "http://127.0.0.1:11434/v1",
            "",
        ),
    }

    if provider_type not in fallbacks:
        raise ValueError(
            f"Unknown provider type: '{provider_type}'. "
            f"Supported: {', '.join(fallbacks)}"
        )

    default_model, default_url, api_key_var = fallbacks[provider_type]
    base_url = os.environ.get("N3RVERBERAGE_DEFAULT_BASE_URL") or default_url

    api_key: str | None = None
    if api_key_var:
        api_key = os.environ.get(api_key_var)
        if not api_key:
            raise ValueError(
                f"{api_key_var} is not set. Set it or install n3rverberage."
            )

    return _GenericProvider(
        model=model or default_model,
        base_url=base_url,
        api_key=api_key or "not-needed",
    )


# ---------------------------------------------------------------------------
# Generic OpenAI-compatible provider (fallback when n3rverberage absent)
# ---------------------------------------------------------------------------


class _GenericProvider:
    """Minimal OpenAI-compatible provider for any endpoint.

    Used as fallback when n3rverberage is not installed.
    No provider-specific error handling — all errors are generic.
    """

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self._api_key = api_key

    def _client(self):
        from openai import OpenAI

        return OpenAI(api_key=self._api_key, base_url=self.base_url, timeout=60.0)

    def complete(self, messages: list[dict], **kwargs: Any) -> str:
        max_tokens = kwargs.pop("max_tokens", 4096)
        try:
            response = self._client().chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                **kwargs,
            )
        except Exception as exc:
            status_code = getattr(exc, "status_code", 500) or 500
            raise RuntimeError(f"[{self.model}] HTTP {status_code}: {exc}") from exc
        return response.choices[0].message.content or ""

    def complete_structured(
        self,
        messages: list[dict],
        output_type: type[BaseModel],
        **kwargs: Any,
    ) -> BaseModel:
        schema = output_type.model_json_schema()
        response_format: dict[str, Any] = {
            "type": "json_schema",
            "json_schema": {
                "name": schema.get("title", output_type.__name__),
                "schema": schema,
                "strict": True,
            },
        }
        max_tokens = kwargs.pop("max_tokens", 4096)
        try:
            response = self._client().chat.completions.create(
                model=self.model,
                messages=messages,
                response_format=response_format,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            status_code = getattr(exc, "status_code", 500) or 500
            raise RuntimeError(f"[{self.model}] HTTP {status_code}: {exc}") from exc

        raw = response.choices[0].message.content
        if not raw:
            raise ValueError(f"Empty structured response from {self.model}")

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON from {self.model}: {exc}") from exc

        return output_type.model_validate(parsed)

    def complete_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        **kwargs: Any,
    ) -> Any:
        max_tokens = kwargs.pop("max_tokens", 4096)
        try:
            response = self._client().chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                max_tokens=max_tokens,
                **kwargs,
            )
        except Exception as exc:
            status_code = getattr(exc, "status_code", 500) or 500
            raise RuntimeError(f"[{self.model}] HTTP {status_code}: {exc}") from exc

        return response.choices[0].message

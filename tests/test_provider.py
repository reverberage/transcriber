"""Tests for provider.py — env-var defaults, no Qwen-specific errors."""

from __future__ import annotations

from rvrb_transcriber.provider import DEFAULT_BASE_URL, DEFAULT_MODEL, get_provider


class TestDefaults:
    """DEFAULT_MODEL and DEFAULT_BASE_URL are valid strings."""

    def test_default_model_is_string(self) -> None:
        assert isinstance(DEFAULT_MODEL, str)

    def test_default_base_url_is_string(self) -> None:
        assert isinstance(DEFAULT_BASE_URL, str)

    def test_default_model_has_fallback(self) -> None:
        assert len(DEFAULT_MODEL) > 0

    def test_default_base_url_has_fallback(self) -> None:
        assert DEFAULT_BASE_URL.startswith("http")


class TestGetProvider:
    """get_provider() returns a valid ModelProvider."""

    def test_get_provider_no_args(self) -> None:
        provider = get_provider()
        assert hasattr(provider, "model")
        assert hasattr(provider, "base_url")
        assert callable(getattr(provider, "complete", None))
        assert callable(getattr(provider, "complete_structured", None))
        assert callable(getattr(provider, "complete_with_tools", None))

    def test_get_provider_with_model(self) -> None:
        provider = get_provider(model="gpt-4")
        assert provider.model == "gpt-4"

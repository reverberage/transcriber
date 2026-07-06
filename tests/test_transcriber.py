"""Unit tests for the top-level transcribe() function."""

import pytest
from rvrb_transcriber import transcribe


def test_transcribe_file_not_found():
    with pytest.raises(FileNotFoundError, match="File not found"):
        transcribe("/nonexistent/path.mp3")


def test_transcribe_unknown_engine():
    with pytest.raises(ValueError, match="Unknown engine"):
        transcribe("dummy.wav", engine="invalid")


def test_transcribe_openai_missing_key(monkeypatch):
    """Should raise ValueError when OPENAI_API_KEY is not set."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    # We need the file to exist to get past file check, then fail on key
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".mp3") as f:
        with pytest.raises(ValueError, match="OpenAI API key required"):
            transcribe(f.name, engine="openai", api_key=None)

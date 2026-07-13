"""rvrb-transcriber: Atomic audio/video transcription.

A standalone LEGO piece. Audio in, text out.

Usage:
    from rvrb_transcriber import transcribe

    result = transcribe("interview.mp3")
    print(result.text)

    # Output formats:
    print(result.to_srt())   # SubRip subtitles
    print(result.to_vtt())   # WebVTT
    print(result.model_dump_json(indent=2))  # JSON
"""

from .models import Transcript, Segment
from .engine import OpenAIWhisperEngine, LocalWhisperEngine, TranscriptionEngine
from .provider import ModelProvider, get_provider, DEFAULT_MODEL, DEFAULT_BASE_URL

__all__ = [
    "Transcript",
    "Segment",
    "transcribe",
    "OpenAIWhisperEngine",
    "LocalWhisperEngine",
    "ModelProvider",
    "get_provider",
    "DEFAULT_MODEL",
    "DEFAULT_BASE_URL",
]


def transcribe(
    file_path: str,
    engine: str = "openai",
    language: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> Transcript:
    """Transcribe an audio or video file.

    Args:
        file_path: Path to the audio/video file.
        engine: 'openai' (API) or 'local' (local whisper).
        language: Optional language code (e.g., 'en', 'es').
        model: Override default model.
        api_key: OpenAI API key. Uses OPENAI_API_KEY env var if omitted.

    Returns:
        Transcript with full text and timed segments.
    """
    if engine not in ("openai", "local"):
        raise ValueError(f"Unknown engine: '{engine}'. Use 'openai' or 'local'.")

    from pathlib import Path

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    engine_obj: TranscriptionEngine
    if engine == "openai":
        import os

        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY env var "
                "or pass api_key parameter."
            )
        engine_obj = OpenAIWhisperEngine(api_key=key, model=model or "whisper-1")
    else:  # engine == "local"
        engine_obj = LocalWhisperEngine(model=model or "base")

    return engine_obj.transcribe(file_path, language=language)

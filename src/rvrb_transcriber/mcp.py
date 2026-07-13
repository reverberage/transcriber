"""MCP server for rvrb-transcriber.

Follows satellite-protocol-v2.md:
- Gated import (mcp is optional dependency)
- Tool registration via @mcp.tool()
- stdio transport via mcp.run()

Install with: pip install rvrb-transcriber[mcp]
"""

from __future__ import annotations

from pathlib import Path

# Gated import - mcp is optional
try:
    from mcp.server import FastMCP
except ImportError:
    raise ImportError(
        "MCP support requires the 'mcp' extra. Install with: pip install rvrb-transcriber[mcp]"
    )

from rvrb_transcriber import Transcript
from rvrb_transcriber.engine import LocalWhisperEngine, OpenAIWhisperEngine

# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

mcp = FastMCP("rvrb-transcriber")


@mcp.tool()
def transcribe(
    file_path: str,
    engine: str = "openai",
    language: str | None = None,
    model: str | None = None,
) -> dict:
    """Transcribe an audio or video file.

    Parameters
    ----------
    file_path : str
        Path to the audio/video file.
    engine : str
        'openai' (API) or 'local' (local whisper). Default: 'openai'.
    language : str | None
        Optional language code (e.g., 'en', 'es', 'pt').
    model : str | None
        Override default model. For 'openai': 'whisper-1'. For 'local': whisper model size.

    Returns
    -------
    dict
        Transcription result with keys:
        - text: Full transcribed text
        - segments: List of timed segments [{start, end, text}, ...]
        - language: Detected language code
        - duration_seconds: Audio duration
        - srt: SubRip subtitle format
        - vtt: WebVTT format
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Select engine
    if engine == "openai":
        import os

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY required for 'openai' engine")
        engine_obj = OpenAIWhisperEngine(api_key=api_key, model=model or "whisper-1")
    elif engine == "local":
        engine_obj = LocalWhisperEngine(model=model or "base")
    else:
        raise ValueError(f"Unknown engine: {engine!r}. Use 'openai' or 'local'.")

    # Transcribe
    result: Transcript = engine_obj.transcribe(str(path), language=language)

    # Return as dict with all formats
    return {
        "text": result.text,
        "segments": [seg.model_dump() for seg in result.segments],
        "language": result.language,
        "duration_seconds": result.duration_seconds,
        "srt": result.to_srt(),
        "vtt": result.to_vtt(),
    }


@mcp.tool()
def transcribe_to_srt(
    file_path: str,
    output_path: str | None = None,
    engine: str = "openai",
    language: str | None = None,
) -> str:
    """Transcribe audio/video and return SRT subtitles.

    Parameters
    ----------
    file_path : str
        Path to the audio/video file.
    output_path : str | None
        If provided, write SRT to this file. Otherwise return as string.
    engine : str
        'openai' or 'local'. Default: 'openai'.
    language : str | None
        Optional language code.

    Returns
    -------
    str
        SRT formatted subtitles (also written to output_path if provided).
    """
    result = transcribe(file_path=file_path, engine=engine, language=language)
    srt_content = result["srt"]

    if output_path:
        Path(output_path).write_text(srt_content, encoding="utf-8")

    return srt_content


@mcp.tool()
def transcribe_to_vtt(
    file_path: str,
    output_path: str | None = None,
    engine: str = "openai",
    language: str | None = None,
) -> str:
    """Transcribe audio/video and return WebVTT subtitles.

    Parameters
    ----------
    file_path : str
        Path to the audio/video file.
    output_path : str | None
        If provided, write VTT to this file. Otherwise return as string.
    engine : str
        'openai' or 'local'. Default: 'openai'.
    language : str | None
        Optional language code.

    Returns
    -------
    str
        WebVTT formatted subtitles (also written to output_path if provided).
    """
    result = transcribe(file_path=file_path, engine=engine, language=language)
    vtt_content = result["vtt"]

    if output_path:
        Path(output_path).write_text(vtt_content, encoding="utf-8")

    return vtt_content


@mcp.resource("transcriber://info")
def get_info() -> str:
    """Get transcriber service information."""
    return """rvrb-transcriber v0.1.0
Audio/video transcription service.

Supported engines:
- openai: Whisper API (requires OPENAI_API_KEY)
- local: Local Whisper model (requires whisper package)

Supported formats:
- JSON (structured result)
- SRT (SubRip subtitles)
- VTT (WebVTT subtitles)

Usage: Call transcribe(file_path, engine, language, model)
"""


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the MCP server with stdio transport."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()

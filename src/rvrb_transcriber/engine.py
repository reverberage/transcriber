"""Transcription engines: OpenAI API and local Whisper."""

from __future__ import annotations

from pathlib import Path

from .models import Transcript, Segment


class TranscriptionEngine:
    """Base transcription engine."""

    def transcribe(
        self, file_path: str | Path, language: str | None = None
    ) -> Transcript:
        raise NotImplementedError


class OpenAIWhisperEngine(TranscriptionEngine):
    """Transcribe using OpenAI's Whisper API."""

    def __init__(self, api_key: str | None = None, model: str = "whisper-1"):
        self.api_key = api_key
        self.model = model

    def transcribe(
        self, file_path: str | Path, language: str | None = None
    ) -> Transcript:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key)
        audio_path = Path(file_path)

        kwargs: dict = {
            "model": self.model,
            "file": audio_path,
            "response_format": "verbose_json",
        }
        if language:
            kwargs["language"] = language

        result = client.audio.transcriptions.create(**kwargs)

        segments = []
        if hasattr(result, "segments") and result.segments:
            for seg in result.segments:
                segments.append(Segment(start=seg.start, end=seg.end, text=seg.text))

        return Transcript(
            text=result.text,
            segments=segments,
            language=getattr(result, "language", "unknown") or "unknown",
            duration_seconds=getattr(result, "duration", 0.0) or 0.0,
        )


class LocalWhisperEngine(TranscriptionEngine):
    """Transcribe using local openai-whisper."""

    def __init__(self, model: str = "base"):
        self.model_name = model
        self._model = None

    @property
    def model(self):
        if self._model is None:
            import whisper

            self._model = whisper.load_model(self.model_name)
        return self._model

    def transcribe(
        self, file_path: str | Path, language: str | None = None
    ) -> Transcript:
        kwargs = {}
        if language:
            kwargs["language"] = language

        result = self.model.transcribe(str(file_path), **kwargs)

        segments = []
        for seg in result.get("segments", []):
            segments.append(
                Segment(start=seg["start"], end=seg["end"], text=seg["text"].strip())
            )

        return Transcript(
            text=result["text"].strip(),
            segments=segments,
            language=result.get("language", "unknown"),
            duration_seconds=result.get("duration", 0.0),
        )

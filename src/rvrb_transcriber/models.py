"""Data models for transcription results."""

from __future__ import annotations

from datetime import timedelta

from pydantic import BaseModel, Field


class Segment(BaseModel):
    """A timed segment of transcribed audio."""

    start: float = Field(description="Start time in seconds")
    end: float = Field(description="End time in seconds")
    text: str = Field(description="Transcribed text for this segment")

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def start_timedelta(self) -> timedelta:
        return timedelta(seconds=self.start)

    @property
    def end_timedelta(self) -> timedelta:
        return timedelta(seconds=self.end)


class Transcript(BaseModel):
    """Complete transcription result."""

    text: str = Field(description="Full transcribed text")
    segments: list[Segment] = Field(default_factory=list, description="Timed segments")
    language: str = Field(default="unknown", description="Detected language code")
    duration_seconds: float = Field(
        default=0.0, description="Audio duration in seconds"
    )

    def to_srt(self) -> str:
        """Export as SRT subtitle format."""
        lines: list[str] = []
        for i, seg in enumerate(self.segments, 1):
            lines.append(str(i))
            lines.append(
                f"{_format_srt_time(seg.start_timedelta)} --> {_format_srt_time(seg.end_timedelta)}"
            )
            lines.append(seg.text.strip())
            lines.append("")
        return "\n".join(lines)

    def to_vtt(self) -> str:
        """Export as WebVTT subtitle format."""
        lines = ["WEBVTT", ""]
        for seg in self.segments:
            lines.append(
                f"{_format_vtt_time(seg.start_timedelta)} --> {_format_vtt_time(seg.end_timedelta)}"
            )
            lines.append(seg.text.strip())
            lines.append("")
        return "\n".join(lines)


def _format_srt_time(td: timedelta) -> str:
    total = td.total_seconds()
    hours = int(total // 3600)
    minutes = int((total % 3600) // 60)
    seconds = int(total % 60)
    millis = int((total * 1000) % 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"


def _format_vtt_time(td: timedelta) -> str:
    total = td.total_seconds()
    hours = int(total // 3600)
    minutes = int((total % 3600) // 60)
    seconds = int(total % 60)
    millis = int((total * 1000) % 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02}.{millis:03}"

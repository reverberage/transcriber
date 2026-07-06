# rvrb-transcriber

**Atomic audio/video transcription.** Audio in, text out.

Part of the [reverberage](https://github.com/reverberage) ecosystem — composable MCP-native toolkits for audio, video, and text.

## Install

```bash
# OpenAI API (recommended for quick start)
pip install "rvrb-transcriber[openai]"

# Local whisper (no API calls, runs on your machine)
pip install "rvrb-transcriber[local]"
```

Requires `OPENAI_API_KEY` env var when using the `openai` engine.

## Use

### CLI

```bash
rvrb-transcribe interview.mp3
rvrb-transcribe recording.wav --engine local --language es
rvrb-transcribe video.mp4 --format srt --output subtitles.srt
```

### Python

```python
from rvrb_transcriber import transcribe

result = transcribe("interview.mp3")
print(result.text)

for segment in result.segments:
    print(f"[{segment.start:.1f}s] {segment.text}")

# Export formats
print(result.to_srt())  # SubRip subtitles
print(result.to_vtt())  # WebVTT
print(result.model_dump_json(indent=2))  # JSON
```

## Output formats

| Format | Flag |
|--------|------|
| Plain text | `--format text` (default) |
| SRT subtitles | `--format srt` |
| WebVTT | `--format vtt` |
| JSON (with segments) | `--format json` |

## Engines

| Engine | Requires | Best for |
|--------|----------|----------|
| `openai` | `pip install rvrb-transcriber[openai]` + API key | Quick, high accuracy, no local GPU |
| `local` | `pip install rvrb-transcriber[local]` | Offline, no API costs, GPU recommended |

## License

Apache-2.0 — same as the reverberage ecosystem.

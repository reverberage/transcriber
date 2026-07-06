"""CLI entry point for rvrb-transcriber."""

from __future__ import annotations

from pathlib import Path

import typer

from . import transcribe

app = typer.Typer(
    name="rvrb-transcribe",
    help="Atomic audio/video transcription. Audio in, text out.",
)


@app.command()
def main(
    file_path: str = typer.Argument(..., help="Path to audio/video file"),
    engine: str = typer.Option("openai", help="Engine: 'openai' or 'local'"),
    language: str | None = typer.Option(None, help="Language code (e.g., 'en', 'es')"),
    model: str | None = typer.Option(None, help="Model name override"),
    output_format: str = typer.Option(
        "text", help="Output format: text, srt, vtt, json"
    ),
    output_file: str | None = typer.Option(
        None, "--output", "-o", help="Write to file instead of stdout"
    ),
):
    """Transcribe an audio or video file to text."""
    try:
        result = transcribe(
            file_path=file_path,
            engine=engine,
            language=language,
            model=model,
        )
    except FileNotFoundError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(2)

    if output_format == "json":
        output = result.model_dump_json(indent=2)
    elif output_format == "srt":
        output = result.to_srt()
    elif output_format == "vtt":
        output = result.to_vtt()
    else:
        output = result.text

    if output_file:
        Path(output_file).write_text(output)
        typer.echo(f"Saved to {output_file}")
    else:
        typer.echo(output)


if __name__ == "__main__":
    app()

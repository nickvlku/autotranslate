"""CLI entry point for autotranslate."""

import os
import sys
from pathlib import Path

import click

from .audio import extract_audio, needs_chunking
from .config import Config
from .srt import read_srt, write_srt
from .transcribe import transcribe
from .translate import translate


@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option(
    "--from",
    "source_lang",
    required=True,
    help="Source language code (e.g., ja, zh, es)",
)
@click.option(
    "--to",
    "target_lang",
    required=True,
    help="Target language code (e.g., en, fr, de)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output SRT file path (default: input_name.{target}.srt)",
)
@click.option(
    "--audio",
    "-a",
    is_flag=True,
    help="Treat input as audio file (skip extraction)",
)
@click.option(
    "--transcript",
    "-t",
    is_flag=True,
    help="Treat input as transcript SRT (skip extraction and transcription)",
)
@click.option(
    "--llm",
    type=click.Choice(["deepseek", "openai", "ollama"]),
    default="deepseek",
    help="LLM provider for translation (openai uses GPT-5.2 by default)",
)
@click.option(
    "--whisper",
    type=click.Choice(["cloud", "local"]),
    default="cloud",
    help="Whisper mode for transcription",
)
@click.option(
    "--ollama-model",
    default="llama3.1:8b",
    help="Ollama model to use for translation",
)
@click.option(
    "--openai-model",
    default="gpt-4o",
    help="OpenAI model for translation (gpt-5.2-chat-latest, gpt-5.2, gpt-4o)",
)
@click.option(
    "--cleanup",
    is_flag=True,
    help="Delete intermediate files (audio, transcript) on success",
)
@click.option(
    "--boost",
    type=float,
    default=None,
    help="Boost audio volume in dB (e.g., 10 for +10dB, helps with quiet audio)",
)
def main(
    input_path: str,
    source_lang: str,
    target_lang: str,
    output: str | None,
    audio: bool,
    transcript: bool,
    llm: str,
    whisper: str,
    ollama_model: str,
    openai_model: str,
    cleanup: bool,
    boost: float | None,
) -> None:
    """Translate video/audio to subtitles in another language.

    INPUT_PATH can be a video, audio (--audio), or transcript SRT (--transcript).

    \b
    Examples:
      autotranslate video.mp4 --from ja --to en
      autotranslate audio.wav --audio --from ja --to en
      autotranslate transcript.ja.srt --transcript --from ja --to en
    """
    config = Config.from_env()

    # Validate flags
    if audio and transcript:
        raise click.ClickException("Cannot use both --audio and --transcript")

    # Validate API keys for selected providers
    if not transcript and whisper == "cloud" and not config.has_openai():
        raise click.ClickException(
            "OPENAI_API_KEY environment variable required for cloud Whisper. "
            "Use --whisper local for offline mode."
        )

    if llm == "deepseek" and not config.has_deepseek():
        raise click.ClickException(
            "DEEPSEEK_API_KEY environment variable required for DeepSeek. "
            "Use --llm ollama for offline mode."
        )

    if llm == "openai" and not config.has_openai():
        raise click.ClickException(
            "OPENAI_API_KEY environment variable required for OpenAI GPT-4o. "
            "Use --llm ollama for offline mode."
        )

    # Determine input type and base name for outputs
    input_p = Path(input_path)
    if transcript:
        input_type = "Transcript"
        # Remove model and language suffixes (e.g., "video.whisper.ja.srt" -> "video")
        base_name = input_p.stem
        # Strip known suffixes: language codes, model names
        for _ in range(3):  # Strip up to 3 suffixes
            if "." in base_name:
                base_name = base_name.rsplit(".", 1)[0]
            else:
                break
    elif audio:
        input_type = "Audio"
        base_name = input_p.stem
    else:
        input_type = "Video"
        base_name = input_p.stem

    # Determine model names for filenames
    whisper_model_name = "whisper" if whisper == "cloud" else "whisper-local"
    if llm == "openai":
        llm_model_name = openai_model
    elif llm == "ollama":
        llm_model_name = ollama_model
    else:
        llm_model_name = llm  # deepseek

    # Default output paths with model names
    if output is None:
        output = f"{base_name}.{llm_model_name}.{target_lang}.srt"
    transcript_output = f"{base_name}.{whisper_model_name}.{source_lang}.srt"

    click.echo(f"{input_type}: {input_path}")
    click.echo(f"Translation: {source_lang} → {target_lang}")
    if not transcript:
        click.echo(f"Whisper: {whisper}, LLM: {llm}")
    else:
        click.echo(f"LLM: {llm}")
    click.echo(f"Output: {output}")
    click.echo()

    audio_path = None
    extracted_audio = False
    transcribed = False

    try:
        # Step 1: Get source subtitles (from transcript or via extraction+transcription)
        if transcript:
            click.echo(f"Loading transcript: {input_path}")
            subtitles = read_srt(input_path)
            click.echo(f"  Loaded {len(subtitles)} segments")
        else:
            # Get audio (extract or use provided)
            if audio:
                audio_path = Path(input_path)
                click.echo(f"Using audio file: {audio_path}")
            else:
                boost_msg = f" (boost: +{boost}dB)" if boost else ""
                click.echo(f"Extracting audio{boost_msg}...")
                audio_path = extract_audio(input_path, boost_db=boost)
                extracted_audio = True
                click.echo(f"  Saved to {audio_path}")

            # Check if chunking will be needed (for cloud mode)
            if whisper == "cloud" and needs_chunking(audio_path):
                click.echo("  Large file detected, will transcribe in chunks")

            # Transcribe
            click.echo(f"Transcribing ({whisper})...")

            def on_transcribe_progress(chunk: int, total: int) -> None:
                if total > 1:
                    click.echo(f"  Chunk {chunk}/{total}")

            subtitles = transcribe(
                audio_path,
                config,
                mode=whisper,
                language=source_lang,
                on_progress=on_transcribe_progress,
            )
            click.echo(f"  Transcribed {len(subtitles)} segments")

            # Save transcript
            click.echo(f"Saving transcript...")
            write_srt(subtitles, transcript_output)
            click.echo(f"  Saved to {transcript_output}")
            transcribed = True

        # Step 2: Translate
        click.echo(f"Translating ({llm})...")

        def on_translate_progress(batch: int, total: int) -> None:
            click.echo(f"  Batch {batch}/{total}")

        translated = translate(
            subtitles,
            config,
            source_lang=source_lang,
            target_lang=target_lang,
            provider=llm,
            ollama_model=ollama_model,
            openai_model=openai_model,
            on_progress=on_translate_progress,
        )

        # Step 3: Write output
        click.echo("Writing translated subtitles...")
        write_srt(translated, output)
        click.echo(f"  Saved to {output}")

        # Cleanup intermediate files on success (only if --cleanup)
        if cleanup:
            if extracted_audio and audio_path:
                try:
                    os.unlink(audio_path)
                    click.echo(f"  Cleaned up {audio_path}")
                except OSError:
                    pass
            if transcribed:
                try:
                    os.unlink(transcript_output)
                    click.echo(f"  Cleaned up {transcript_output}")
                except OSError:
                    pass

        click.echo()
        click.secho("Done!", fg="green", bold=True)

    except Exception as e:
        click.secho(f"Error: {e}", fg="red", err=True)
        click.echo()

        # Build LLM args for retry command
        llm_args = f"--llm {llm}"
        if llm == "openai":
            llm_args += f" --openai-model {openai_model}"
        elif llm == "ollama":
            llm_args += f" --ollama-model {ollama_model}"

        # Show retry hints
        if transcribed:
            click.echo(f"Transcript saved: {transcript_output}")
            click.echo(f"Retry translation: autotranslate {transcript_output} --transcript --from {source_lang} --to {target_lang} {llm_args}")
        elif extracted_audio and audio_path:
            click.echo(f"Audio saved: {audio_path}")
            whisper_arg = f"--whisper {whisper}"
            boost_arg = f" --boost {boost}" if boost else ""
            click.echo(f"Retry from audio: autotranslate {audio_path} --audio --from {source_lang} --to {target_lang} {whisper_arg}{boost_arg} {llm_args}")

        sys.exit(1)


if __name__ == "__main__":
    main()

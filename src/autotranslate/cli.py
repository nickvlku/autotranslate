"""CLI entry point for autotranslate."""

import json
import os
import sys
from pathlib import Path

import click

from .audio import extract_audio, needs_chunking
from .config import Config
from .mux import embed_subtitles
from .srt import read_srt, write_srt
from .transcribe import transcribe
from .translate import translate, get_language_name


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
    type=click.Choice(["deepseek", "openai", "openrouter", "ollama"]),
    default="deepseek",
    help="LLM provider for translation",
)
@click.option(
    "--whisper",
    type=click.Choice(["openai", "groq", "openrouter", "local", "whisper-cli", "qwen-asr"]),
    default="openai",
    help="Transcription provider (openai, groq, openrouter, local, whisper-cli, qwen-asr)",
)
@click.option(
    "--whisper-model",
    default=None,
    help="Whisper model: an API model name (e.g. whisper-1, large-v3); for "
    "--whisper whisper-cli a ggml model file path (e.g. ./ggml-large-v3.bin); "
    "for --whisper qwen-asr an ASR repo id (default Qwen/Qwen3-ASR-1.7B)",
)
@click.option(
    "--vad-model",
    default=None,
    help="whisper-cli only: VAD model file path (e.g. ./ggml-silero-v6.2.0.bin); "
    "enables VAD when provided",
)
@click.option(
    "--vad-threshold",
    type=float,
    default=0.5,
    help="whisper-cli only: VAD threshold for speech recognition (0.0-1.0, default 0.5)",
)
@click.option(
    "--vad-min-speech-duration-ms",
    type=float,
    default=250,
    help="whisper-cli only: VAD min speech duration in ms (default 250)",
)
@click.option(
    "--vad-min-silence-duration-ms",
    type=float,
    default=100,
    help="whisper-cli only: VAD min silence duration in ms (default 100)",
)
@click.option(
    "--vad-max-speech-duration-s",
    type=float,
    default=0,
    help="whisper-cli only: VAD max speech duration in seconds before auto-split (0 = unlimited)",
)
@click.option(
    "--vad-speech-pad-ms",
    type=float,
    default=30,
    help="whisper-cli only: VAD speech padding in ms (default 30)",
)
@click.option(
    "--vad-samples-overlap",
    type=float,
    default=0.1,
    help="whisper-cli only: VAD samples overlap in seconds (default 0.1)",
)
@click.option(
    "--max-subtitle-duration",
    type=float,
    default=0,
    help="Max subtitle duration in seconds (0 = unlimited). Subtitles exceeding this are dropped.",
)
@click.option(
    "--ollama-model",
    default="llama3.1:8b",
    help="Ollama model for translation. Comma-separate multiple models to "
    "bake off (one output SRT / subtitle track each).",
)
@click.option(
    "--openai-model",
    default="gpt-4o",
    help="OpenAI model for translation (gpt-5.2-chat-latest, gpt-5.2, gpt-4o). "
    "Comma-separate multiple models to bake off (one output SRT / track each).",
)
@click.option(
    "--openai-base-url",
    default=None,
    help="Override the OpenAI-compatible API base URL for the openai provider "
    "(translation and transcription), e.g. http://localhost:1234/v1 for LM "
    "Studio. Takes precedence over the OPENAI_BASE_URL env var.",
)
@click.option(
    "--openai-extra-body",
    default=None,
    help="Extra JSON merged into openai translation request bodies, for "
    "OpenAI-compatible servers. E.g. to disable thinking on models that "
    'support it: \'{"chat_template_kwargs": {"enable_thinking": false}}\'',
)
@click.option(
    "--openrouter-model",
    default="anthropic/claude-3.5-sonnet",
    help="OpenRouter model for translation (e.g., anthropic/claude-3.5-sonnet, "
    "openai/gpt-4o). Comma-separate multiple models to bake off (one SRT / track each).",
)
@click.option(
    "--cleanup",
    is_flag=True,
    help="Delete intermediate files (audio, transcript) on success",
)
@click.option(
    "--embed",
    is_flag=True,
    help="Embed the generated subtitle track(s) into the source video as a new "
    "<name>.subbed.<ext> file (soft subs, no re-encode). Saved after each model "
    "(atomic replace), so a bake-off never loses finished tracks. Video input only.",
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
    whisper_model: str | None,
    vad_model: str | None,
    vad_threshold: float,
    vad_min_speech_duration_ms: float,
    vad_min_silence_duration_ms: float,
    vad_max_speech_duration_s: float,
    vad_speech_pad_ms: float,
    vad_samples_overlap: float,
    max_subtitle_duration: float,
    ollama_model: str,
    openai_model: str,
    openai_base_url: str | None,
    openai_extra_body: str | None,
    openrouter_model: str,
    cleanup: bool,
    embed: bool,
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

    # CLI flag overrides the OPENAI_BASE_URL env var (which from_env already read)
    if openai_base_url:
        config.openai_base_url = openai_base_url

    if openai_extra_body:
        try:
            config.openai_extra_body = json.loads(openai_extra_body)
        except json.JSONDecodeError as e:
            raise click.ClickException(f"--openai-extra-body must be valid JSON: {e}")
        if not isinstance(config.openai_extra_body, dict):
            raise click.ClickException("--openai-extra-body must be a JSON object")

    # Validate flags
    if audio and transcript:
        raise click.ClickException("Cannot use both --audio and --transcript")

    if embed and (audio or transcript):
        raise click.ClickException(
            "--embed requires a video input (not --audio or --transcript), "
            "since it muxes the subtitles back into the source video."
        )

    # Validate API keys for selected providers
    if not transcript and whisper == "openai" and not config.has_openai():
        raise click.ClickException(
            "OPENAI_API_KEY environment variable required for OpenAI Whisper. "
            "Use --whisper local for offline mode."
        )

    if not transcript and whisper == "groq" and not config.has_groq():
        raise click.ClickException(
            "GROQ_API_KEY environment variable required for Groq Whisper. "
            "Use --whisper local for offline mode."
        )

    if not transcript and whisper == "openrouter" and not config.has_openrouter():
        raise click.ClickException(
            "OPENROUTER_API_KEY environment variable required for OpenRouter transcription. "
            "Use --whisper local for offline mode."
        )

    if not transcript and whisper == "whisper-cli" and not whisper_model:
        raise click.ClickException(
            "--whisper-model is required for whisper-cli: pass the ggml model "
            "file path, e.g. --whisper-model /path/to/ggml-large-v3.bin"
        )

    vad_violations = []
    if vad_model and whisper != "whisper-cli":
        vad_violations.append("--vad-model")
    if vad_threshold != 0.5 and whisper != "whisper-cli":
        vad_violations.append("--vad-threshold")
    if vad_min_speech_duration_ms != 250 and whisper != "whisper-cli":
        vad_violations.append("--vad-min-speech-duration-ms")
    if vad_min_silence_duration_ms != 100 and whisper != "whisper-cli":
        vad_violations.append("--vad-min-silence-duration-ms")
    if vad_max_speech_duration_s != 0 and whisper != "whisper-cli":
        vad_violations.append("--vad-max-speech-duration-s")
    if vad_speech_pad_ms != 30 and whisper != "whisper-cli":
        vad_violations.append("--vad-speech-pad-ms")
    if vad_samples_overlap != 0.1 and whisper != "whisper-cli":
        vad_violations.append("--vad-samples-overlap")

    if vad_violations:
        raise click.ClickException(
            f"{', '.join(vad_violations)} are only supported with --whisper whisper-cli"
        )

    if max_subtitle_duration > 0 and whisper != "whisper-cli":
        raise click.ClickException(
            "--max-subtitle-duration is only supported with --whisper whisper-cli"
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

    if llm == "openrouter" and not config.has_openrouter():
        raise click.ClickException(
            "OPENROUTER_API_KEY environment variable required for OpenRouter. "
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
    if whisper == "whisper-cli" and whisper_model:
        # Model is a ggml file path; use its basename without extension
        # (e.g. /models/ggml-large-v3.bin -> ggml-large-v3)
        whisper_model_name = Path(whisper_model).stem
    elif whisper_model:
        whisper_model_name = whisper_model.replace("/", "-")
    elif whisper == "openai":
        whisper_model_name = "whisper-1"
    elif whisper == "groq":
        whisper_model_name = "whisper-large-v3-turbo"
    elif whisper == "openrouter":
        whisper_model_name = "openai-whisper-1"
    elif whisper == "qwen-asr":
        whisper_model_name = "qwen3-asr-1.7b"
    else:
        whisper_model_name = "whisper-local"
    # Parse the active provider's model flag into a list. A comma-delimited
    # value runs the translation once per model (a "bake-off"); each model gets
    # its own output SRT (and, with --embed, its own subtitle track). Model
    # names may contain "/" (e.g. "google/gemma-3-4b"); we sanitize that to "-"
    # wherever a name is used in a filename or track title.
    if llm == "openai":
        llm_models = [m.strip() for m in openai_model.split(",") if m.strip()]
    elif llm == "openrouter":
        llm_models = [m.strip() for m in openrouter_model.split(",") if m.strip()]
    elif llm == "ollama":
        llm_models = [m.strip() for m in ollama_model.split(",") if m.strip()]
    else:
        llm_models = ["deepseek"]  # deepseek-chat (no model flag)

    if not llm_models:
        raise click.ClickException("No translation model specified")

    if output is not None and len(llm_models) > 1:
        raise click.ClickException(
            "--output/-o cannot be used with multiple translation models; "
            "each model is written to its own {name}.{model}.{lang}.srt"
        )

    # For a single model, fall back to the default output name (-o still wins).
    if len(llm_models) == 1 and output is None:
        output = f"{base_name}.{llm_models[0].replace('/', '-')}.{target_lang}.srt"

    transcript_output = f"{base_name}.{whisper_model_name}.{source_lang}.srt"

    click.echo(f"{input_type}: {input_path}")
    click.echo(f"Translation: {source_lang} → {target_lang}")
    if not transcript:
        click.echo(f"Whisper: {whisper}, LLM: {llm}")
    else:
        click.echo(f"LLM: {llm}")
    if len(llm_models) > 1:
        click.echo(f"Models: {', '.join(llm_models)}")
    else:
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
                provider=whisper,
                language=source_lang,
                model=whisper_model,
                vad_model=vad_model,
                vad_threshold=vad_threshold,
                vad_min_speech_duration_ms=vad_min_speech_duration_ms,
                vad_min_silence_duration_ms=vad_min_silence_duration_ms,
                vad_max_speech_duration_s=vad_max_speech_duration_s,
                vad_speech_pad_ms=vad_speech_pad_ms,
                vad_samples_overlap=vad_samples_overlap,
                on_progress=on_transcribe_progress,
            )
            click.echo(f"  Transcribed {len(subtitles)} segments")

            # Save transcript
            click.echo(f"Saving transcript...")
            write_srt(subtitles, transcript_output)
            click.echo(f"  Saved to {transcript_output}")
            transcribed = True

            # Filter subtitles by max duration
            if max_subtitle_duration > 0:
                filtered = [
                    sub for sub in subtitles
                    if (sub.end - sub.start) <= max_subtitle_duration
                ]
                dropped = len(subtitles) - len(filtered)
                if dropped > 0:
                    subtitles = filtered
                    click.echo(f"  Dropped {dropped} subtitles exceeding {max_subtitle_duration}s")
                    write_srt(subtitles, transcript_output)

        # Step 2/3: Translate with each model (bake-off) and write each SRT.
        # Progressive: if one model fails, warn and continue with the rest.
        def on_translate_progress(batch: int, total: int) -> None:
            click.echo(f"  Batch {batch}/{total}")

        multi = len(llm_models) > 1
        results = []  # (model_display_name, output_path) for each success
        subbed_output = f"{base_name}.subbed{input_p.suffix}" if embed else None

        def update_embedded_video() -> None:
            """(Re)build the embedded video from the original plus every
            successful translation so far and the transcript, atomically
            replacing the .subbed file. Run after each model so an unexpected
            failure never loses tracks already written. Best-effort: a mux
            failure warns and the bake-off continues."""
            tracks = [
                (path, f"{get_language_name(target_lang)} ({name})", target_lang)
                for (name, path) in results
            ]
            tracks.append(
                (transcript_output, f"{get_language_name(source_lang)} ({whisper_model_name})", source_lang)
            )
            n = len(tracks)
            click.echo(f"  Embedding into {subbed_output} ({n} track{'s' if n != 1 else ''})...")
            try:
                embed_subtitles(input_path, tracks, subbed_output)
                click.echo(f"    Saved {subbed_output}")
            except Exception as e:
                click.secho(f"    Subtitle embedding failed: {e}", fg="yellow", err=True)

        for model in llm_models:
            model_name = model.replace("/", "-")
            out_path = output if not multi else f"{base_name}.{model_name}.{target_lang}.srt"
            llm_kwargs = {}
            if llm == "openai":
                llm_kwargs["openai_model"] = model
            elif llm == "openrouter":
                llm_kwargs["openrouter_model"] = model
            elif llm == "ollama":
                llm_kwargs["ollama_model"] = model

            label = f" [{model}]" if multi else ""
            click.echo(f"Translating ({llm}){label}...")
            try:
                translated = translate(
                    subtitles,
                    config,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    provider=llm,
                    on_progress=on_translate_progress,
                    **llm_kwargs,
                )
                write_srt(translated, out_path)
                click.echo(f"  Saved to {out_path}")
                results.append((model_name, out_path))
                # Embed incrementally so each finished model is durably saved.
                if embed:
                    update_embedded_video()
            except Exception as e:
                if not multi:
                    raise  # single model: surface via the outer retry handler
                click.secho(f"  Model '{model}' failed: {e}", fg="yellow", err=True)

        if not results:
            raise RuntimeError("All translation models failed")

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
            if openai_base_url:
                llm_args += f" --openai-base-url {openai_base_url}"
            if openai_extra_body:
                llm_args += f" --openai-extra-body '{openai_extra_body}'"
        elif llm == "openrouter":
            llm_args += f" --openrouter-model {openrouter_model}"
        elif llm == "ollama":
            llm_args += f" --ollama-model {ollama_model}"

        # Show retry hints
        if transcribed:
            click.echo(f"Transcript saved: {transcript_output}")
            click.echo(f"Retry translation: autotranslate {transcript_output} --transcript --from {source_lang} --to {target_lang} {llm_args}")
        elif extracted_audio and audio_path:
            click.echo(f"Audio saved: {audio_path}")
            whisper_arg = f"--whisper {whisper}"
            whisper_model_arg = f" --whisper-model {whisper_model}" if whisper_model else ""
            vad_model_arg = f" --vad-model {vad_model}" if vad_model else ""
            vad_threshold_arg = f" --vad-threshold {vad_threshold}" if vad_threshold != 0.5 else ""
            vad_min_speech_arg = f" --vad-min-speech-duration-ms {vad_min_speech_duration_ms}" if vad_min_speech_duration_ms != 250 else ""
            vad_min_silence_arg = f" --vad-min-silence-duration-ms {vad_min_silence_duration_ms}" if vad_min_silence_duration_ms != 100 else ""
            vad_max_speech_arg = f" --vad-max-speech-duration-s {vad_max_speech_duration_s}" if vad_max_speech_duration_s != 0 else ""
            vad_pad_arg = f" --vad-speech-pad-ms {vad_speech_pad_ms}" if vad_speech_pad_ms != 30 else ""
            vad_overlap_arg = f" --vad-samples-overlap {vad_samples_overlap}" if vad_samples_overlap != 0.1 else ""
            max_dur_arg = f" --max-subtitle-duration {max_subtitle_duration}" if max_subtitle_duration > 0 else ""
            boost_arg = f" --boost {boost}" if boost else ""
            click.echo(f"Retry from audio: autotranslate {audio_path} --audio --from {source_lang} --to {target_lang} {whisper_arg}{whisper_model_arg}{vad_model_arg}{vad_threshold_arg}{vad_min_speech_arg}{vad_min_silence_arg}{vad_max_speech_arg}{vad_pad_arg}{vad_overlap_arg}{max_dur_arg}{boost_arg} {llm_args}")

        sys.exit(1)


if __name__ == "__main__":
    main()

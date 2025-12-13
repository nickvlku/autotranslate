"""Audio transcription using Whisper (cloud and local)."""

import os
from pathlib import Path
from typing import Callable

from openai import OpenAI

from .audio import get_audio_duration, needs_chunking, split_audio
from .config import Config
from .models import Subtitle


def _transcribe_single_cloud(
    audio_path: Path,
    client: OpenAI,
    language: str | None = None,
) -> list[Subtitle]:
    """Transcribe a single audio file using OpenAI Whisper API."""
    with open(audio_path, "rb") as audio_file:
        kwargs = {
            "model": "whisper-1",
            "file": audio_file,
            "response_format": "verbose_json",
            "timestamp_granularities": ["segment"],
        }
        if language:
            kwargs["language"] = language

        response = client.audio.transcriptions.create(**kwargs)

    subtitles = []
    for i, segment in enumerate(response.segments, start=1):
        subtitles.append(
            Subtitle(
                index=i,
                start=segment.start,
                end=segment.end,
                text=segment.text.strip(),
            )
        )

    return subtitles


def transcribe_cloud(
    audio_path: str | Path,
    config: Config,
    language: str | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[Subtitle]:
    """Transcribe audio using OpenAI Whisper API.

    Automatically chunks large files that exceed the 25MB API limit.

    Args:
        audio_path: Path to the audio file
        config: Application configuration
        language: Optional source language code (e.g., "ja", "en")
        on_progress: Optional callback(chunk_num, total_chunks)

    Returns:
        List of Subtitle objects with timestamps
    """
    if not config.openai_api_key:
        raise RuntimeError("OpenAI API key not configured")

    client = OpenAI(api_key=config.openai_api_key)
    audio_path = Path(audio_path)

    # Check if we need to chunk the file
    if not needs_chunking(audio_path):
        if on_progress:
            on_progress(1, 1)
        return _transcribe_single_cloud(audio_path, client, language)

    # Split into chunks and transcribe each
    chunks = split_audio(audio_path)
    all_subtitles = []
    time_offset = 0.0
    subtitle_index = 1

    try:
        for i, chunk_path in enumerate(chunks):
            if on_progress:
                on_progress(i + 1, len(chunks))

            chunk_subs = _transcribe_single_cloud(chunk_path, client, language)

            # Adjust timestamps and indices for this chunk
            for sub in chunk_subs:
                all_subtitles.append(
                    Subtitle(
                        index=subtitle_index,
                        start=sub.start + time_offset,
                        end=sub.end + time_offset,
                        text=sub.text,
                    )
                )
                subtitle_index += 1

            # Update time offset for next chunk
            chunk_duration = get_audio_duration(chunk_path)
            time_offset += chunk_duration

    finally:
        # Clean up chunk files (but not the original)
        for chunk_path in chunks:
            if chunk_path != audio_path:
                try:
                    os.unlink(chunk_path)
                except OSError:
                    pass

    return all_subtitles


def transcribe_local(
    audio_path: str | Path,
    language: str | None = None,
    model_size: str = "large-v3",
) -> list[Subtitle]:
    """Transcribe audio using local faster-whisper.

    Args:
        audio_path: Path to the audio file
        language: Optional source language code (e.g., "ja", "en")
        model_size: Whisper model size (tiny, base, small, medium, large-v3)

    Returns:
        List of Subtitle objects with timestamps
    """
    from faster_whisper import WhisperModel

    # Use CPU by default, auto-detect if CUDA available
    model = WhisperModel(model_size, device="auto", compute_type="auto")

    segments, info = model.transcribe(
        str(audio_path),
        language=language,
        vad_filter=True,  # Filter out non-speech
    )

    subtitles = []
    for i, segment in enumerate(segments, start=1):
        subtitles.append(
            Subtitle(
                index=i,
                start=segment.start,
                end=segment.end,
                text=segment.text.strip(),
            )
        )

    return subtitles


def transcribe(
    audio_path: str | Path,
    config: Config,
    mode: str = "cloud",
    language: str | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[Subtitle]:
    """Transcribe audio using specified mode.

    Args:
        audio_path: Path to the audio file
        config: Application configuration
        mode: "cloud" for OpenAI API, "local" for faster-whisper
        language: Optional source language code
        on_progress: Optional callback(chunk_num, total_chunks) for cloud mode

    Returns:
        List of Subtitle objects with timestamps
    """
    if mode == "cloud":
        return transcribe_cloud(audio_path, config, language, on_progress)
    elif mode == "local":
        return transcribe_local(audio_path, language)
    else:
        raise ValueError(f"Unknown transcription mode: {mode}")

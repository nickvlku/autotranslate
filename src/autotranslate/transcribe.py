"""Audio transcription using Whisper and chat-based audio models."""

import base64
import json
import os
import re
from pathlib import Path
from typing import Callable

from openai import OpenAI

from .audio import get_audio_duration, needs_chunking, split_audio
from .config import Config
from .models import Subtitle


# Models that use the Whisper transcription API (not chat completions)
WHISPER_API_MODELS = {
    "whisper-1",
    "whisper-large-v3",
    "whisper-large-v3-turbo",
    "distil-whisper-large-v3-en",
}


def _is_whisper_model(model: str) -> bool:
    """Check if a model uses the Whisper transcription API."""
    model_lower = model.lower()
    # Check exact match or if it contains "whisper"
    return model in WHISPER_API_MODELS or "whisper" in model_lower


TRANSCRIPTION_SYSTEM_PROMPT = """You are a professional audio transcriber. Transcribe the audio accurately with timestamps.

Return ONLY a JSON array with segments in this exact format:
[{{"start": 0.0, "end": 2.5, "text": "First sentence here."}}, {{"start": 2.5, "end": 5.0, "text": "Second sentence."}}]

Rules:
1. Timestamps must be in seconds (floats)
2. Break into natural sentence/phrase segments (2-6 seconds each)
3. Transcribe exactly what is said, including filler words
4. Return ONLY the JSON array, no other text"""


def _parse_chat_transcription(response_text: str) -> list[Subtitle]:
    """Parse JSON response from chat-based transcription."""
    # Try to extract JSON from the response
    text = response_text.strip()

    # Remove markdown code blocks if present
    text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s*```$', '', text, flags=re.MULTILINE)
    text = text.strip()

    # Find JSON array
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        text = match.group()

    try:
        segments = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse transcription response as JSON: {e}\nResponse: {response_text[:500]}")

    subtitles = []
    for i, seg in enumerate(segments, start=1):
        subtitles.append(
            Subtitle(
                index=i,
                start=float(seg["start"]),
                end=float(seg["end"]),
                text=seg["text"].strip(),
            )
        )

    return subtitles


def _get_audio_mime_type(audio_path: Path) -> str:
    """Get MIME type for audio file."""
    suffix = audio_path.suffix.lower()
    mime_types = {
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".m4a": "audio/mp4",
        ".ogg": "audio/ogg",
        ".flac": "audio/flac",
        ".webm": "audio/webm",
    }
    return mime_types.get(suffix, "audio/mpeg")


def _transcribe_single_chat(
    audio_path: Path,
    client: OpenAI,
    model: str,
    language: str | None = None,
) -> list[Subtitle]:
    """Transcribe audio using chat completions API with audio input."""
    # Read and base64 encode the audio
    with open(audio_path, "rb") as f:
        audio_data = base64.standard_b64encode(f.read()).decode("utf-8")

    mime_type = _get_audio_mime_type(audio_path)

    # Build the user message with audio
    user_content = [
        {
            "type": "input_audio",
            "input_audio": {
                "data": audio_data,
                "format": audio_path.suffix.lstrip(".") or "mp3",
            },
        },
    ]

    # Add language hint if provided
    if language:
        user_content.append({
            "type": "text",
            "text": f"The audio is in {language}. Transcribe it accurately.",
        })
    else:
        user_content.append({
            "type": "text",
            "text": "Transcribe this audio accurately.",
        })

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": TRANSCRIPTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.0,
    )

    result_text = response.choices[0].message.content
    return _parse_chat_transcription(result_text)


def _transcribe_chat_with_chunking(
    audio_path: Path,
    client: OpenAI,
    model: str,
    language: str | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[Subtitle]:
    """Transcribe audio using chat API with automatic chunking."""
    # Check if we need to chunk the file
    if not needs_chunking(audio_path):
        if on_progress:
            on_progress(1, 1)
        return _transcribe_single_chat(audio_path, client, model, language)

    # Split into chunks and transcribe each
    chunks = split_audio(audio_path)
    all_subtitles = []
    time_offset = 0.0
    subtitle_index = 1

    try:
        for i, chunk_path in enumerate(chunks):
            if on_progress:
                on_progress(i + 1, len(chunks))

            chunk_subs = _transcribe_single_chat(chunk_path, client, model, language)

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
        # Clean up chunk files
        for chunk_path in chunks:
            if chunk_path != audio_path:
                try:
                    os.unlink(chunk_path)
                except OSError:
                    pass

    return all_subtitles


def _transcribe_single_cloud(
    audio_path: Path,
    client: OpenAI,
    language: str | None = None,
    model: str = "whisper-1",
) -> list[Subtitle]:
    """Transcribe a single audio file using OpenAI-compatible API."""
    with open(audio_path, "rb") as audio_file:
        kwargs = {
            "model": model,
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


def _transcribe_with_chunking(
    audio_path: Path,
    client: OpenAI,
    model: str,
    language: str | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[Subtitle]:
    """Transcribe audio with automatic chunking for large files."""
    # Check if we need to chunk the file
    if not needs_chunking(audio_path):
        if on_progress:
            on_progress(1, 1)
        return _transcribe_single_cloud(audio_path, client, language, model)

    # Split into chunks and transcribe each
    chunks = split_audio(audio_path)
    all_subtitles = []
    time_offset = 0.0
    subtitle_index = 1

    try:
        for i, chunk_path in enumerate(chunks):
            if on_progress:
                on_progress(i + 1, len(chunks))

            chunk_subs = _transcribe_single_cloud(chunk_path, client, language, model)

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


def transcribe_openai(
    audio_path: str | Path,
    config: Config,
    language: str | None = None,
    model: str = "whisper-1",
    on_progress: Callable[[int, int], None] | None = None,
) -> list[Subtitle]:
    """Transcribe audio using OpenAI API (Whisper or chat-based audio models)."""
    if not config.openai_api_key:
        raise RuntimeError("OpenAI API key not configured")

    client = OpenAI(api_key=config.openai_api_key)

    if _is_whisper_model(model):
        return _transcribe_with_chunking(Path(audio_path), client, model, language, on_progress)
    else:
        # Use chat-based transcription for models like gpt-4o-audio-preview
        return _transcribe_chat_with_chunking(Path(audio_path), client, model, language, on_progress)


def transcribe_groq(
    audio_path: str | Path,
    config: Config,
    language: str | None = None,
    model: str = "whisper-large-v3-turbo",
    on_progress: Callable[[int, int], None] | None = None,
) -> list[Subtitle]:
    """Transcribe audio using Groq Whisper API."""
    if not config.groq_api_key:
        raise RuntimeError("Groq API key not configured")

    client = OpenAI(api_key=config.groq_api_key, base_url=config.groq_base_url)
    return _transcribe_with_chunking(Path(audio_path), client, model, language, on_progress)


def transcribe_openrouter(
    audio_path: str | Path,
    config: Config,
    language: str | None = None,
    model: str = "openai/whisper-1",
    on_progress: Callable[[int, int], None] | None = None,
) -> list[Subtitle]:
    """Transcribe audio using OpenRouter API (Whisper or chat-based audio models)."""
    if not config.openrouter_api_key:
        raise RuntimeError("OpenRouter API key not configured")

    client = OpenAI(api_key=config.openrouter_api_key, base_url=config.openrouter_base_url)

    if _is_whisper_model(model):
        return _transcribe_with_chunking(Path(audio_path), client, model, language, on_progress)
    else:
        # Use chat-based transcription for audio-capable models
        return _transcribe_chat_with_chunking(Path(audio_path), client, model, language, on_progress)


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
    provider: str = "openai",
    language: str | None = None,
    model: str | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[Subtitle]:
    """Transcribe audio using specified provider.

    Args:
        audio_path: Path to the audio file
        config: Application configuration
        provider: "openai", "groq", "openrouter", or "local"
        language: Optional source language code
        model: Optional model name (provider-specific defaults if not specified)
        on_progress: Optional callback(chunk_num, total_chunks)

    Returns:
        List of Subtitle objects with timestamps
    """
    if provider == "openai":
        return transcribe_openai(
            audio_path, config, language, model or "whisper-1", on_progress
        )
    elif provider == "groq":
        return transcribe_groq(
            audio_path, config, language, model or "whisper-large-v3-turbo", on_progress
        )
    elif provider == "openrouter":
        return transcribe_openrouter(
            audio_path, config, language, model or "openai/whisper-1", on_progress
        )
    elif provider == "local":
        return transcribe_local(audio_path, language, model or "large-v3")
    else:
        raise ValueError(f"Unknown transcription provider: {provider}")

"""Audio transcription using Whisper and chat-based audio models."""

import base64
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable

from openai import OpenAI

from .audio import get_audio_duration, needs_chunking, split_audio
from .config import Config
from .models import Subtitle
from .srt import read_srt


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

    client = OpenAI(api_key=config.openai_api_key, base_url=config.openai_base_url)

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


def transcribe_whisper_cli(
    audio_path: str | Path,
    language: str | None = None,
    model: str | None = None,
    vad_model: str | None = None,
    vad_threshold: float = 0.5,
    vad_min_speech_duration_ms: float = 250,
    vad_min_silence_duration_ms: float = 100,
    vad_max_speech_duration_s: float = 0,
    vad_speech_pad_ms: float = 30,
    vad_samples_overlap: float = 0.1,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[Subtitle]:
    """Transcribe audio using the local whisper-cli binary (whisper.cpp).

    Shells out to ``whisper-cli``, which must be installed and on PATH. The
    model is a ggml model file passed via ``-m`` (e.g. ``ggml-large-v3.bin``).
    Output is written as SRT and parsed back into Subtitle objects, which
    preserves the per-segment timestamps. Runs locally, so there is no API
    size limit and no chunking.

    Args:
        audio_path: Path to the audio file (wav/mp3/ogg/flac)
        language: Optional source language code (e.g. "ja", "en"); "auto" detects
        model: Path to the ggml model file (required; passed to whisper-cli -m)
        vad_model: Optional path to a VAD model file (e.g. ggml-silero-*.bin);
            when given, enables Voice Activity Detection via "--vad --vad-model"
        vad_threshold: VAD threshold for speech recognition (0.0-1.0)
        vad_min_speech_duration_ms: VAD min speech duration in ms
        vad_min_silence_duration_ms: VAD min silence duration in ms
        vad_max_speech_duration_s: VAD max speech duration in seconds (0 = unlimited)
        vad_speech_pad_ms: VAD speech padding in ms
        vad_samples_overlap: VAD samples overlap in seconds
        on_progress: Optional callback(step, total); called once as (1, 1)

    Returns:
        List of Subtitle objects with timestamps

    Raises:
        RuntimeError: if the model is missing, whisper-cli is not on PATH, a
            model file does not exist, or transcription fails
    """
    if not model:
        raise RuntimeError(
            "whisper-cli requires a model file. Pass --whisper-model "
            "/path/to/ggml-<size>.bin (a whisper.cpp ggml model)."
        )

    binary = shutil.which("whisper-cli")
    if binary is None:
        raise RuntimeError(
            "whisper-cli not found on PATH. Install whisper.cpp and ensure the "
            "'whisper-cli' binary is available."
        )

    model_path = Path(model)
    if not model_path.is_file():
        raise RuntimeError(f"whisper-cli model file not found: {model_path}")

    vad_args: list[str] = []
    if vad_model:
        vad_model_path = Path(vad_model)
        if not vad_model_path.is_file():
            raise RuntimeError(f"whisper-cli VAD model file not found: {vad_model_path}")
        vad_args = [
            "--vad",
            "--vad-model", str(vad_model_path),
            "--vad-threshold", str(vad_threshold),
            "--vad-min-speech-duration-ms", str(vad_min_speech_duration_ms),
            "--vad-min-silence-duration-ms", str(vad_min_silence_duration_ms),
            "--vad-speech-pad-ms", str(vad_speech_pad_ms),
            "--vad-samples-overlap", str(vad_samples_overlap),
        ]
        if vad_max_speech_duration_s > 0:
            vad_args.extend(["--vad-max-speech-duration-s", str(vad_max_speech_duration_s)])

    if on_progress:
        on_progress(1, 1)

    # whisper-cli writes "<prefix>.srt"; use a temp prefix we control so the
    # SRT is cleaned up automatically once we've parsed it.
    audio_path = Path(audio_path)
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_prefix = Path(tmp_dir) / "transcript"
        cmd = [
            binary,
            "-m", str(model_path),
            "-f", str(audio_path),
            "-l", language or "auto",
            "-osrt",
            "-of", str(out_prefix),
            "-np",  # suppress prints other than results
            *vad_args,
        ]

        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"whisper-cli failed: {e.stderr or e.stdout}"
            ) from e

        srt_path = out_prefix.with_suffix(".srt")
        if not srt_path.is_file():
            raise RuntimeError(
                f"whisper-cli did not produce an SRT file at {srt_path}"
            )
        return read_srt(srt_path)


# Languages the Qwen3 forced aligner supports, keyed by 2-letter source code.
# Unknown codes fall back to None (auto-detect).
QWEN_LANGUAGES = {
    "zh": "Chinese", "en": "English", "yue": "Cantonese", "fr": "French",
    "de": "German", "it": "Italian", "ja": "Japanese", "ko": "Korean",
    "pt": "Portuguese", "ru": "Russian", "es": "Spanish",
}

# Characters that end a subtitle cue (CJK + Latin sentence terminators).
_SENTENCE_END = set("。！？!?.…")


def _group_aligned_items(
    items: list,
    full_text: str,
    max_cue_duration: float,
    max_cue_chars: int,
) -> list[tuple[str, float, float]]:
    """Group word/morpheme-level aligned items into subtitle cues.

    The Qwen forced aligner returns fine-grained items (text + start/end), and
    ``full_text`` is the transcript including punctuation that the items omit.
    We map each item back to its span in ``full_text`` to recover trailing
    punctuation, then close a cue on a sentence terminator OR when the cue hits
    the duration / length cap (hybrid grouping).

    Returns a list of ``(text, start, end)`` cues.
    """
    # Locate each item's text in full_text so we can include trailing punctuation.
    spans: list[tuple[int, int] | None] = []
    cursor = 0
    for it in items:
        idx = full_text.find(it.text, cursor) if it.text else -1
        if idx == -1:
            spans.append(None)
        else:
            spans.append((idx, idx + len(it.text)))
            cursor = idx + len(it.text)

    n = len(items)
    cues: list[tuple[str, float, float]] = []
    cur_text = ""
    cur_start: float | None = None
    cur_end: float | None = None

    for i, it in enumerate(items):
        if cur_start is None:
            cur_start = it.start_time

        # Display segment: from this item's start to the next located item's
        # start, so trailing punctuation/spaces stay attached to this item.
        if spans[i] is not None:
            seg_start = spans[i][0]
            seg_end = len(full_text)
            for j in range(i + 1, n):
                if spans[j] is not None:
                    seg_end = spans[j][0]
                    break
            seg = full_text[seg_start:seg_end]
        else:
            seg = it.text or ""

        cur_text += seg
        cur_end = it.end_time

        ends_sentence = any(c in _SENTENCE_END for c in seg)
        too_long = (
            (cur_end - cur_start) >= max_cue_duration
            or len(cur_text.strip()) >= max_cue_chars
        )
        if ends_sentence or too_long:
            text = cur_text.strip()
            if text:
                cues.append((text, cur_start, cur_end))
            cur_text = ""
            cur_start = None
            cur_end = None

    if cur_text.strip() and cur_start is not None:
        cues.append((cur_text.strip(), cur_start, cur_end))

    return cues


def transcribe_qwen_asr(
    audio_path: str | Path,
    language: str | None = None,
    model: str | None = None,
    aligner: str = "Qwen/Qwen3-ForcedAligner-0.6B",
    on_progress: Callable[[int, int], None] | None = None,
    max_cue_duration: float = 6.0,
    max_cue_chars: int = 40,
) -> list[Subtitle]:
    """Transcribe audio locally with Qwen3-ASR + the Qwen3 forced aligner.

    Qwen3-ASR produces only a flat transcript, so the forced aligner supplies
    word/character-level timestamps, which we group into subtitle cues. Runs on
    Apple Silicon (MPS) or CPU. The aligner caps at ~5 minutes, so audio is
    split into <5-minute chunks and timestamps are offset per chunk.

    Args:
        audio_path: Path to the audio file (wav/mp3/flac/...)
        language: Optional source language code (e.g. "ja"); mapped to the
            aligner's language name, or auto-detected if unknown/None
        model: ASR model repo id (default Qwen/Qwen3-ASR-1.7B)
        aligner: Forced aligner repo id (provides timestamps)
        on_progress: Optional callback(chunk_num, total_chunks)
        max_cue_duration: Max seconds per subtitle cue before forcing a split
        max_cue_chars: Max characters per cue before forcing a split

    Returns:
        List of Subtitle objects with timestamps

    Raises:
        RuntimeError: if the qwen-asr package is not installed or inference fails
    """
    try:
        import torch
        from qwen_asr import Qwen3ASRModel
    except ImportError as e:
        raise RuntimeError(
            "qwen-asr is not installed. Install the optional extra with "
            "`pip install 'autotranslate[qwen]'` (or `pip install qwen-asr`)."
        ) from e

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32

    asr = Qwen3ASRModel.from_pretrained(
        model or "Qwen/Qwen3-ASR-1.7B",
        forced_aligner=aligner,
        forced_aligner_kwargs={"dtype": dtype, "device_map": device},
        max_new_tokens=1024,
        dtype=dtype,
        device_map=device,
    )

    lang_name = QWEN_LANGUAGES.get(language.lower()) if language else None

    # Keep each chunk under the aligner's ~5-minute limit.
    chunks = split_audio(audio_path, chunk_duration=270.0)
    subtitles: list[Subtitle] = []
    time_offset = 0.0
    index = 1

    try:
        for i, chunk in enumerate(chunks):
            if on_progress:
                on_progress(i + 1, len(chunks))

            result = asr.transcribe(
                str(chunk), language=lang_name, return_time_stamps=True
            )[0]
            items = list(result.time_stamps) if result.time_stamps is not None else []
            cues = _group_aligned_items(
                items, result.text or "", max_cue_duration, max_cue_chars
            )
            for text, start, end in cues:
                subtitles.append(
                    Subtitle(
                        index=index,
                        start=start + time_offset,
                        end=end + time_offset,
                        text=text,
                    )
                )
                index += 1

            time_offset += get_audio_duration(chunk)
    finally:
        for chunk in chunks:
            if Path(chunk) != Path(audio_path):
                try:
                    os.unlink(chunk)
                except OSError:
                    pass

    return subtitles


def transcribe(
    audio_path: str | Path,
    config: Config,
    provider: str = "openai",
    language: str | None = None,
    model: str | None = None,
    vad_model: str | None = None,
    vad_threshold: float = 0.5,
    vad_min_speech_duration_ms: float = 250,
    vad_min_silence_duration_ms: float = 100,
    vad_max_speech_duration_s: float = 0,
    vad_speech_pad_ms: float = 30,
    vad_samples_overlap: float = 0.1,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[Subtitle]:
    """Transcribe audio using specified provider.

    Args:
        audio_path: Path to the audio file
        config: Application configuration
        provider: "openai", "groq", "openrouter", "local", "whisper-cli", or "qwen-asr"
        language: Optional source language code
        model: Optional model name (provider-specific defaults if not specified)
        vad_model: Optional VAD model file path (whisper-cli provider only)
        vad_threshold: VAD threshold (whisper-cli only)
        vad_min_speech_duration_ms: VAD min speech duration ms (whisper-cli only)
        vad_min_silence_duration_ms: VAD min silence duration ms (whisper-cli only)
        vad_max_speech_duration_s: VAD max speech duration s (whisper-cli only)
        vad_speech_pad_ms: VAD speech padding ms (whisper-cli only)
        vad_samples_overlap: VAD samples overlap s (whisper-cli only)
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
    elif provider == "whisper-cli":
        return transcribe_whisper_cli(
            audio_path, language, model, vad_model,
            vad_threshold, vad_min_speech_duration_ms,
            vad_min_silence_duration_ms, vad_max_speech_duration_s,
            vad_speech_pad_ms, vad_samples_overlap,
            on_progress,
        )
    elif provider == "qwen-asr":
        return transcribe_qwen_asr(audio_path, language, model, on_progress=on_progress)
    else:
        raise ValueError(f"Unknown transcription provider: {provider}")

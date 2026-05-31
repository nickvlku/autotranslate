"""Embed generated subtitles into a video file as soft subtitle tracks (ffmpeg)."""

import os
import subprocess
from pathlib import Path

# 2-letter (ISO 639-1) -> 3-letter (ISO 639-2) for the subtitle "language" tag.
_LANG_3 = {
    "en": "eng", "ja": "jpn", "zh": "chi", "ko": "kor", "es": "spa",
    "fr": "fre", "de": "ger", "it": "ita", "pt": "por", "ru": "rus",
    "ar": "ara", "hi": "hin", "th": "tha", "vi": "vie",
}

# Subtitle codec per container, keyed by the output file extension.
_SUB_CODEC = {
    ".mp4": "mov_text", ".m4v": "mov_text", ".mov": "mov_text",
    ".mkv": "srt", ".webm": "webvtt",
}


def _lang3(code: str) -> str:
    """Map a 2-letter language code to ISO 639-2 (falls back to 'und')."""
    return _LANG_3.get(code.lower(), "und")


def embed_subtitles(
    video_path: str | Path,
    tracks: list[tuple[str | Path, str, str]],
    output_path: str | Path,
) -> Path:
    """Mux subtitle tracks into a copy of a video as soft subtitle tracks.

    Copies the video and audio streams without re-encoding and adds one soft
    subtitle track per entry in ``tracks``. Subtitle tracks already present in
    the source are not carried over.

    Args:
        video_path: Path to the source video.
        tracks: List of ``(srt_path, title, language_code)`` tuples, in the
            order the tracks should appear. ``language_code`` is a 2-letter code
            (e.g. "ja"), mapped to ISO 639-2 for the track's language metadata.
        output_path: Path to write the muxed video. Its extension selects the
            subtitle codec (mp4/m4v/mov -> mov_text, mkv -> srt, webm -> webvtt).

    Returns:
        Path to the written video.

    Raises:
        RuntimeError: if there are no tracks, the container is unsupported,
            ffmpeg is missing, or the mux fails.
    """
    video_path = Path(video_path)
    output_path = Path(output_path)

    if not tracks:
        raise RuntimeError("No subtitle tracks to embed")

    codec = _SUB_CODEC.get(output_path.suffix.lower())
    if codec is None:
        raise RuntimeError(
            f"Cannot embed subtitles into '{output_path.suffix}' files. "
            "Supported containers: .mp4, .m4v, .mov, .mkv, .webm"
        )

    # Write to a temp file in the same directory, then atomically replace the
    # destination. An interrupted or failed mux therefore never corrupts an
    # existing output (important when re-muxing repeatedly to add tracks).
    tmp_out = output_path.parent / f"{output_path.stem}.__partial__{output_path.suffix}"

    cmd = ["ffmpeg", "-i", str(video_path)]
    for srt_path, _title, _lang in tracks:
        cmd += ["-i", str(srt_path)]

    # Keep all video/audio from the source ("?" so missing audio is not an
    # error); drop any pre-existing subtitle tracks; then add our subtitles.
    cmd += ["-map", "0:v?", "-map", "0:a?"]
    for i in range(len(tracks)):
        cmd += ["-map", str(i + 1)]

    # Copy video/audio untouched; only the (text) subtitle streams are encoded.
    cmd += ["-c", "copy", "-c:s", codec]

    for i, (_srt, title, lang) in enumerate(tracks):
        cmd += [
            f"-metadata:s:s:{i}", f"title={title}",
            f"-metadata:s:s:{i}", f"language={_lang3(lang)}",
        ]

    cmd += ["-y", str(tmp_out)]

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        tmp_out.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg failed to embed subtitles: {e.stderr}") from e
    except FileNotFoundError:
        tmp_out.unlink(missing_ok=True)
        raise RuntimeError("ffmpeg not found. Please install ffmpeg.")

    os.replace(tmp_out, output_path)
    return output_path

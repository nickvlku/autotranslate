"""Audio extraction from video files using ffmpeg."""

import os
import subprocess
import tempfile
from pathlib import Path

# OpenAI Whisper API limit is 25MB, use 24MB to be safe
MAX_AUDIO_SIZE_BYTES = 24 * 1024 * 1024


def get_audio_duration(audio_path: str | Path) -> float:
    """Get audio duration in seconds using ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def split_audio(audio_path: str | Path, chunk_duration: float = 600) -> list[Path]:
    """Split audio file into chunks of specified duration.

    Args:
        audio_path: Path to the audio file
        chunk_duration: Maximum duration per chunk in seconds (default 10 min)

    Returns:
        List of paths to chunk files
    """
    audio_path = Path(audio_path)
    total_duration = get_audio_duration(audio_path)

    if total_duration <= chunk_duration:
        return [audio_path]

    chunks = []
    start = 0.0
    chunk_num = 0

    while start < total_duration:
        chunk_file = tempfile.NamedTemporaryFile(
            suffix=f"_chunk{chunk_num}.wav", delete=False
        )
        chunk_path = Path(chunk_file.name)
        chunk_file.close()

        cmd = [
            "ffmpeg",
            "-i", str(audio_path),
            "-ss", str(start),
            "-t", str(chunk_duration),
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            "-y",
            str(chunk_path),
        ]
        subprocess.run(cmd, capture_output=True, check=True)

        chunks.append(chunk_path)
        start += chunk_duration
        chunk_num += 1

    return chunks


def needs_chunking(audio_path: str | Path) -> bool:
    """Check if audio file exceeds the size limit for Whisper API."""
    return os.path.getsize(audio_path) > MAX_AUDIO_SIZE_BYTES


def extract_audio(
    video_path: str | Path,
    output_path: str | Path | None = None,
    boost_db: float | None = None,
) -> Path:
    """Extract audio track from video file as WAV.

    Args:
        video_path: Path to the input video file
        output_path: Optional output path. If None, creates a temp file.
        boost_db: Optional audio boost in decibels (e.g., 10 for +10dB)

    Returns:
        Path to the extracted audio file

    Raises:
        RuntimeError: If ffmpeg fails to extract audio
    """
    video_path = Path(video_path)

    if output_path is None:
        # Create temp file that persists until explicitly deleted
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        output_path = Path(temp_file.name)
        temp_file.close()
    else:
        output_path = Path(output_path)

    # Build audio filter chain
    audio_filters = []
    if boost_db is not None and boost_db != 0:
        audio_filters.append(f"volume={boost_db}dB")

    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-vn",  # No video
        "-acodec", "pcm_s16le",  # PCM 16-bit
        "-ar", "16000",  # 16kHz sample rate (optimal for Whisper)
        "-ac", "1",  # Mono
    ]

    if audio_filters:
        cmd.extend(["-af", ",".join(audio_filters)])

    cmd.extend(["-y", str(output_path)])  # Overwrite output

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg failed: {e.stderr}") from e
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found. Please install ffmpeg.")

    return output_path


def get_video_duration(video_path: str | Path) -> float:
    """Get video duration in seconds using ffprobe.

    Args:
        video_path: Path to the video file

    Returns:
        Duration in seconds
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError) as e:
        raise RuntimeError(f"Failed to get video duration: {e}") from e

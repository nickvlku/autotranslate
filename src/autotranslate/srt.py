"""SRT subtitle file parsing and generation."""

import re
from pathlib import Path

from .models import Subtitle


def parse_timestamp(timestamp: str) -> float:
    """Parse SRT timestamp to seconds.

    Args:
        timestamp: SRT timestamp format "HH:MM:SS,mmm"

    Returns:
        Time in seconds
    """
    match = re.match(r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})", timestamp)
    if not match:
        raise ValueError(f"Invalid timestamp format: {timestamp}")

    hours, minutes, seconds, millis = match.groups()
    return (
        int(hours) * 3600
        + int(minutes) * 60
        + int(seconds)
        + int(millis) / 1000
    )


def parse_srt(content: str) -> list[Subtitle]:
    """Parse SRT content into Subtitle objects.

    Args:
        content: Raw SRT file content

    Returns:
        List of Subtitle objects
    """
    subtitles = []
    blocks = re.split(r"\n\n+", content.strip())

    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue

        try:
            index = int(lines[0])
            timestamp_line = lines[1]
            text = "\n".join(lines[2:])

            # Parse timestamps
            match = re.match(
                r"(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]\d{3})",
                timestamp_line,
            )
            if not match:
                continue

            start = parse_timestamp(match.group(1))
            end = parse_timestamp(match.group(2))

            subtitles.append(Subtitle(index=index, start=start, end=end, text=text))
        except (ValueError, IndexError):
            continue

    return subtitles


def read_srt(path: str | Path) -> list[Subtitle]:
    """Read and parse an SRT file.

    Args:
        path: Path to the SRT file

    Returns:
        List of Subtitle objects
    """
    path = Path(path)
    content = path.read_text(encoding="utf-8")
    return parse_srt(content)


def write_srt(subtitles: list[Subtitle], path: str | Path) -> None:
    """Write subtitles to an SRT file.

    Args:
        subtitles: List of Subtitle objects
        path: Output file path
    """
    path = Path(path)
    content = "\n".join(sub.to_srt_block() for sub in subtitles)
    path.write_text(content, encoding="utf-8")


def subtitles_to_srt(subtitles: list[Subtitle]) -> str:
    """Convert subtitles to SRT format string.

    Args:
        subtitles: List of Subtitle objects

    Returns:
        SRT formatted string
    """
    return "\n".join(sub.to_srt_block() for sub in subtitles)

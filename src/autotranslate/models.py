"""Data models for autotranslate."""

from pydantic import BaseModel


class Subtitle(BaseModel):
    """A single subtitle entry with timing and text."""

    index: int
    start: float  # seconds
    end: float  # seconds
    text: str

    def format_timestamp(self, seconds: float) -> str:
        """Format seconds as SRT timestamp (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    def to_srt_block(self) -> str:
        """Convert to SRT format block."""
        start_ts = self.format_timestamp(self.start)
        end_ts = self.format_timestamp(self.end)
        return f"{self.index}\n{start_ts} --> {end_ts}\n{self.text}\n"

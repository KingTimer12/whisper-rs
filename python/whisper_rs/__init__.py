"""Whisper transcription with a Rust core.

The heavy lifting (decoding, VAD, windowing, inference) happens in Rust.
"""

from .whisper_rs import (
    Segment,
    SegmentIterator,
    TranscriptionInfo,
    WhisperModel,
    Word,
)

__all__ = [
    "Segment",
    "SegmentIterator",
    "TranscriptionInfo",
    "WhisperModel",
    "Word",
]

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

try:
    from .whisper_rs import NemotronModel
except ImportError:
    # Built without the `nemotron` Cargo feature: the class does not exist
    # in this wheel, and `whisper_rs.NemotronModel` correctly raises
    # AttributeError rather than the package failing to import at all.
    pass
else:
    __all__.append("NemotronModel")

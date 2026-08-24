"""Tests over the built wheel. Run with: pytest tests/python -v

Tests marked `model` download the tiny model on first run.
"""

import math
import struct
import wave
from pathlib import Path

import pytest

import whisper_rs


def write_speechlike_wav(path: Path, secs: float = 3.0, rate: int = 16_000) -> Path:
    """A tone burst surrounded by silence. Not speech, but valid audio."""
    frames = bytearray()
    total = int(rate * secs)
    for i in range(total):
        t = i / rate
        # silent for the first and last second
        amp = 0.4 if 1.0 <= t <= secs - 1.0 else 0.0
        value = int(amp * math.sin(t * 440.0 * math.tau) * 32767)
        frames += struct.pack("<h", value)

    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(bytes(frames))
    return path


def test_module_exports_the_public_classes():
    for name in ("WhisperModel", "SegmentIterator", "Segment", "Word", "TranscriptionInfo"):
        assert hasattr(whisper_rs, name), f"{name} is missing from the module"


def test_missing_model_raises_oserror():
    with pytest.raises(OSError):
        whisper_rs.WhisperModel("definitely-not-a-real-model", local_files_only=True)


@pytest.mark.model
def test_missing_audio_file_raises_valueerror():
    model = whisper_rs.WhisperModel("tiny")
    with pytest.raises(ValueError):
        model.transcribe("/nonexistent/audio.wav")


@pytest.mark.model
def test_unsupported_task_raises_valueerror(tmp_path):
    model = whisper_rs.WhisperModel("tiny")
    audio = write_speechlike_wav(tmp_path / "a.wav")
    with pytest.raises(ValueError, match="translate"):
        model.transcribe(str(audio), task="translate")


@pytest.mark.model
def test_unknown_vad_parameter_raises_valueerror(tmp_path):
    model = whisper_rs.WhisperModel("tiny")
    audio = write_speechlike_wav(tmp_path / "a.wav")
    with pytest.raises(ValueError, match="not_a_real_key"):
        model.transcribe(str(audio), vad_parameters={"not_a_real_key": 1})


@pytest.mark.model
def test_transcribe_returns_an_iterator_and_info(tmp_path):
    model = whisper_rs.WhisperModel("tiny")
    audio = write_speechlike_wav(tmp_path / "a.wav", secs=4.0)

    segments, info = model.transcribe(str(audio))

    assert isinstance(info, whisper_rs.TranscriptionInfo)
    assert info.duration == pytest.approx(4.0, abs=0.1)
    assert info.language_probability is None, "v1 never fabricates this value"
    assert info.duration_after_vad <= info.duration
    assert iter(segments) is segments, "the iterator must be self-iterable"


@pytest.mark.model
def test_segments_are_produced_lazily(tmp_path):
    """The generator must not decode until it is iterated."""
    import time

    model = whisper_rs.WhisperModel("tiny")
    audio = write_speechlike_wav(tmp_path / "a.wav", secs=4.0)

    start = time.perf_counter()
    segments, _ = model.transcribe(str(audio))
    setup = time.perf_counter() - start

    start = time.perf_counter()
    list(segments)
    consume = time.perf_counter() - start

    assert consume > setup, (
        f"consuming ({consume:.3f}s) should cost more than setup ({setup:.3f}s); "
        "decoding appears to be happening eagerly"
    )


@pytest.mark.model
def test_word_timestamps_are_absent_unless_requested(tmp_path):
    model = whisper_rs.WhisperModel("tiny")
    audio = write_speechlike_wav(tmp_path / "a.wav", secs=4.0)

    segments, _ = model.transcribe(str(audio), word_timestamps=False)
    for seg in segments:
        assert seg.words is None

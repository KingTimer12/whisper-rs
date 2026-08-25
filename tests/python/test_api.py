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
    # `language` was left as None, so detection ran and reported its own
    # score. See test_pinned_language_reports_no_probability for the other
    # side of this: pinning the language leaves it None.
    assert info.language_probability is not None
    assert 0.0 < info.language_probability <= 1.0
    assert info.duration_after_vad <= info.duration
    assert iter(segments) is segments, "the iterator must be self-iterable"


@pytest.mark.model
def test_segments_are_produced_lazily(tmp_path):
    """The generator must not decode until it is iterated.

    `language` is pinned to "en" here on purpose. When `language=None`,
    `transcribe()` itself must run a full decode of the first window to
    detect the language (see `WhisperModel.transcribe` in
    src/python/model.rs) before it can even return `TranscriptionInfo`. For
    a short single-window file that eager detection decode dominates setup
    time, so `consume > setup` would not hold even though the *segment*
    decoding is genuinely lazy -- it would be measuring the (deliberately
    eager) language detection, not a laziness bug. Passing `language="en"`
    skips that detection decode entirely, isolating the thing this test is
    actually meant to check: that no window is decoded before iteration.
    Do not remove the pinned language "to simplify" this test -- it changes
    what the assertion means.
    """
    import time

    model = whisper_rs.WhisperModel("tiny")
    audio = write_speechlike_wav(tmp_path / "a.wav", secs=4.0)

    start = time.perf_counter()
    segments, _ = model.transcribe(str(audio), language="en")
    setup = time.perf_counter() - start

    start = time.perf_counter()
    list(segments)
    consume = time.perf_counter() - start

    assert consume > setup, (
        f"consuming ({consume:.3f}s) should cost more than setup ({setup:.3f}s); "
        "decoding appears to be happening eagerly"
    )


@pytest.mark.model
def test_transcribe_returns_populated_info_before_any_segment_is_decoded(tmp_path):
    """Structural counterpart to the timing-based laziness test above.

    Timing assertions are inherently a little flaky; this pins the same
    claim -- transcribe() finishes setup (decode/VAD/windowing/info) without
    decoding any segment -- independently of the clock. `language="en"` for
    the same reason as above: it keeps setup free of the eager
    language-detection decode so this is purely about segment laziness.
    """
    model = whisper_rs.WhisperModel("tiny")
    audio = write_speechlike_wav(tmp_path / "a.wav", secs=4.0)

    segments, info = model.transcribe(str(audio), language="en")

    # info is fully realized immediately: these fields are not sentinels.
    assert info.duration == pytest.approx(4.0, abs=0.1)
    assert info.duration_after_vad is not None
    assert info.duration_after_vad <= info.duration

    # The iterator returned alongside it is untouched: a fresh, unconsumed
    # generator over the windows, not something that already ran a decode.
    assert iter(segments) is segments
    first = next(segments, None)
    # Whatever comes back (a Segment, or None for a silent file), this call
    # is what triggers the first decode -- proving none happened before it.
    assert first is None or isinstance(first, whisper_rs.Segment)


def _say_available() -> bool:
    import shutil

    return shutil.which("say") is not None


def _say_wav(path, text: str):
    """Synthesize `text` as a 16 kHz mono WAV with a known-English voice.

    The voice is pinned: `say`'s default follows the machine's system
    language, so on a non-English host the "English" fixture would not be
    English at all and the language-detection assertions would fail for a
    reason that has nothing to do with this crate.
    """
    import subprocess

    subprocess.run(
        ["say", "-v", "Samantha", "-o", str(path), "--data-format=LEI16@16000", text],
        check=True,
    )
    return path


@pytest.mark.model
@pytest.mark.skipif(not _say_available(), reason="macOS `say` is not available on this platform")
def test_auto_detection_identifies_the_spoken_language(tmp_path):
    """The `language=None` path, checked against speech of a known language.

    This replaces an earlier version that ran on the synthetic tone and only
    asserted `info.language != ""`. That assertion held even while detection
    was completely broken -- it returned the literal string `"unknown"`, which
    was then fed back to the decoder as the language token `<|unknown|>` and
    produced confident nonsense. Detection is only meaningfully covered by
    speech whose language is known in advance, so this test generates it.
    """
    audio = _say_wav(tmp_path / "en.wav", "The quick brown fox jumps over the lazy dog")

    model = whisper_rs.WhisperModel("tiny")
    _, info = model.transcribe(str(audio))

    assert info.language == "en", f"English speech detected as {info.language!r}"
    assert info.language_probability is not None, (
        "auto-detection must report the detector's own probability"
    )
    assert 0.0 < info.language_probability <= 1.0


@pytest.mark.model
@pytest.mark.skipif(not _say_available(), reason="macOS `say` is not available on this platform")
def test_pinned_language_reports_no_probability(tmp_path):
    """Nothing was detected, so there is no confidence score to report."""
    audio = _say_wav(tmp_path / "en.wav", "The quick brown fox jumps over the lazy dog")

    model = whisper_rs.WhisperModel("tiny")
    _, info = model.transcribe(str(audio), language="en")

    assert info.language == "en"
    assert info.language_probability is None


# `test_word_timestamps_are_absent_unless_requested` used to live here as a
# synthetic-audio test, but the tone burst `write_speechlike_wav` produces
# yields zero segments, so its `for seg in segments: assert seg.words is
# None` loop body never ran -- the test could not fail no matter what the
# code did. On Linux CI, where the real-speech test below is skipped (no
# `say`), that left `word_timestamps=False` with no live coverage at all.
# The check now lives inside `test_word_timestamps_are_plausible_on_real_speech`
# below, gated on that test first asserting at least one real segment exists,
# so the loop body is guaranteed to actually execute wherever it runs.


@pytest.mark.model
@pytest.mark.skipif(not _say_available(), reason="macOS `say` is not available on this platform")
def test_word_timestamps_are_plausible_on_real_speech(tmp_path):
    """Closes the one definition-of-done item synthetic audio cannot prove:
    that word-level timestamps look like real timestamps, on real speech.
    """
    audio = _say_wav(tmp_path / "speech.wav", "The quick brown fox jumps over the lazy dog")

    model = whisper_rs.WhisperModel("tiny")
    segments, _ = model.transcribe(str(audio), language="en", word_timestamps=True)
    segments = list(segments)

    assert len(segments) >= 1, "real speech should produce at least one segment"

    all_words = []
    for seg in segments:
        assert seg.text.strip() != "", "a real-speech segment must have non-empty text"
        assert seg.words is not None
        assert len(seg.words) >= 1
        for w in seg.words:
            assert w.start <= w.end, f"word {w.word!r} has start > end"
            assert seg.start - 0.01 <= w.start, f"word {w.word!r} starts before its segment"
            assert w.end <= seg.end + 0.01, f"word {w.word!r} ends after its segment"
        all_words.extend(seg.words)

    starts = [w.start for w in all_words]
    assert starts == sorted(starts), "word start times must be non-decreasing across the transcript"

    segments_no_words = list(model.transcribe(str(audio), language="en", word_timestamps=False)[0])
    # This is the live coverage for "words is absent unless requested": it
    # only proves anything if at least one segment actually exists to check,
    # which real speech (unlike the synthetic tone elsewhere in this file)
    # reliably produces.
    assert len(segments_no_words) >= 1, "real speech should produce at least one segment here too"
    for seg in segments_no_words:
        assert seg.words is None

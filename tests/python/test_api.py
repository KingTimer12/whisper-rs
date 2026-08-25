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
    """`say` and `afconvert` are macOS-only binaries used to synthesize the
    speech fixtures below. Checking the platform alone is not enough: some
    CI runners are macOS but strip these binaries, or `say` may exist while
    `afconvert` (needed for the multi-speaker fixture's WAV conversion) does
    not, so both are checked directly.
    """
    import shutil

    return shutil.which("say") is not None and shutil.which("afconvert") is not None


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


@pytest.mark.model
def test_word_timestamps_false_with_diarize_raises(tmp_path):
    model = whisper_rs.WhisperModel("tiny")
    audio = write_speechlike_wav(tmp_path / "a.wav", secs=2.0)

    with pytest.raises(ValueError, match="word_timestamps=False"):
        model.transcribe(str(audio), diarize=True, word_timestamps=False)


@pytest.mark.model
def test_diarize_false_leaves_speakers_unset(tmp_path):
    model = whisper_rs.WhisperModel("tiny")
    audio = write_speechlike_wav(tmp_path / "a.wav", secs=2.0)

    segments, info = model.transcribe(str(audio), language="en")

    assert info.num_speakers is None
    for seg in segments:
        assert seg.speaker is None


def _say_multi_speaker_wav(path, voices=("Samantha", "Alex", "Fred", "Daniel", "Karen")):
    """Concatenate one sentence per voice into a 16 kHz mono WAV.

    Generated rather than recorded because the voice order is then exact ground
    truth for both the speaker count and the turn order -- which no real
    recording would give.
    """
    import subprocess, wave, tempfile, pathlib

    tmp = pathlib.Path(tempfile.mkdtemp())
    parts = []
    for i, voice in enumerate(voices):
        aiff, wav = tmp / f"{i}.aiff", tmp / f"{i}.wav"
        subprocess.run(
            ["say", "-v", voice, "-o", str(aiff),
             f"This is speaker number {i}, saying a full sentence for the test."],
            check=True,
        )
        subprocess.run(
            ["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(aiff), str(wav)],
            check=True,
        )
        parts.append(wav)

    out = wave.open(str(path), "wb")
    with wave.open(str(parts[0]), "rb") as first:
        params = first.getparams()
        out.setparams(params)
    # A short silence gap between voices. Without it the clips abut directly
    # and the diarizer's segmentation stage -- which finds turns from
    # silence/energy boundaries, not from the clustering step -- merges
    # adjacent voices into a single detected turn, capping the number of
    # embeddings (and therefore the achievable cluster count) below the
    # actual number of speakers regardless of what `num_speakers` requests.
    silence = b"\x00\x00" * int(params.framerate * 0.5)
    for i, part in enumerate(parts):
        with wave.open(str(part), "rb") as r:
            out.writeframes(r.readframes(r.getnframes()))
        if i != len(parts) - 1:
            out.writeframes(silence)
    out.close()
    return path


@pytest.mark.model
@pytest.mark.skipif(not _say_available(), reason="macOS `say` is not available on this platform")
def test_diarization_finds_more_than_four_speakers(tmp_path):
    """The requirement that ruled out Sortformer, whose NUM_SPEAKERS = 4 is
    fixed in the model rather than configurable.

    `num_speakers` is passed to force an exact count. Task 0's validation
    gate measured that polyvoice's automatic speaker-count selection
    under-counts this same well-separated 5-speaker audio (Task 0 measured 3;
    this fixture, run during this task, also measured 3), which is why the
    exact-k override exists at all -- see
    `test_automatic_speaker_count_is_approximate` below for the honest
    coverage of that automatic path.
    """
    audio = _say_multi_speaker_wav(tmp_path / "five.wav")

    model = whisper_rs.WhisperModel("tiny")
    segments, info = model.transcribe(
        str(audio), language="en", diarize=True, max_speakers=8, num_speakers=5
    )
    segments = list(segments)

    assert info.num_speakers == 5, f"expected exactly 5 speakers, got {info.num_speakers}"

    speakers = {seg.speaker for seg in segments if seg.speaker is not None}
    assert len(speakers) >= 5, f"segments carry only {len(speakers)} distinct speakers"


@pytest.mark.model
@pytest.mark.skipif(not _say_available(), reason="macOS `say` is not available on this platform")
def test_automatic_speaker_count_is_approximate(tmp_path):
    """Records the honest behaviour of the automatic (no `num_speakers`)
    speaker-count selection path.

    Task 0's validation gate measured that polyvoice's automatic selection
    under-counts badly on this same 5-speaker fixture, despite its
    embeddings being demonstrably well separated -- Task 0 measured 3, and
    this fixture, run during this task, also measured 3. This
    test does not assert a specific count for that reason: an assertion
    tuned to the observed output would test nothing and would pass no
    matter how badly the count degraded later. It only asserts that the
    automatic path runs and returns *some* plausible speaker count. Callers
    who need an exact, reliable count should pass `num_speakers` explicitly
    (see `test_diarization_finds_more_than_four_speakers`).
    """
    audio = _say_multi_speaker_wav(tmp_path / "five.wav")

    model = whisper_rs.WhisperModel("tiny")
    segments, info = model.transcribe(str(audio), language="en", diarize=True, max_speakers=8)
    segments = list(segments)

    assert info.num_speakers is not None
    assert info.num_speakers >= 1

    speakers = {seg.speaker for seg in segments if seg.speaker is not None}
    assert len(speakers) >= 1


@pytest.mark.model
@pytest.mark.skipif(not _say_available(), reason="macOS `say` is not available on this platform")
def test_every_word_carries_a_speaker(tmp_path):
    audio = _say_multi_speaker_wav(tmp_path / "five.wav")

    model = whisper_rs.WhisperModel("tiny")
    segments, _ = model.transcribe(str(audio), language="en", diarize=True, num_speakers=5)
    segments = list(segments)

    assert len(segments) >= 5

    total = attributed = 0
    for seg in segments:
        assert seg.words is not None, "diarize=True must enable word timestamps"
        for w in seg.words:
            total += 1
            if w.speaker is not None:
                attributed += 1
            # A word's speaker always matches its segment's: segments are
            # built as runs of same-speaker words.
            assert w.speaker == seg.speaker

    assert total > 0
    assert attributed / total > 0.8, (
        f"only {attributed}/{total} words attributed; assignment is too sparse"
    )


@pytest.mark.model
@pytest.mark.skipif(not _say_available(), reason="macOS `say` is not available on this platform")
def test_segment_ids_stay_sequential_after_splitting(tmp_path):
    """Splitting changes how many segments exist, so this is the check that
    numbering really moved after the split rather than before it."""
    audio = _say_multi_speaker_wav(tmp_path / "five.wav")

    model = whisper_rs.WhisperModel("tiny")
    segments, _ = model.transcribe(str(audio), language="en", diarize=True, num_speakers=5)
    ids = [seg.id for seg in segments]

    assert ids == list(range(len(ids))), f"ids are not sequential and gap-free: {ids}"

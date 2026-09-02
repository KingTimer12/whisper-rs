"""Tests over the `nemotron`-feature build of the wheel. Run with: pytest tests/python -v

`whisper_rs.NemotronModel` only exists when the extension was built with
`--features nemotron`; every test in this file is skipped (not errored) on a
build without it. Tests marked `model` download the ~0.6B Nemotron model on
first run (`nvidia/nemotron-3.5-asr-streaming-0.6b`, via the `"nemotron"`
alias) -- see `tests/python/test_api.py` for the audio-fixture helper this
file reuses.
"""

import pytest

import whisper_rs
from test_api import write_speechlike_wav

pytestmark = pytest.mark.skipif(
    not hasattr(whisper_rs, "NemotronModel"),
    reason="whisper_rs was built without the 'nemotron' Cargo feature",
)


def test_nemotron_model_is_exported():
    assert hasattr(whisper_rs, "NemotronModel")


@pytest.mark.model
def test_nemotron_transcribes_real_speech(tmp_path):
    model = whisper_rs.NemotronModel("nemotron")
    audio = write_speechlike_wav(tmp_path / "a.wav", secs=4.0)

    segments, info = model.transcribe(str(audio), word_timestamps=True)
    segments = list(segments)

    assert len(segments) >= 1
    assert info.language is not None


@pytest.mark.model
def test_max_speakers_without_diarize_is_rejected():
    # No audio work needed: this is checked before any decoding begins.
    # A model instance is still required because construction is what loads
    # the weights (see the equivalent WhisperModel test in test_api.py).
    model = whisper_rs.NemotronModel("nemotron")

    with pytest.raises(ValueError, match="max_speakers"):
        model.transcribe("unused.wav", max_speakers=999)


@pytest.mark.model
def test_num_speakers_without_diarize_is_rejected():
    model = whisper_rs.NemotronModel("nemotron")

    with pytest.raises(ValueError, match="num_speakers"):
        model.transcribe("unused.wav", num_speakers=3)


@pytest.mark.model
def test_word_timestamps_false_with_diarize_raises(tmp_path):
    model = whisper_rs.NemotronModel("nemotron")
    audio = write_speechlike_wav(tmp_path / "a.wav", secs=2.0)

    with pytest.raises(ValueError, match="word_timestamps=False"):
        model.transcribe(str(audio), diarize=True, word_timestamps=False)


@pytest.mark.model
def test_diarize_composes_with_nemotron(tmp_path):
    """`diarize=True` reuses `diarize::polyvoice::PolyvoiceDiarizer` unchanged
    -- there is no Nemotron-specific diarization code path (Task 8). This
    only proves the composition doesn't error; it does not assert on speaker
    identity, since `write_speechlike_wav` is a single tone burst, not real
    multi-speaker speech.
    """
    model = whisper_rs.NemotronModel("nemotron")
    audio = write_speechlike_wav(tmp_path / "a.wav", secs=2.0)

    segments, info = model.transcribe(str(audio), diarize=True)
    segments = list(segments)

    assert info.num_speakers is not None

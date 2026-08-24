# whisper-rs

Whisper transcription with a Rust core. The heavy lifting -- audio decoding,
voice activity detection (VAD), windowing, and inference -- happens in Rust;
Python gets a small, `pyo3`-backed extension module.

## Usage

```python
import whisper_rs

model = whisper_rs.WhisperModel("tiny")
segments, info = model.transcribe("audio.wav")

print(info.duration, info.duration_after_vad)
for segment in segments:
    print(segment.start, segment.end, segment.text)
```

`model` may be a short alias (`"tiny"`, `"base"`, `"small"`, `"medium"`,
`"large-v3"`, ...), an explicit `org/repo` on the Hugging Face Hub, or a local
directory already in CTranslate2 format. `transcribe()` returns a tuple:
`segments` is a lazily-decoded iterator (nothing is decoded until you iterate
it) and `info` is a `TranscriptionInfo` that is fully populated by the time
`transcribe()` returns.

### Converting a model

`WhisperModel` needs a CTranslate2-format checkpoint. To convert an arbitrary
Hugging Face Whisper checkpoint yourself (rather than relying on one of the
pre-converted `Systran/faster-whisper-*` repos the built-in aliases point at):

```bash
pip install "whisper-rs[convert]"
python -m whisper_rs.convert openai/whisper-large-v3 ./my-model --quantization float16
```

or from Python:

```python
from whisper_rs.convert import convert_model

path = convert_model("openai/whisper-large-v3", "./my-model", quantization="float16")
model = whisper_rs.WhisperModel(path)
```

This wraps the official `ct2-transformers-converter` (from the `ctranslate2`/
`transformers` packages) rather than reimplementing model conversion in Rust:
conversion is one-shot weight I/O and tensor renaming, and the official
converter tracks CTranslate2 format changes.

## Known behaviors and limitations (v1)

These are deliberate properties of the current implementation, not bugs --
documented here so they aren't rediscovered the hard way.

**Voice activity detection defaults to WebRTC, not Silero.** `wavekat-vad`
supports both backends, but Silero is compiled in only behind the
`silero-vad` Cargo feature, off by default. The reason is a hard runtime
conflict, not a preference: Silero's ONNX Runtime backend and CTranslate2
both statically link `protobuf`, and having both in the same process causes a
SIGBUS at runtime on at least the platforms this project has tested. WebRTC
has no such collision (no ONNX Runtime, no model file), so it is the safe
default. One consequence: WebRTC emits only binary 0.0/1.0 speech
probabilities, so `vad_parameters["threshold"]`/`["neg_threshold"]` --
which select a hysteresis band between the two -- degenerate to a no-op on
the default build (a `tracing::warn!` fires if either is set while
`silero-vad` is off). Build with `--features silero-vad` for a graded VAD
backend where those two parameters have an effect.

**`info.language_probability` is always `None`.** `ct2rs` (the CTranslate2
Rust binding this crate uses) exposes no language-detection API returning a
confidence score. The language code itself is recovered from the `<|xx|>`
token Whisper's own output starts with, but there is no probability to go
with it, and this API does not fabricate one.

**The whole audio file is decoded into memory before anything else
happens.** `transcribe()` decodes, resamples, VADs, and windows the entire
file synchronously and holds it as `f32` samples for the duration of the
call -- there is no streaming/chunked-from-disk decoding path in v1. For very
long files this is a real, current memory cost proportional to file length
(16 kHz mono `f32` is 64 KB/s of decoded audio, so roughly 230 MB for a 1
hour file), independent of the model itself.

**Auto-detecting the language costs an extra full decode.** If `language` is
left as `None` (the default), `transcribe()` runs a full 30 s decode of the
first window up front, purely to read the language token off its output,
before it can return `TranscriptionInfo` (whose `language` field must
already be populated). That decode is in addition to the ordinary
transcription decode the same window gets later, once you start iterating
`segments` -- so on an auto-detected file, the first window is decoded
twice. Passing `language=` explicitly (e.g. `model.transcribe(path,
language="en")`) skips the detection decode entirely and is the cheaper
path whenever you already know the language.

## Development

```bash
cargo test --all-targets
cargo clippy --all-targets -- -D warnings

python3 -m venv .venv && source .venv/bin/activate
pip install maturin pytest
maturin develop
pytest tests/python -v -m "not model"   # no network, no model download
pytest tests/python -v -m model         # downloads the tiny model on first run
```

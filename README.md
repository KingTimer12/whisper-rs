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

### Speaker diarization

```bash
pip install "whisper-rs[diarization]"
```

Pass `diarize=True` to assign a speaker to each segment and each word:

```python
import whisper_rs

model = whisper_rs.WhisperModel("tiny")
segments, info = model.transcribe("meeting.wav", diarize=True, num_speakers=3)

print(info.num_speakers)
for segment in segments:
    print(segment.speaker, segment.start, segment.end, segment.text)
    for word in segment.words:
        print(" ", word.speaker, word.start, word.end, word.word)
```

`num_speakers` (default `None`) forces an exact speaker count when you know
it in advance. `max_speakers` (default `8`) only bounds the automatic search
that runs when `num_speakers` is left unset -- see the limitation below on
why that automatic search is not reliable. `info.num_speakers` reports the
distinct speaker count actually found in the returned turns (bounded by
`max_speakers`, and at most `num_speakers` when it was passed -- it can come
back lower, because the segmentation stage may find fewer distinct turns than
there are speakers, and the count reports what was actually found rather than
what was asked for), and is
`None` when `diarize=False`. `Segment.speaker` and `Word.speaker` are the
assigned speaker index, or `None` when no diarization ran or no turn covered
that span.

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

**Auto-detecting the language loads a second, transient copy of the
weights.** `ct2rs`'s high-level `Whisper` detects the language internally
when `language` is `None`, but never reports what it detected, and it keeps
the low-level handle that *can* report it (`ct2rs::sys::Whisper`, whose
`detect_language` returns every language with its probability) private. So
when `language` is left as `None`, `transcribe()` loads its own detector from
the same model directory, runs one encoder pass over the first window, reads
off the code and its probability, and drops the detector before returning.
The extra copy is transient rather than resident, but the load is real: for a
large model it is the dominant cost of the call. Passing `language=`
explicitly (e.g. `model.transcribe(path, language="en")`) skips detection and
that load entirely, and is the cheaper path whenever you already know the
language.

`info.language_probability` carries the detector's probability for the
detected language, and is `None` when `language=` was pinned (nothing was
detected, so there is no score to report) or when the file contained no
speech at all.

**The whole audio file is decoded into memory before anything else
happens.** `transcribe()` decodes, resamples, VADs, and windows the entire
file synchronously and holds it as `f32` samples for the duration of the
call -- there is no streaming/chunked-from-disk decoding path in v1. For very
long files this is a real, current memory cost proportional to file length
(16 kHz mono `f32` is 64 KB/s of decoded audio, so roughly 230 MB for a 1
hour file), independent of the model itself.

**`diarize=True` breaks the iterator's laziness.** Diarization needs the
whole file before the first speaker can be assigned, so it runs eagerly
inside `transcribe()`, alongside VAD, windowing, and language detection,
before `transcribe()` returns. ASR decoding itself stays lazy: segment text
is still produced only as you iterate `segments`.

**Diarization needs onnxruntime, and it must be dynamically loaded.** The
`diarization` feature links both CTranslate2 and `ort` (onnxruntime) into
the same process. Both statically link `protobuf`, and having both static
copies in one process is exactly the collision described above for Silero
VAD -- it aborts the process with `signal: 10, SIGBUS: access to undefined
memory`. Loading `ort` in `load-dynamic` mode instead of linking it
statically is therefore not a preference, it is the only way the feature can
coexist with CTranslate2 at all. `pip install whisper-rs[diarization]` pulls
in `onnxruntime>=1.28`; that floor is named in the "could not initialise"
error message as a hypothesis about what an initialization failure means,
not something checked at runtime -- an older or incompatible onnxruntime
is not rejected up front, it is left to fail however it fails.

**The automatic speaker count under-counts.** With `num_speakers` left
unset, the automatic clustering search (bounded above by `max_speakers`) was
measured on a five-voice fixture with well-separated embeddings (pairwise
cosine off-diagonal 0.0176-0.6070) to find only 3 of the 5 speakers. Passing
`num_speakers=5` on the same audio yields exactly 5. If you know the speaker
count, pass `num_speakers` -- relying on the automatic search to report the
right number is not currently safe.

**There is no `min_speakers`.** `polyvoice` has no such knob; the clustering
decides the count on its own (see above), bounded only above by
`max_speakers`, or forced exactly by `num_speakers`.

**Segments split where the speaker changes**, and a split segment's text is
rebuilt from its words, so it can differ from the unsplit text in
whitespace. It has to be rebuilt: the underlying decoder trims every word,
so the original spacing is already gone by the time a split is possible.
The rebuild puts one space between words and none before punctuation that
attaches to what it follows, which recovers ordinary prose but will not
reproduce unusual spacing. Segments that are *not* split keep the decoder's
own text verbatim.

**Diarization's CTranslate2/onnxruntime coexistence is verified on macOS,
not yet confirmed on Linux.** The `diarization` feature links both
CTranslate2 and `ort` (onnxruntime) into the same process; both statically
link `protobuf`, which is exactly the collision described above for Silero
VAD and used to abort the process with `signal: 10, SIGBUS: access to
undefined memory`. The fix was moving `ort` to `load-dynamic`
(`tests/coexistence.rs`, `ctranslate2_and_onnxruntime_coexist`, run via
`cargo test --features diarization --test coexistence -- --ignored`), which
has been confirmed clean locally on macOS. macOS's two-level namespaces and
Linux's `RTLD_GLOBAL` symbol resolution are different enough that the fix
holding on one platform does not guarantee it holds on the other, so CI runs
this test on both `macos-latest` and `ubuntu-latest` (manual dispatch only,
in `.github/workflows/CI.yml`'s `coexistence` job) to close that gap. Until
that job has actually run and passed on Linux, treat Linux diarization
support as unconfirmed rather than assumed.

**Warnings go to stderr, and are off unless you ask for more.** The crate
emits `tracing` events for things that are worth knowing but not worth
failing on -- a synthesized `preprocessor_config.json`, a VAD parameter that
is inert on the current backend, a corrupt packet padded with silence. A
plain Python process installs no `tracing` subscriber, so importing
`whisper_rs` installs one on stderr showing warnings and errors. Raise or
lower it with `WHISPER_RS_LOG` (standard `EnvFilter` syntax, e.g.
`WHISPER_RS_LOG=whisper_rs=debug`). If the embedding application already set
a global subscriber, that one wins and this is a no-op.

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

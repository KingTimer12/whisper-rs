# whisper-rs — Design Spec

Date: 2026-08-24
Status: Approved for planning

## 1. Purpose and Scope

`whisper-rs` = Python package, Rust core, transcribe audio with Whisper models. Takes good ideas from faster-whisper (CTranslate2 inference, lazy segment generator), whisperx (VAD chunking), whisper-diarization (speaker attribution — post-v1).

Max work in Rust. Python only where better tool: thin wrapper around official `ct2-transformers-converter` for users bringing own checkpoint.

### v1 scope

- Whisper ASR via `ct2rs` (CTranslate2 bindings).
- VAD speech detection + windowing via `wavekat-vad`.
- Audio decode + resample in Rust (`symphonia` + `rubato`).
- Model resolution + download from Hugging Face in Rust (`hf-hub`).
- Lazy segment generator plus eager `TranscriptionInfo` on Python side.

### Explicitly out of v1

| Deferred | Reason |
|---|---|
| Diarization / speaker attribution | Needs speaker-embedding model plus clustering; no chosen crate provide it. Comes later as post-processor. |
| Parakeet ASR backend | `Asr` trait exists for it; no impl in v1. |
| Forced alignment (wav2vec2, whisperx-style) | `ct2rs` already return per-word timestamps. Only worth adding if precision prove insufficient. |
| Temperature fallback, `condition_on_previous_text` | Need per-token probabilities current `ct2rs` API not expose. See Approach B/C below. |
| Microphone streaming, multi-file `transcribe_batch`, incremental decoding of long audio | Not needed for v1. |

## 2. Upstream API Constraints (verified)

These facts shape whole design. Confirmed against docs.rs before writing.

`ct2rs`:

```rust
pub fn new<T: AsRef<Path>>(model_path: T, config: Config) -> Result<Self>
pub fn generate(&self, samples: &[f32], language: Option<&str>, timestamp: bool,
                options: &WhisperOptions) -> Result<Vec<String>>
pub fn generate_segments(&self, samples: &[f32], language: Option<&str>,
                         options: &WhisperOptions) -> Result<Vec<Segment>>
pub fn sampling_rate(&self) -> usize
pub fn n_samples(&self) -> usize
```

- Input = **raw f32 samples** normalised to `[-1, 1]` at `sampling_rate()`
  (16 kHz). Mel spectrogram, beam search, timestamp extraction all happen
  inside CTranslate2.
- `n_samples()` = hard 30 s window ceiling. One call decode one window.
- `Segment` carry segment timings and `Word { start, end, text, probability }`.
- **No language-detection API.** No `detect_language`, no
  `DetectionResult`. Drives decision in section 6.

`wavekat-vad`:

- Trait `VoiceActivityDetector` with `process(&samples, sample_rate) -> Result<f32, VadError>`,
  return per-frame speech probability (0.0–1.0 for neural backends).
- Backends: `SileroVad` (8/16 kHz, ONNX LSTM), `TenVad` (16 kHz),
  `FireRedVad` (16 kHz), `WebRtcVad` (8/16/32/48 kHz, binary output).
- `FrameAdapter` feed backend's required frame size from arbitrary chunks.
- `Preprocessor` optionally clean audio.

Grouping frame probabilities into speech regions = **our** logic, not crate's. That where transcription quality won or lost.

## 3. Chosen Approach

**VAD-first chunking (whisperx-style).** VAD run over whole signal, produce speech regions; regions merged into windows of max 30 s respecting silence boundaries; each window = one `generate_segments` call; window timestamps offset into global timeline.

Considered and rejected for v1:

- **B. Sequential sliding window (faster-whisper-style)** — window, decode,
  advance by last emitted timestamp, with `condition_on_previous_text`,
  temperature fallback, `compression_ratio` / `no_speech_prob` checks. Better
  continuous text, but strictly serial, and most fallback logic need
  probabilities `generate_segments` not expose — would require dropping
  to low-level `ct2rs` API.
- **C. Hybrid** — VAD to cut silence, sequential windows with context inside
  each speech region. Best quality, complexity of both.

Windows independent under A: trivially parallelisable, no cross-window decoder state. B and C stay reachable later as another impl behind `Asr` / chunker boundary, no change to Python API.

**Accepted cost:** each window decode in isolation, so `Segment` never cross window boundary. Continuous speech show segment break roughly every 30 s that faster-whisper would not produce. Approach C would fix this specific weakness.

## 4. Architecture

One cdylib crate (delivery target = Python package), split into modules with hard boundaries: no pyo3 outside `python/`. Every non-Python module testable with plain `cargo test`; core extractable into own crate later, no rewrite.

```
src/
  lib.rs            #[pymodule], registration only, no logic
  types.rs          shared plain data types
  python/
    mod.rs          Rust <-> Python conversion, allow_threads
    model.rs        WhisperModel class
    segment.rs      Segment / Word / TranscriptionInfo classes
    iter.rs         lazy generator (__next__)
  audio/
    decode.rs       symphonia: any container -> f32 mono
    resample.rs     rubato -> 16 kHz
  vad/
    mod.rs          Vad trait + config; wavekat-vad wrapper
    speech.rs       frame probabilities -> Vec<SpeechRegion> (hysteresis)
  chunk.rs          SpeechRegion[] -> Vec<Window> (<= 30 s)
  asr/
    mod.rs          Asr trait (lets Parakeet in later)
    ct2.rs          ct2rs::Whisper implementation
  stitch.rs         per-window Segment -> global timeline
  models/
    hub.rs          hf-hub: name -> cached directory
    registry.rs     aliases ("large-v3" -> Systran/...)
  error.rs          thiserror -> Python exceptions
python/whisper_rs/
  __init__.py       re-exports the native module
  convert.py        ct2-transformers-converter wrapper (optional)
```

Dependency rule: `audio`, `vad`, `chunk`, `asr`, `stitch` know nothing about `python` and nothing about each other beyond types in `types.rs`. `python/` = only module touching `Py*`.

### Data flow

```
path -> decode + resample -> Vec<f32> @ 16 kHz mono
     -> vad probabilities -> SpeechRegion[] -> Window[] (offset, samples)
     -> per window: Asr::transcribe(&samples) -> Segment[]
     -> stitch: add offset, renumber ids -> Python iterator
```

### Accepted limitation

Whole decoded signal held in memory as `Vec<f32>` (1 h ≈ 230 MB). Incremental decoding would complicate both global VAD pass and lazy generator. Very long audio = documented limitation, not solved problem.

## 5. Core Types and Algorithms

```rust
pub struct SpeechRegion { pub start: usize, pub end: usize }   // sample indices
pub struct Window {
    pub offset: usize,          // start sample in the original audio
    pub samples: Vec<f32>,      // <= 30 s, zero-padded
}
pub struct Word  { pub start: f32, pub end: f32, pub text: String, pub probability: f32 }
pub struct Seg   { pub id: u32, pub start: f32, pub end: f32, pub text: String,
                   pub words: Option<Vec<Word>> }
pub struct Info  { pub language: String, pub language_probability: Option<f32>,
                   pub duration: f32, pub duration_after_vad: f32 }
```

All `Send`, free of pyo3 types.

### VAD -> regions (`vad/speech.rs`)

Two-threshold hysteresis state machine, following reference silero-vad behaviour:

1. Frame above `threshold` (default 0.5) open region.
2. Region close only after probability stay below `threshold - 0.15`
   longer than `min_silence_duration_ms` (default 2000). Two thresholds stop
   speech being chopped at every breath.
3. Regions shorter than `min_speech_duration_ms` (default 250) discarded.
4. Each edge padded by `speech_pad_ms` (default 400), then now-overlapping regions merged.

Default backend = `SileroVad` at 16 kHz, fed through `FrameAdapter`. Crate's `Preprocessor` off by default: audio reaching ASR must not be altered.

### Regions -> windows (`chunk.rs`)

Two constraints in tension: `n_samples()` = hard 30 s ceiling, and cutting mid-word wreck transcription.

1. Accumulate regions in order while `region.end - window.offset <= 30 s`.
2. Single region longer than 30 s sliced into 30 s pieces, no overlap,
   cutting at **lowest speech probability** within last 2 s of window. Not
   perfect cut, but far better than cutting at exactly 30.000 s.
3. Silence between regions inside one window kept, not collapsed. Collapsing
   would shift timestamps and require remapping table — complexity Approach A
   need not pay.
4. Each window zero-padded to 30 s before `ct2rs` call; model expect full window.

### Stitching (`stitch.rs`)

`t_global = t_window + offset / 16000`. Segments whose `start` fall inside zero padding dropped; `end` clamped to real window end; `id` renumbered sequentially along global timeline. `duration_after_vad` = sum of speech region durations.

## 6. Python API

```python
from whisper_rs import WhisperModel

model = WhisperModel("large-v3", device="cuda", compute_type="float16")
segments, info = model.transcribe("audio.mp3", language=None, word_timestamps=True)
print(info.language, info.duration_after_vad)
for s in segments:                    # nothing has decoded yet
    print(s.start, s.end, s.text)
```

**Constructor:** `WhisperModel(model, *, device="cpu", device_index=0,
compute_type="default", cpu_threads=0, num_workers=1, download_root=None,
local_files_only=False)` — names match faster-whisper, so drop-in. `model` accept alias (`"large-v3"`), explicit HF repo, or local path.

**transcribe:** `(audio, *, language=None, task="transcribe", beam_size=5,
patience=1.0, length_penalty=1.0, temperature=0.0, repetition_penalty=1.0,
no_repeat_ngram_size=0, max_initial_timestamp=1.0, suppress_blank=True,
word_timestamps=False, vad_filter=True, vad_parameters=None) ->
(SegmentIterator, TranscriptionInfo)`.

Only params `WhisperOptions` actually support accepted. Anything faster-whisper offer that `ct2rs` cannot honour = **absent from signature** — no argument accepted then silently ignored.

### Eager / lazy split

`transcribe` eagerly do decode, resample, VAD, windowing, language detection — minimum needed to return populated `info`. ASR calls lazy, one per `__next__`.

### Language detection

No dedicated API exists, so one `generate(&first_window, None, true, opts)` call run before iteration and raw output scanned for `<|xx|>` language token Whisper emit at start. Detected language then passed **explicitly** to every later window. Cost = one extra 30 s decode; also kill classic bug where language flip between windows mid-file.

Token absent → `info.language` = `"unknown"`.
`info.language_probability` = `Optional[float]`, always `None` on this path — probability not recoverable through current API, and `None` better than invented number. Passing explicit `language="pt"` skip extra call entirely.

### Concurrency and the GIL

`WhisperModel` hold `Arc<Ct2Asr>`; iterator clone that `Arc` and own window queue plus queue of already-produced segments:

```rust
#[pyclass] struct SegmentIterator {
    asr: Arc<Ct2Asr>, windows: VecDeque<Window>,
    pending: VecDeque<Seg>, next_id: u32, language: String, opts: WhisperOptions,
}
```

`__next__` serve from `pending` when non-empty; else take next window and decode inside `py.allow_threads(|| ...)` so other Python threads run during compute. Intra-model parallelism left to CTranslate2's own `num_workers` / `cpu_threads` thread pool; stacking rayon pool on top would just contend for cores.

**Risk to validate in first implementation task:** if `ct2rs::Whisper` not `Send + Sync`, shared `Arc` invalid and model must sit behind `Mutex` (serialising decodes, still releasing GIL). First thing plan checks.

### Errors (`error.rs`, thiserror -> exception)

| Condition | Python exception |
|---|---|
| Unreadable audio, unsupported format | `ValueError` |
| Missing model, failed download | `OSError` |
| Internal CTranslate2 failure | `RuntimeError` |

Messages carry file path and model name.

## 7. Models, Cache and the Python Side

**Registry** (`models/registry.rs`): static alias table — `tiny`, `base`, `small`, `medium`, `large-v1`..`large-v3`, `distil-large-v3` — mapping to matching `Systran/faster-whisper-*` repos (`distil-*` to `distil-whisper/...`). Any string containing `/` treated as explicit HF repo; any string existing on disk treated as local path. Resolution order: local path, then alias, then HF repo.

**Hub** (`models/hub.rs`): `hf-hub` with blocking backend, fetching only files CTranslate2 need — `model.bin`, `config.json`, `tokenizer.json`, `vocabulary.*`, `preprocessor_config.json` — not whole repo. Cache lives in `$HF_HOME` / `~/.cache/huggingface`, or `download_root` when given; `local_files_only=True` fail rather than download. Reusing HF cache means anyone who already have faster-whisper installed re-download nothing.

**Validation before instantiation:** if `model.bin` absent from resolved directory, fail immediately naming what missing and where it looked, rather than letting CTranslate2 abort with C++ message.

**Python side** (`python/whisper_rs/convert.py`) — only logic staying in Python: thin wrapper over `ct2-transformers-converter` for users with own checkpoint.

```python
from whisper_rs.convert import convert_model
convert_model("openai/whisper-large-v3", "./my-ct2", quantization="float16")
```

Shell out to converter via `subprocess`, check `ctranslate2` installed, print `pip install ctranslate2 transformers` if not. `ctranslate2` and `transformers` live in `[project.optional-dependencies]
convert`, never in required dependencies — users who only transcribe install nothing beyond wheel. Also runnable as `python -m whisper_rs.convert ...`.

Deliberately not done: reimplementing conversion in Rust. It weight I/O and tensor renaming, run once per model, and official converter track CTranslate2 format changes. No gain in porting.

**Crate dependencies:** `pyo3`, `ct2rs` (feature `whisper`), `wavekat-vad` (silero backend), `symphonia` (features `mp3`, `isomp4`, `aac`, `flac`, `vorbis`, `wav`), `rubato`, `hf-hub`, `thiserror`, `tracing`. No tokio — whole path blocking, laziness come from iterator, not async.

## 8. Testing

No logic of ours need model to test. Parts that actually break — VAD hysteresis, windowing, stitching — pure functions over `Vec<f32>` / `Vec<SpeechRegion>`, covered by fast `cargo test`, no download, no GPU. TDD applies to those.

| Layer | How it is tested | Needs a model? |
|---|---|---|
| `vad/speech.rs` | synthetic probabilities (ramps, spikes, gaps) -> expected regions; edge cases: all silence, all speech, speech at 0 s, speech to last sample | no |
| `chunk.rs` | synthetic regions -> windows; invariants: never > 30 s, monotonic offsets, 90 s region become 3 windows, no speech sample lost | no |
| `stitch.rs` | per-window segments -> timeline; sequential ids, no overlap, nothing past end of audio | no |
| `audio/` | in-memory generated WAV (sine) round-trip; sample rate and channel conversion | no |
| `models/registry.rs` | alias / repo / local path resolution | no |
| End to end | `tiny` plus short committed clip, asserting expected words | yes, `#[ignore]` |
| Python API | pytest over wheel: generator laziness (nothing run before first `next`), types, mapped exceptions | `tiny` |

Two invariants get explicit tests because easiest thing to break in refactor: sum of window speech durations equals `duration_after_vad`, and no speech sample appear in two windows.

## 9. Build and CI

`maturin` already configured. Real cost = `ct2rs`, which build CTranslate2 through CMake — slow, need C++ toolchain.

CI run `cargo test` (no model features) on Linux and macOS, `cargo clippy -D warnings`, and `maturin build` plus wheel smoke test. `#[ignore]` tests run only in manual or nightly job with `tiny` model cached. Release wheels come from `maturin-action`. CUDA = documented opt-in build, not published wheel — packaging CTranslate2 with CUDA is project of its own.

## 10. Definition of Done (v1)

`pip install`, `WhisperModel("tiny")`, transcribe multi-minute mp3 and wav with plausible word timestamps, VAD cutting silence, model-free test suite green.
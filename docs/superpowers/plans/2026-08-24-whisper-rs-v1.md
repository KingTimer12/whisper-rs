# whisper-rs v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a Python package with a Rust core that transcribes audio with Whisper via CTranslate2, using VAD-driven 30 s windowing and a lazy segment generator.

**Architecture:** VAD-first chunking. Audio is decoded and resampled to 16 kHz mono in Rust, a VAD produces per-frame speech probabilities, a hysteresis pass turns those into speech regions, regions are packed into windows of at most 30 s, each window is one `ct2rs` Whisper call, and per-window segments are stitched into a global timeline exposed to Python as a lazy iterator. Windows are independent, so no cross-window decoder state exists.

**Tech Stack:** Rust 2024, pyo3 0.28 + maturin, `ct2rs` (CTranslate2), `wavekat-vad` (Silero), `symphonia`, `rubato`, `hf-hub`, `thiserror`. Python only for the optional `ct2-transformers-converter` wrapper.

**Spec:** `docs/superpowers/specs/2026-08-24-whisper-rs-design.md`

## Global Constraints

- Sample rate is fixed at 16000 Hz, mono, `f32` normalised to `[-1, 1]`. Constant `SAMPLE_RATE: usize = 16_000`.
- Window ceiling is a hard 30 s: `WINDOW_SAMPLES: usize = 480_000`. No window may exceed it.
- No pyo3 types outside `src/python/`. `audio`, `vad`, `chunk`, `asr`, `stitch`, `models` must compile and test without Python.
- Modules other than `python/` do not depend on each other beyond `types.rs` and `error.rs`.
- No tokio. The whole path is blocking; laziness comes from the iterator.
- Public Python constructor and `transcribe` keyword names match faster-whisper exactly, and no accepted argument is ever silently ignored.
- Every test in tasks 1-9 runs with `cargo test` and requires no model download, no network, and no GPU. Tests that need a model are `#[ignore]`.
- `info.language_probability` is always `None` in v1 — the value is not recoverable through the `ct2rs` API. Never fabricate it.
- Commit after every task. Conventional commit prefixes (`feat:`, `test:`, `chore:`, `docs:`).

### Deviation from the spec, recorded here

The spec's `Window` has `offset` and `samples`. Implementation adds a third field, `real_len: usize` — the sample count before zero padding. Stitching needs it to drop segments that start inside the padding, and deriving it later from the region list would mean passing that list around. Same behaviour, one honest field.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/types.rs` | Plain data types and the two global constants. No logic. |
| `src/error.rs` | `Error` enum and `Result` alias. Maps to Python exceptions in `python/mod.rs`. |
| `src/audio/decode.rs` | symphonia: container/codec to interleaved `f32`, downmix to mono. |
| `src/audio/resample.rs` | rubato: any input rate to 16 kHz. |
| `src/audio/mod.rs` | `load_16k_mono(path) -> Result<Vec<f32>>`, composing the two above. |
| `src/vad/speech.rs` | Pure: frame probabilities to `Vec<SpeechRegion>` (hysteresis). |
| `src/vad/mod.rs` | `Vad` trait, `VadParams`, the `wavekat-vad` Silero wrapper producing probabilities. |
| `src/chunk.rs` | Pure: regions to sample ranges (`plan_windows`), plus `build_window` copying and padding. |
| `src/stitch.rs` | Pure: per-window segments to global timeline. |
| `src/asr/mod.rs` | `Asr` trait, `parse_language_token`. |
| `src/asr/ct2.rs` | `Ct2Asr`: `ct2rs::Whisper` implementation of `Asr`. |
| `src/models/registry.rs` | Alias table and `resolve(name) -> ModelRef`. |
| `src/models/hub.rs` | `hf-hub` download of the needed files, cache handling, validation. |
| `src/pipeline.rs` | `prepare(path, params) -> Prepared` — the eager half of `transcribe`. |
| `src/python/mod.rs` | Error to exception mapping, module registration helper. |
| `src/python/segment.rs` | `Segment`, `Word`, `TranscriptionInfo` pyclasses. |
| `src/python/model.rs` | `WhisperModel` pyclass. |
| `src/python/iter.rs` | `SegmentIterator` pyclass (lazy `__next__`). |
| `src/lib.rs` | `#[pymodule]` registration only. |
| `python/whisper_rs/__init__.py` | Re-exports the native module. |
| `python/whisper_rs/convert.py` | `ct2-transformers-converter` wrapper. |
| `tests/e2e.rs` | `#[ignore]` end-to-end test with the `tiny` model. |
| `tests/python/test_api.py` | pytest over the built wheel. |
| `.github/workflows/CI.yml` | cargo test, clippy, maturin build, wheel smoke test. |

---

## Task 1: Foundations — dependencies, types, errors, and the Send+Sync check

The single most important thing this task establishes is whether `ct2rs::Whisper` is `Send + Sync`. The whole `Arc<Ct2Asr>` design in Task 10 depends on it, so it is settled by a compile-time assertion before any other code is written.

**Files:**
- Modify: `Cargo.toml`
- Create: `src/types.rs`
- Create: `src/error.rs`
- Modify: `src/lib.rs`

**Interfaces:**
- Consumes: nothing.
- Produces: `SAMPLE_RATE: usize`, `WINDOW_SAMPLES: usize`, `SpeechRegion { start: usize, end: usize }`, `Window { offset: usize, samples: Vec<f32>, real_len: usize }`, `Word { start: f32, end: f32, text: String, probability: f32 }`, `Seg { id: u32, start: f32, end: f32, text: String, words: Option<Vec<Word>> }`, `Info { language: String, language_probability: Option<f32>, duration: f32, duration_after_vad: f32 }`, `enum Error`, `type Result<T> = std::result::Result<T, Error>`.

- [ ] **Step 1: Add dependencies**

Replace the `[dependencies]` section of `Cargo.toml` with:

```toml
[dependencies]
pyo3 = { version = "0.28.3", features = ["extension-module", "abi3-py38"] }
ct2rs = { version = "0.9", features = ["whisper"] }
wavekat-vad = { version = "0.2", features = ["silero"] }
symphonia = { version = "0.5", features = ["mp3", "isomp4", "aac", "flac", "vorbis", "wav", "pcm"] }
rubato = "0.16"
hf-hub = { version = "0.4", features = ["ureq"], default-features = false }
thiserror = "2"
tracing = "0.1"

[dev-dependencies]
hound = "3"
```

Then run `cargo add --dry-run ct2rs wavekat-vad` style verification by simply running the build in Step 2. If a version above does not exist, run `cargo search <crate>` and use the latest published version — record the version you used in the commit message.

- [ ] **Step 2: Verify the dependency tree builds**

Run: `cargo build`
Expected: success. This compiles CTranslate2 through CMake and is slow (10-30 min on a cold cache). A CMake or C++ compiler error here means the toolchain is missing, not that the plan is wrong — install a C++ toolchain and CMake and retry.

- [ ] **Step 3: Write the Send+Sync assertion test**

Create `src/asr/mod.rs` with only this content for now:

```rust
#[cfg(test)]
mod tests {
    /// The Arc<Ct2Asr> design in the Python layer requires this.
    /// If this fails to compile, Ct2Asr must wrap Whisper in a Mutex.
    #[test]
    fn whisper_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ct2rs::Whisper>();
    }
}
```

And add `mod asr;` to `src/lib.rs`.

- [ ] **Step 4: Run the assertion**

Run: `cargo test whisper_is_send_and_sync`
Expected: PASS.

**If it FAILS to compile:** stop and record the outcome. `Ct2Asr` in Task 9 must then hold `Mutex<ct2rs::Whisper>` and `transcribe` must lock it. Everything else in the plan is unchanged — the `Arc<Ct2Asr>` sharing still works, decodes just serialise. Note the decision in the Task 9 commit message.

- [ ] **Step 5: Write types.rs**

```rust
//! Plain data types shared across modules. No logic, no pyo3.

/// Sample rate every module assumes, in Hz. Matches ct2rs::Whisper::sampling_rate().
pub const SAMPLE_RATE: usize = 16_000;

/// Hard window ceiling in samples (30 s). Matches ct2rs::Whisper::n_samples().
pub const WINDOW_SAMPLES: usize = 480_000;

/// A contiguous run of speech, in sample indices into the decoded audio.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpeechRegion {
    pub start: usize,
    pub end: usize,
}

impl SpeechRegion {
    pub fn len(&self) -> usize {
        self.end.saturating_sub(self.start)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// One decoder input: at most 30 s of audio, zero-padded to exactly 30 s.
#[derive(Debug, Clone)]
pub struct Window {
    /// Start sample of this window in the original audio.
    pub offset: usize,
    /// Exactly WINDOW_SAMPLES long, zero-padded at the end.
    pub samples: Vec<f32>,
    /// Real sample count before padding. Stitching uses it to drop
    /// segments that start inside the padding.
    pub real_len: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Word {
    pub start: f32,
    pub end: f32,
    pub text: String,
    pub probability: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Seg {
    pub id: u32,
    pub start: f32,
    pub end: f32,
    pub text: String,
    pub words: Option<Vec<Word>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Info {
    pub language: String,
    /// Always None in v1: not recoverable through the ct2rs API.
    pub language_probability: Option<f32>,
    pub duration: f32,
    pub duration_after_vad: f32,
}
```

- [ ] **Step 6: Write error.rs**

```rust
//! Error type for the whole crate. Mapped to Python exceptions in python/mod.rs.

use std::path::PathBuf;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("cannot read audio file {path}: {message}")]
    AudioRead { path: PathBuf, message: String },

    #[error("unsupported audio format in {path}: {message}")]
    AudioFormat { path: PathBuf, message: String },

    #[error("audio file {path} contains no audio track")]
    AudioEmpty { path: PathBuf },

    #[error("resampling failed: {0}")]
    Resample(String),

    #[error("model {name} not found: {message} (looked in {path})")]
    ModelNotFound {
        name: String,
        path: PathBuf,
        message: String,
    },

    #[error("failed to download model {name}: {message}")]
    Download { name: String, message: String },

    #[error("voice activity detection failed: {0}")]
    Vad(String),

    #[error("CTranslate2 failed while transcribing: {0}")]
    Ct2(String),
}
```

- [ ] **Step 7: Wire up lib.rs**

```rust
mod asr;
mod error;
mod types;

use pyo3::prelude::*;

/// Placeholder module. Task 10 replaces this with the real classes.
#[pymodule]
mod whisper_rs {}
```

- [ ] **Step 8: Run the full suite**

Run: `cargo test`
Expected: PASS (the Send+Sync test plus zero others).

Run: `cargo clippy --all-targets -- -D warnings`
Expected: no warnings.

- [ ] **Step 9: Commit**

```bash
git add Cargo.toml Cargo.lock src/types.rs src/error.rs src/asr/mod.rs src/lib.rs
git commit -m "feat: add core types, error enum, and ct2rs Send+Sync assertion"
```

---

## Task 2: Audio decoding and resampling

**Files:**
- Create: `src/audio/mod.rs`
- Create: `src/audio/decode.rs`
- Create: `src/audio/resample.rs`
- Modify: `src/lib.rs` (add `mod audio;`)

**Interfaces:**
- Consumes: `types::SAMPLE_RATE`, `error::{Error, Result}`.
- Produces: `audio::load_16k_mono(path: &Path) -> Result<Vec<f32>>`, `audio::decode::DecodedAudio { samples: Vec<f32>, sample_rate: u32 }`, `audio::decode::decode_file(path: &Path) -> Result<DecodedAudio>`, `audio::resample::to_16k(samples: Vec<f32>, from_rate: u32) -> Result<Vec<f32>>`.

The test strategy here matters: writing a real WAV to a temp file with `hound` gives an honest round-trip without committing binary fixtures.

- [ ] **Step 1: Write the failing tests**

Create `src/audio/mod.rs`:

```rust
//! Audio loading: any supported container to 16 kHz mono f32.

pub mod decode;
pub mod resample;

use crate::error::Result;
use std::path::Path;

/// Decode `path` and return 16 kHz mono f32 samples in [-1, 1].
pub fn load_16k_mono(path: &Path) -> Result<Vec<f32>> {
    let decoded = decode::decode_file(path)?;
    resample::to_16k(decoded.samples, decoded.sample_rate)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Write a mono WAV of a 440 Hz sine at `rate` Hz for `secs` seconds.
    fn write_sine_wav(path: &Path, rate: u32, secs: f32, channels: u16) {
        let spec = hound::WavSpec {
            channels,
            sample_rate: rate,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        let mut writer = hound::WavWriter::create(path, spec).unwrap();
        let frames = (rate as f32 * secs) as usize;
        for i in 0..frames {
            let t = i as f32 / rate as f32;
            let v = (t * 440.0 * std::f32::consts::TAU).sin() * 0.5;
            for _ in 0..channels {
                writer.write_sample(v).unwrap();
            }
        }
        writer.finalize().unwrap();
    }

    #[test]
    fn loads_16k_mono_wav_unchanged_in_length() {
        let dir = std::env::temp_dir().join("whisper_rs_t2_a");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("mono16k.wav");
        write_sine_wav(&path, 16_000, 1.0, 1);

        let samples = load_16k_mono(&path).unwrap();

        assert!(
            (samples.len() as i64 - 16_000).abs() <= 1,
            "expected ~16000 samples, got {}",
            samples.len()
        );
        assert!(samples.iter().all(|s| s.abs() <= 1.0), "samples must be normalised");
    }

    #[test]
    fn resamples_44100_to_16k() {
        let dir = std::env::temp_dir().join("whisper_rs_t2_b");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("mono44k.wav");
        write_sine_wav(&path, 44_100, 1.0, 1);

        let samples = load_16k_mono(&path).unwrap();

        // One second in, so within 1% of 16000 samples.
        let diff = (samples.len() as f32 - 16_000.0).abs();
        assert!(diff < 160.0, "expected ~16000 samples, got {}", samples.len());
    }

    #[test]
    fn downmixes_stereo_to_mono() {
        let dir = std::env::temp_dir().join("whisper_rs_t2_c");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("stereo16k.wav");
        write_sine_wav(&path, 16_000, 1.0, 2);

        let samples = load_16k_mono(&path).unwrap();

        assert!(
            (samples.len() as i64 - 16_000).abs() <= 1,
            "stereo must collapse to one channel, got {} samples",
            samples.len()
        );
    }

    #[test]
    fn missing_file_is_an_audio_read_error() {
        let err = load_16k_mono(Path::new("/nonexistent/nope.wav")).unwrap_err();
        assert!(matches!(err, crate::error::Error::AudioRead { .. }), "got {err:?}");
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test audio::`
Expected: FAIL — `decode` and `resample` modules do not exist yet.

- [ ] **Step 3: Implement decode.rs**

```rust
//! symphonia-backed decoding to interleaved f32, downmixed to mono.

use crate::error::{Error, Result};
use std::fs::File;
use std::path::Path;
use symphonia::core::audio::{AudioBufferRef, Signal};
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::probe::Hint;

pub struct DecodedAudio {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

pub fn decode_file(path: &Path) -> Result<DecodedAudio> {
    let file = File::open(path).map_err(|e| Error::AudioRead {
        path: path.to_path_buf(),
        message: e.to_string(),
    })?;

    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &Default::default(), &Default::default())
        .map_err(|e| Error::AudioFormat {
            path: path.to_path_buf(),
            message: e.to_string(),
        })?;

    let mut format = probed.format;
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| Error::AudioEmpty {
            path: path.to_path_buf(),
        })?;
    let track_id = track.id;

    let sample_rate = track.codec_params.sample_rate.ok_or_else(|| Error::AudioFormat {
        path: path.to_path_buf(),
        message: "stream declares no sample rate".into(),
    })?;

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| Error::AudioFormat {
            path: path.to_path_buf(),
            message: e.to_string(),
        })?;

    let mut out: Vec<f32> = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            // End of stream is signalled as an IO error by symphonia.
            Err(symphonia::core::errors::Error::IoError(_)) => break,
            Err(symphonia::core::errors::Error::ResetRequired) => break,
            Err(e) => {
                return Err(Error::AudioRead {
                    path: path.to_path_buf(),
                    message: e.to_string(),
                })
            }
        };

        if packet.track_id() != track_id {
            continue;
        }

        match decoder.decode(&packet) {
            Ok(buf) => push_mono(&buf, &mut out),
            // A corrupt packet mid-file should not lose the whole file.
            Err(symphonia::core::errors::Error::DecodeError(_)) => continue,
            Err(e) => {
                return Err(Error::AudioRead {
                    path: path.to_path_buf(),
                    message: e.to_string(),
                })
            }
        }
    }

    if out.is_empty() {
        return Err(Error::AudioEmpty {
            path: path.to_path_buf(),
        });
    }

    Ok(DecodedAudio {
        samples: out,
        sample_rate,
    })
}

/// Average all channels of `buf` into `out`, converting to f32 in [-1, 1].
fn push_mono(buf: &AudioBufferRef<'_>, out: &mut Vec<f32>) {
    let mut sample_buf =
        symphonia::core::audio::SampleBuffer::<f32>::new(buf.capacity() as u64, *buf.spec());
    sample_buf.copy_interleaved_ref(buf.clone());

    let channels = buf.spec().channels.count().max(1);
    let interleaved = sample_buf.samples();

    for frame in interleaved.chunks(channels) {
        let sum: f32 = frame.iter().sum();
        out.push(sum / channels as f32);
    }
}
```

- [ ] **Step 4: Implement resample.rs**

```rust
//! rubato-backed resampling to SAMPLE_RATE.

use crate::error::{Error, Result};
use crate::types::SAMPLE_RATE;
use rubato::{Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction};

/// Resample mono `samples` from `from_rate` to SAMPLE_RATE.
/// Returns the input untouched when the rate already matches.
pub fn to_16k(samples: Vec<f32>, from_rate: u32) -> Result<Vec<f32>> {
    if from_rate as usize == SAMPLE_RATE {
        return Ok(samples);
    }
    if samples.is_empty() {
        return Ok(samples);
    }

    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };

    let chunk = 1024usize;
    let ratio = SAMPLE_RATE as f64 / from_rate as f64;
    let mut resampler = SincFixedIn::<f32>::new(ratio, 2.0, params, chunk, 1)
        .map_err(|e| Error::Resample(e.to_string()))?;

    let mut out: Vec<f32> = Vec::with_capacity((samples.len() as f64 * ratio) as usize + chunk);
    let mut pos = 0usize;

    while pos < samples.len() {
        let end = (pos + chunk).min(samples.len());
        let mut block = samples[pos..end].to_vec();
        // The final block must be padded to the fixed chunk size.
        block.resize(chunk, 0.0);

        let produced = resampler
            .process(&[block], None)
            .map_err(|e| Error::Resample(e.to_string()))?;
        out.extend_from_slice(&produced[0]);

        pos = end;
    }

    // Trim the tail produced by zero padding the last block.
    let expected = (samples.len() as f64 * ratio).round() as usize;
    out.truncate(expected.min(out.len()));

    Ok(out)
}
```

- [ ] **Step 5: Add the module to lib.rs**

Add `mod audio;` to `src/lib.rs`, keeping the existing `mod` lines.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `cargo test audio::`
Expected: 4 tests PASS.

Run: `cargo clippy --all-targets -- -D warnings`
Expected: no warnings.

- [ ] **Step 7: Commit**

```bash
git add src/audio src/lib.rs Cargo.toml Cargo.lock
git commit -m "feat: decode and resample audio to 16 kHz mono in Rust"
```

---

## Task 3: VAD hysteresis — probabilities to speech regions

This is pure arithmetic over a probability slice and the highest-value test target in the project. No VAD model is involved.

**Files:**
- Create: `src/vad/speech.rs`
- Create: `src/vad/mod.rs` (params only in this task)
- Modify: `src/lib.rs` (add `mod vad;`)

**Interfaces:**
- Consumes: `types::SpeechRegion`.
- Produces: `vad::VadParams { threshold: f32, neg_threshold: f32, min_speech_ms: u32, min_silence_ms: u32, speech_pad_ms: u32, frame_samples: usize }` with `Default`, and `vad::speech::regions_from_probs(probs: &[f32], params: &VadParams, total_samples: usize) -> Vec<SpeechRegion>`.

- [ ] **Step 1: Write vad/mod.rs with the params type only**

```rust
//! Voice activity detection: parameters, region extraction, backend wrapper.

pub mod speech;

use crate::types::SAMPLE_RATE;

#[derive(Debug, Clone)]
pub struct VadParams {
    /// A frame at or above this probability opens a speech region.
    pub threshold: f32,
    /// A region only closes below this. Lower than `threshold` on purpose:
    /// hysteresis stops speech being chopped at every breath.
    pub neg_threshold: f32,
    pub min_speech_ms: u32,
    pub min_silence_ms: u32,
    pub speech_pad_ms: u32,
    /// Samples per probability frame, set by the backend.
    pub frame_samples: usize,
}

impl Default for VadParams {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            neg_threshold: 0.35,
            min_speech_ms: 250,
            min_silence_ms: 2000,
            speech_pad_ms: 400,
            frame_samples: 512,
        }
    }
}

impl VadParams {
    pub fn ms_to_samples(ms: u32) -> usize {
        ms as usize * SAMPLE_RATE / 1000
    }
}
```

- [ ] **Step 2: Write the failing tests**

Create `src/vad/speech.rs` with only the test module and a stub signature:

```rust
//! Pure: per-frame speech probabilities to speech regions.

use super::VadParams;
use crate::types::SpeechRegion;

pub fn regions_from_probs(
    _probs: &[f32],
    _params: &VadParams,
    _total_samples: usize,
) -> Vec<SpeechRegion> {
    todo!("step 4")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Params with no padding and no minimum durations, so tests can check the
    /// hysteresis machine in isolation. 1600 samples per frame = 100 ms.
    fn bare_params() -> VadParams {
        VadParams {
            threshold: 0.5,
            neg_threshold: 0.35,
            min_speech_ms: 0,
            min_silence_ms: 0,
            speech_pad_ms: 0,
            frame_samples: 1600,
        }
    }

    #[test]
    fn all_silence_yields_no_regions() {
        let probs = vec![0.0; 50];
        let r = regions_from_probs(&probs, &bare_params(), 50 * 1600);
        assert!(r.is_empty(), "got {r:?}");
    }

    #[test]
    fn all_speech_yields_one_region_covering_everything() {
        let probs = vec![0.9; 10];
        let r = regions_from_probs(&probs, &bare_params(), 10 * 1600);
        assert_eq!(r, vec![SpeechRegion { start: 0, end: 16_000 }]);
    }

    #[test]
    fn speech_at_the_very_start_is_kept() {
        // speech in frames 0..2, then silence
        let mut probs = vec![0.0; 10];
        probs[0] = 0.9;
        probs[1] = 0.9;
        let r = regions_from_probs(&probs, &bare_params(), 10 * 1600);
        assert_eq!(r, vec![SpeechRegion { start: 0, end: 3_200 }]);
    }

    #[test]
    fn speech_running_to_the_last_frame_is_closed_at_total_samples() {
        let mut probs = vec![0.0; 5];
        probs[3] = 0.9;
        probs[4] = 0.9;
        let total = 5 * 1600;
        let r = regions_from_probs(&probs, &bare_params(), total);
        assert_eq!(r, vec![SpeechRegion { start: 4_800, end: total }]);
    }

    #[test]
    fn dip_between_thresholds_does_not_split_the_region() {
        // 0.9, 0.4 (below threshold but above neg_threshold), 0.9
        let probs = vec![0.9, 0.4, 0.9];
        let r = regions_from_probs(&probs, &bare_params(), 3 * 1600);
        assert_eq!(
            r,
            vec![SpeechRegion { start: 0, end: 4_800 }],
            "a dip inside the hysteresis band must not split speech"
        );
    }

    #[test]
    fn short_silence_below_min_silence_does_not_split_the_region() {
        let params = VadParams {
            min_silence_ms: 300, // 3 frames
            ..bare_params()
        };
        // speech, 2 frames of true silence (200 ms < 300 ms), speech
        let probs = vec![0.9, 0.0, 0.0, 0.9];
        let r = regions_from_probs(&probs, &params, 4 * 1600);
        assert_eq!(r, vec![SpeechRegion { start: 0, end: 6_400 }]);
    }

    #[test]
    fn long_silence_splits_into_two_regions() {
        let params = VadParams {
            min_silence_ms: 200, // 2 frames
            ..bare_params()
        };
        let probs = vec![0.9, 0.0, 0.0, 0.0, 0.9];
        let r = regions_from_probs(&probs, &params, 5 * 1600);
        assert_eq!(
            r,
            vec![
                SpeechRegion { start: 0, end: 1_600 },
                SpeechRegion { start: 6_400, end: 8_000 },
            ]
        );
    }

    #[test]
    fn regions_shorter_than_min_speech_are_discarded() {
        let params = VadParams {
            min_speech_ms: 250, // needs 4000 samples, one frame is 1600
            min_silence_ms: 100,
            ..bare_params()
        };
        // a single 100 ms speech blip
        let probs = vec![0.0, 0.9, 0.0, 0.0, 0.0];
        let r = regions_from_probs(&probs, &params, 5 * 1600);
        assert!(r.is_empty(), "a 100 ms blip must be dropped, got {r:?}");
    }

    #[test]
    fn padding_expands_edges_and_is_clamped_to_the_audio() {
        let params = VadParams {
            speech_pad_ms: 100, // 1600 samples
            ..bare_params()
        };
        // speech only in frame 0
        let probs = vec![0.9, 0.0, 0.0];
        let r = regions_from_probs(&probs, &params, 3 * 1600);
        // start clamps to 0, end grows by 1600
        assert_eq!(r, vec![SpeechRegion { start: 0, end: 3_200 }]);
    }

    #[test]
    fn padding_merges_regions_that_come_to_overlap() {
        let params = VadParams {
            min_silence_ms: 100,
            speech_pad_ms: 200, // 3200 samples each side
            ..bare_params()
        };
        // speech, silence, speech — padding closes the 1600 sample gap
        let probs = vec![0.9, 0.0, 0.9];
        let r = regions_from_probs(&probs, &params, 3 * 1600);
        assert_eq!(
            r,
            vec![SpeechRegion { start: 0, end: 4_800 }],
            "padded regions that overlap must merge"
        );
    }

    #[test]
    fn empty_probs_yields_no_regions() {
        let r = regions_from_probs(&[], &bare_params(), 0);
        assert!(r.is_empty());
    }
}
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `cargo test vad::speech`
Expected: FAIL — all tests panic on `todo!("step 4")`.

- [ ] **Step 4: Implement regions_from_probs**

Replace the stub with:

```rust
/// Turn per-frame speech probabilities into speech regions in sample space.
///
/// Two thresholds (hysteresis): a frame at or above `threshold` opens a region,
/// and the region only closes once the probability has stayed below
/// `neg_threshold` for at least `min_silence_ms`.
pub fn regions_from_probs(
    probs: &[f32],
    params: &VadParams,
    total_samples: usize,
) -> Vec<SpeechRegion> {
    let frame = params.frame_samples;
    let min_silence = VadParams::ms_to_samples(params.min_silence_ms);
    let min_speech = VadParams::ms_to_samples(params.min_speech_ms);
    let pad = VadParams::ms_to_samples(params.speech_pad_ms);

    let sample_at = |i: usize| (i * frame).min(total_samples);

    let mut regions: Vec<SpeechRegion> = Vec::new();
    let mut start: Option<usize> = None;
    // First frame index of the current below-neg_threshold run, if any.
    let mut silence_from: Option<usize> = None;

    for (i, &p) in probs.iter().enumerate() {
        if p >= params.threshold {
            if start.is_none() {
                start = Some(i);
            }
            silence_from = None;
        } else if p < params.neg_threshold {
            if let Some(s) = start {
                let run_start = *silence_from.get_or_insert(i);
                let silence_len = sample_at(i + 1) - sample_at(run_start);
                if silence_len >= min_silence {
                    regions.push(SpeechRegion {
                        start: sample_at(s),
                        end: sample_at(run_start),
                    });
                    start = None;
                    silence_from = None;
                }
            }
        }
        // Between neg_threshold and threshold: hold the current state.
    }

    if let Some(s) = start {
        regions.push(SpeechRegion {
            start: sample_at(s),
            end: total_samples,
        });
    }

    regions.retain(|r| r.len() >= min_speech && !r.is_empty());

    if pad > 0 {
        for r in &mut regions {
            r.start = r.start.saturating_sub(pad);
            r.end = (r.end + pad).min(total_samples);
        }
    }

    merge_overlapping(regions)
}

/// Merge regions that touch or overlap. Input must be sorted by `start`.
fn merge_overlapping(regions: Vec<SpeechRegion>) -> Vec<SpeechRegion> {
    let mut out: Vec<SpeechRegion> = Vec::with_capacity(regions.len());
    for r in regions {
        match out.last_mut() {
            Some(prev) if r.start <= prev.end => prev.end = prev.end.max(r.end),
            _ => out.push(r),
        }
    }
    out
}
```

- [ ] **Step 5: Add the module to lib.rs**

Add `mod vad;` to `src/lib.rs`.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `cargo test vad::speech`
Expected: 11 tests PASS.

If `dip_between_thresholds_does_not_split_the_region` or `short_silence_below_min_silence_does_not_split_the_region` fails, the bug is in the silence-run accounting, not in the test — the run length must be measured from the first below-`neg_threshold` frame, not from the current one.

Run: `cargo clippy --all-targets -- -D warnings`
Expected: no warnings.

- [ ] **Step 7: Commit**

```bash
git add src/vad src/lib.rs
git commit -m "feat: extract speech regions from VAD probabilities with hysteresis"
```

---

## Task 4: VAD backend wrapper

**Files:**
- Modify: `src/vad/mod.rs`

**Interfaces:**
- Consumes: `VadParams`, `speech::regions_from_probs`, `error::{Error, Result}`.
- Produces: `vad::Vad` trait with `fn probabilities(&mut self, samples: &[f32]) -> Result<Vec<f32>>` and `fn frame_samples(&self) -> usize`; `vad::SileroBackend::new() -> Result<SileroBackend>`; `vad::detect(vad: &mut dyn Vad, samples: &[f32], params: &VadParams) -> Result<(Vec<f32>, Vec<SpeechRegion>)>` returning the probabilities alongside the regions — Task 5 needs the probabilities to pick split points.

- [ ] **Step 1: Write the failing test**

Append to `src/vad/mod.rs`:

```rust
use crate::error::Result;
use crate::types::SpeechRegion;

/// A voice activity detector producing one probability per frame.
pub trait Vad {
    fn probabilities(&mut self, samples: &[f32]) -> Result<Vec<f32>>;
    fn frame_samples(&self) -> usize;
}

/// Run `vad` over `samples` and return both the raw probabilities and the
/// speech regions. Task 5 uses the probabilities to choose split points, so
/// they are returned rather than discarded.
pub fn detect(
    vad: &mut dyn Vad,
    samples: &[f32],
    params: &VadParams,
) -> Result<(Vec<f32>, Vec<SpeechRegion>)> {
    let mut params = params.clone();
    params.frame_samples = vad.frame_samples();
    let probs = vad.probabilities(samples)?;
    let regions = speech::regions_from_probs(&probs, &params, samples.len());
    Ok((probs, regions))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fake VAD: speech wherever the sample magnitude exceeds 0.1.
    /// Lets `detect` be tested with no model and no ONNX runtime.
    struct MagnitudeVad {
        frame: usize,
    }

    impl Vad for MagnitudeVad {
        fn probabilities(&mut self, samples: &[f32]) -> Result<Vec<f32>> {
            Ok(samples
                .chunks(self.frame)
                .map(|c| {
                    let peak = c.iter().fold(0.0f32, |a, s| a.max(s.abs()));
                    if peak > 0.1 { 0.9 } else { 0.0 }
                })
                .collect())
        }

        fn frame_samples(&self) -> usize {
            self.frame
        }
    }

    #[test]
    fn detect_returns_probabilities_and_regions() {
        let frame = 1600;
        let mut samples = vec![0.0f32; frame * 5];
        // loud in frames 1 and 2
        for s in samples[frame..frame * 3].iter_mut() {
            *s = 0.8;
        }

        let mut vad = MagnitudeVad { frame };
        let params = VadParams {
            min_speech_ms: 0,
            min_silence_ms: 0,
            speech_pad_ms: 0,
            frame_samples: frame,
            ..VadParams::default()
        };

        let (probs, regions) = detect(&mut vad, &samples, &params).unwrap();

        assert_eq!(probs.len(), 5, "one probability per frame");
        assert_eq!(regions, vec![SpeechRegion { start: frame, end: frame * 3 }]);
    }

    #[test]
    fn detect_overrides_frame_samples_from_the_backend() {
        let mut vad = MagnitudeVad { frame: 512 };
        // Params claim 1600, the backend says 512; the backend wins.
        let params = VadParams { frame_samples: 1600, ..VadParams::default() };
        let samples = vec![0.0f32; 512 * 4];

        let (probs, _) = detect(&mut vad, &samples, &params).unwrap();

        assert_eq!(probs.len(), 4, "frame size must come from the backend");
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test vad::tests`
Expected: FAIL to compile — the `Vad` trait and `detect` do not exist until Step 1's code is in place; once it is, these two tests should already pass, because `detect` is fully implemented above. Run them and confirm PASS before moving on.

- [ ] **Step 3: Implement the Silero backend**

Append to `src/vad/mod.rs`. Consult the `wavekat-vad` docs for the exact constructor and frame size — the crate exposes `SileroVad::new(sample_rate)` and a `VoiceActivityDetector::process(&samples, sample_rate)` returning one probability per call, plus `FrameAdapter` for arbitrary chunk sizes.

```rust
/// Silero VAD via wavekat-vad. 16 kHz only.
pub struct SileroBackend {
    inner: wavekat_vad::SileroVad,
    frame: usize,
}

impl SileroBackend {
    pub fn new() -> Result<Self> {
        let inner = wavekat_vad::SileroVad::new(crate::types::SAMPLE_RATE as u32)
            .map_err(|e| crate::error::Error::Vad(e.to_string()))?;
        // Silero's native window at 16 kHz.
        Ok(Self { inner, frame: 512 })
    }
}

impl Vad for SileroBackend {
    fn probabilities(&mut self, samples: &[f32]) -> Result<Vec<f32>> {
        use wavekat_vad::VoiceActivityDetector;

        let mut out = Vec::with_capacity(samples.len() / self.frame + 1);
        for chunk in samples.chunks(self.frame) {
            // The model needs a full frame; pad the tail with silence.
            let p = if chunk.len() == self.frame {
                self.inner
                    .process(chunk, crate::types::SAMPLE_RATE as u32)
                    .map_err(|e| crate::error::Error::Vad(e.to_string()))?
            } else {
                let mut padded = chunk.to_vec();
                padded.resize(self.frame, 0.0);
                self.inner
                    .process(&padded, crate::types::SAMPLE_RATE as u32)
                    .map_err(|e| crate::error::Error::Vad(e.to_string()))?
            };
            out.push(p);
        }
        Ok(out)
    }

    fn frame_samples(&self) -> usize {
        self.frame
    }
}
```

If `SileroVad::new` has a different signature or the crate requires `FrameAdapter` to feed frames, adapt this wrapper — the `Vad` trait boundary is what the rest of the crate depends on, and nothing else changes.

- [ ] **Step 4: Add a smoke test for the real backend**

Append to the `tests` module in `src/vad/mod.rs`:

```rust
    #[test]
    fn silero_backend_produces_one_probability_per_frame() {
        let mut vad = match SileroBackend::new() {
            Ok(v) => v,
            // The ONNX model ships with the crate; if loading fails, that is a
            // real failure, not something to skip.
            Err(e) => panic!("SileroBackend::new failed: {e}"),
        };
        let frame = vad.frame_samples();
        let samples = vec![0.0f32; frame * 3];

        let probs = vad.probabilities(&samples).unwrap();

        assert_eq!(probs.len(), 3);
        assert!(
            probs.iter().all(|p| (0.0..=1.0).contains(p)),
            "probabilities must be in [0, 1], got {probs:?}"
        );
        assert!(
            probs.iter().all(|&p| p < 0.5),
            "pure silence must not be detected as speech, got {probs:?}"
        );
    }
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cargo test vad::`
Expected: all Task 3 and Task 4 tests PASS.

Run: `cargo clippy --all-targets -- -D warnings`
Expected: no warnings.

- [ ] **Step 6: Commit**

```bash
git add src/vad/mod.rs Cargo.toml Cargo.lock
git commit -m "feat: add Vad trait and wavekat-vad Silero backend"
```

---

## Task 5: Windowing — regions to 30 s windows

**Files:**
- Create: `src/chunk.rs`
- Modify: `src/lib.rs` (add `mod chunk;`)

**Interfaces:**
- Consumes: `types::{SpeechRegion, Window, WINDOW_SAMPLES}`, `vad::VadParams` (for `frame_samples`).
- Produces: `chunk::plan_windows(regions: &[SpeechRegion], probs: &[f32], frame_samples: usize) -> Vec<(usize, usize)>` (inclusive-exclusive sample ranges), and `chunk::build_window(samples: &[f32], range: (usize, usize)) -> Window`.

Splitting planning from copying keeps every invariant testable without allocating audio.

- [ ] **Step 1: Write the failing tests**

Create `src/chunk.rs`:

```rust
//! Pack speech regions into decoder windows of at most WINDOW_SAMPLES.

use crate::types::{SpeechRegion, Window, WINDOW_SAMPLES};

/// Plan window boundaries as (start, end) sample ranges. Pure.
pub fn plan_windows(
    _regions: &[SpeechRegion],
    _probs: &[f32],
    _frame_samples: usize,
) -> Vec<(usize, usize)> {
    todo!("step 3")
}

/// Copy `range` out of `samples` and zero-pad to exactly WINDOW_SAMPLES.
pub fn build_window(_samples: &[f32], _range: (usize, usize)) -> Window {
    todo!("step 3")
}

#[cfg(test)]
mod tests {
    use super::*;

    const S: usize = crate::types::SAMPLE_RATE; // 1 second

    #[test]
    fn no_regions_yields_no_windows() {
        assert!(plan_windows(&[], &[], 512).is_empty());
    }

    #[test]
    fn several_short_regions_pack_into_one_window() {
        let regions = vec![
            SpeechRegion { start: 0, end: 5 * S },
            SpeechRegion { start: 7 * S, end: 12 * S },
            SpeechRegion { start: 15 * S, end: 20 * S },
        ];
        let w = plan_windows(&regions, &[], 512);
        assert_eq!(w, vec![(0, 20 * S)], "all within 30 s, so one window");
    }

    #[test]
    fn regions_crossing_thirty_seconds_start_a_new_window() {
        let regions = vec![
            SpeechRegion { start: 0, end: 20 * S },
            SpeechRegion { start: 25 * S, end: 40 * S },
        ];
        let w = plan_windows(&regions, &[], 512);
        assert_eq!(
            w,
            vec![(0, 20 * S), (25 * S, 40 * S)],
            "the second region would exceed 30 s from offset 0"
        );
    }

    #[test]
    fn every_window_respects_the_thirty_second_ceiling() {
        let regions = vec![
            SpeechRegion { start: 0, end: 10 * S },
            SpeechRegion { start: 11 * S, end: 29 * S },
            SpeechRegion { start: 30 * S, end: 95 * S },
        ];
        let w = plan_windows(&regions, &[], 512);
        for (start, end) in &w {
            assert!(
                end - start <= WINDOW_SAMPLES,
                "window {start}..{end} is longer than 30 s"
            );
        }
    }

    #[test]
    fn window_offsets_are_strictly_increasing() {
        let regions = vec![
            SpeechRegion { start: 0, end: 95 * S },
            SpeechRegion { start: 100 * S, end: 130 * S },
        ];
        let w = plan_windows(&regions, &[], 512);
        for pair in w.windows(2) {
            assert!(pair[0].0 < pair[1].0, "offsets must increase: {w:?}");
        }
    }

    #[test]
    fn a_ninety_second_region_becomes_three_windows() {
        let regions = vec![SpeechRegion { start: 0, end: 90 * S }];
        let w = plan_windows(&regions, &[], 512);
        assert_eq!(w.len(), 3, "got {w:?}");
        assert_eq!(w[0].0, 0);
        assert_eq!(w.last().unwrap().1, 90 * S, "the tail must not be lost");
    }

    #[test]
    fn no_speech_sample_is_lost() {
        let regions = vec![
            SpeechRegion { start: 3 * S, end: 40 * S },
            SpeechRegion { start: 50 * S, end: 55 * S },
        ];
        let w = plan_windows(&regions, &[], 512);

        for r in &regions {
            for sample in [r.start, (r.start + r.end) / 2, r.end - 1] {
                assert!(
                    w.iter().any(|&(s, e)| sample >= s && sample < e),
                    "sample {sample} of region {r:?} is in no window: {w:?}"
                );
            }
        }
    }

    #[test]
    fn no_speech_sample_appears_in_two_windows() {
        let regions = vec![SpeechRegion { start: 0, end: 95 * S }];
        let w = plan_windows(&regions, &[], 512);
        for pair in w.windows(2) {
            assert!(
                pair[0].1 <= pair[1].0,
                "windows overlap: {:?} and {:?}",
                pair[0],
                pair[1]
            );
        }
    }

    #[test]
    fn long_region_splits_at_the_lowest_probability_in_the_last_two_seconds() {
        let frame = S; // 1 s frames keep the arithmetic obvious
        // 40 s region. Candidate cut window is 28..30 s, i.e. frames 28 and 29.
        let mut probs = vec![0.9f32; 40];
        probs[28] = 0.1; // the quietest candidate
        let regions = vec![SpeechRegion { start: 0, end: 40 * S }];

        let w = plan_windows(&regions, &probs, frame);

        assert_eq!(w[0], (0, 28 * S), "must cut at the quietest frame, got {w:?}");
        assert_eq!(w[1].0, 28 * S, "the next window resumes at the cut");
    }

    #[test]
    fn long_region_falls_back_to_a_hard_cut_without_probabilities() {
        let regions = vec![SpeechRegion { start: 0, end: 40 * S }];
        let w = plan_windows(&regions, &[], 512);
        assert_eq!(w[0], (0, WINDOW_SAMPLES), "no probs means cut at exactly 30 s");
    }

    #[test]
    fn build_window_pads_to_exactly_thirty_seconds() {
        let samples = vec![0.5f32; 10 * S];
        let win = build_window(&samples, (2 * S, 7 * S));

        assert_eq!(win.offset, 2 * S);
        assert_eq!(win.real_len, 5 * S);
        assert_eq!(win.samples.len(), WINDOW_SAMPLES, "must be padded to 30 s");
        assert!(win.samples[..5 * S].iter().all(|&s| s == 0.5));
        assert!(win.samples[5 * S..].iter().all(|&s| s == 0.0), "tail must be silence");
    }

    #[test]
    fn build_window_clamps_a_range_past_the_end_of_the_audio() {
        let samples = vec![0.5f32; 3 * S];
        let win = build_window(&samples, (2 * S, 9 * S));

        assert_eq!(win.real_len, S, "only one second of audio actually exists");
        assert_eq!(win.samples.len(), WINDOW_SAMPLES);
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test chunk::`
Expected: FAIL — every test panics on `todo!("step 3")`.

- [ ] **Step 3: Implement plan_windows and build_window**

Replace both stubs with:

```rust
/// Samples searched for a good split point at the end of an over-long window.
const CUT_SEARCH_SAMPLES: usize = 2 * crate::types::SAMPLE_RATE;

/// Plan window boundaries as (start, end) sample ranges.
///
/// Regions are packed in order while they fit inside WINDOW_SAMPLES from the
/// current offset. A single region longer than the ceiling is sliced, cutting at
/// the lowest speech probability inside the last CUT_SEARCH_SAMPLES of the
/// window — a slightly early cut in a quiet moment beats a hard cut mid-word.
pub fn plan_windows(
    regions: &[SpeechRegion],
    probs: &[f32],
    frame_samples: usize,
) -> Vec<(usize, usize)> {
    let mut out: Vec<(usize, usize)> = Vec::new();
    let mut current: Option<(usize, usize)> = None;

    for region in regions {
        if region.is_empty() {
            continue;
        }

        // Flush the accumulator when this region cannot fit.
        if let Some((start, end)) = current {
            if region.end - start > WINDOW_SAMPLES {
                out.push((start, end));
                current = None;
            }
        }

        if current.is_none() && region.len() > WINDOW_SAMPLES {
            // Slice the over-long region on its own.
            let mut pos = region.start;
            while region.end - pos > WINDOW_SAMPLES {
                let cut = choose_cut(pos, probs, frame_samples);
                out.push((pos, cut));
                pos = cut;
            }
            current = Some((pos, region.end));
            continue;
        }

        current = match current {
            Some((start, _)) => Some((start, region.end)),
            None => Some((region.start, region.end)),
        };
    }

    if let Some(range) = current {
        out.push(range);
    }

    out
}

/// Pick the end of a window starting at `start`, at or before start+WINDOW_SAMPLES.
fn choose_cut(start: usize, probs: &[f32], frame_samples: usize) -> usize {
    let hard = start + WINDOW_SAMPLES;
    if probs.is_empty() || frame_samples == 0 {
        return hard;
    }

    let search_from = hard - CUT_SEARCH_SAMPLES;
    let first = search_from / frame_samples;
    let last = (hard / frame_samples).min(probs.len());
    if first >= last {
        return hard;
    }

    let quietest = (first..last)
        .min_by(|&a, &b| probs[a].partial_cmp(&probs[b]).unwrap_or(std::cmp::Ordering::Equal));

    match quietest {
        Some(i) => {
            let cut = i * frame_samples;
            // Never go backwards or produce an empty window.
            if cut > start { cut } else { hard }
        }
        None => hard,
    }
}

/// Copy `range` out of `samples` and zero-pad to exactly WINDOW_SAMPLES.
pub fn build_window(samples: &[f32], range: (usize, usize)) -> Window {
    let (start, end) = range;
    let start = start.min(samples.len());
    let end = end.min(samples.len()).max(start);

    let mut buf = Vec::with_capacity(WINDOW_SAMPLES);
    buf.extend_from_slice(&samples[start..end]);
    let real_len = buf.len().min(WINDOW_SAMPLES);
    buf.truncate(WINDOW_SAMPLES);
    buf.resize(WINDOW_SAMPLES, 0.0);

    Window {
        offset: start,
        samples: buf,
        real_len,
    }
}
```

- [ ] **Step 4: Add the module to lib.rs**

Add `mod chunk;` to `src/lib.rs`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cargo test chunk::`
Expected: 12 tests PASS.

Run: `cargo clippy --all-targets -- -D warnings`
Expected: no warnings.

- [ ] **Step 6: Commit**

```bash
git add src/chunk.rs src/lib.rs
git commit -m "feat: pack speech regions into 30 s decoder windows"
```

---

## Task 6: Stitching windows into a global timeline

**Files:**
- Create: `src/stitch.rs`
- Modify: `src/lib.rs` (add `mod stitch;`)

**Interfaces:**
- Consumes: `types::{Seg, Word, Window, SAMPLE_RATE}`.
- Produces: `stitch::stitch(window: &Window, segs: Vec<Seg>, next_id: &mut u32) -> Vec<Seg>`.

- [ ] **Step 1: Write the failing tests**

Create `src/stitch.rs`:

```rust
//! Shift per-window segments onto the global timeline.

use crate::types::{Seg, Window, SAMPLE_RATE};

pub fn stitch(_window: &Window, _segs: Vec<Seg>, _next_id: &mut u32) -> Vec<Seg> {
    todo!("step 3")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Word, WINDOW_SAMPLES};

    fn window(offset_secs: f32, real_secs: f32) -> Window {
        Window {
            offset: (offset_secs * SAMPLE_RATE as f32) as usize,
            samples: vec![0.0; WINDOW_SAMPLES],
            real_len: (real_secs * SAMPLE_RATE as f32) as usize,
        }
    }

    fn seg(start: f32, end: f32, text: &str) -> Seg {
        Seg { id: 0, start, end, text: text.into(), words: None }
    }

    #[test]
    fn timestamps_are_shifted_by_the_window_offset() {
        let w = window(60.0, 30.0);
        let mut id = 0;
        let out = stitch(&w, vec![seg(1.0, 2.5, "hello")], &mut id);

        assert_eq!(out.len(), 1);
        assert!((out[0].start - 61.0).abs() < 1e-4, "got {}", out[0].start);
        assert!((out[0].end - 62.5).abs() < 1e-4, "got {}", out[0].end);
    }

    #[test]
    fn ids_are_sequential_across_windows() {
        let mut id = 0;
        let a = stitch(&window(0.0, 30.0), vec![seg(0.0, 1.0, "a"), seg(1.0, 2.0, "b")], &mut id);
        let b = stitch(&window(30.0, 30.0), vec![seg(0.0, 1.0, "c")], &mut id);

        assert_eq!(a.iter().map(|s| s.id).collect::<Vec<_>>(), vec![0, 1]);
        assert_eq!(b[0].id, 2, "numbering must continue across windows");
        assert_eq!(id, 3, "the counter must be left ready for the next window");
    }

    #[test]
    fn segments_starting_inside_the_padding_are_dropped() {
        // Only 10 s of real audio in this window.
        let w = window(0.0, 10.0);
        let mut id = 0;
        let out = stitch(
            &w,
            vec![seg(2.0, 4.0, "real"), seg(12.0, 14.0, "hallucinated in padding")],
            &mut id,
        );

        assert_eq!(out.len(), 1, "got {out:?}");
        assert_eq!(out[0].text, "real");
    }

    #[test]
    fn segment_end_is_clamped_to_the_real_window_end() {
        let w = window(0.0, 10.0);
        let mut id = 0;
        let out = stitch(&w, vec![seg(9.0, 25.0, "runs into padding")], &mut id);

        assert_eq!(out.len(), 1);
        assert!((out[0].end - 10.0).abs() < 1e-4, "end must clamp to 10 s, got {}", out[0].end);
    }

    #[test]
    fn word_timestamps_are_shifted_and_clamped_too() {
        let w = window(10.0, 10.0);
        let mut id = 0;
        let segs = vec![Seg {
            id: 0,
            start: 1.0,
            end: 12.0,
            text: "two words".into(),
            words: Some(vec![
                Word { start: 1.0, end: 1.5, text: "two".into(), probability: 0.9 },
                Word { start: 9.5, end: 12.0, text: "words".into(), probability: 0.8 },
            ]),
        }];

        let out = stitch(&w, segs, &mut id);
        let words = out[0].words.as_ref().unwrap();

        assert!((words[0].start - 11.0).abs() < 1e-4, "got {}", words[0].start);
        assert!((words[1].end - 20.0).abs() < 1e-4, "word end must clamp, got {}", words[1].end);
    }

    #[test]
    fn no_segments_in_yields_no_segments_out() {
        let mut id = 5;
        let out = stitch(&window(0.0, 30.0), vec![], &mut id);
        assert!(out.is_empty());
        assert_eq!(id, 5, "the counter must not move");
    }

    #[test]
    fn empty_text_segments_are_dropped() {
        let mut id = 0;
        let out = stitch(&window(0.0, 30.0), vec![seg(0.0, 1.0, "   ")], &mut id);
        assert!(out.is_empty(), "whitespace-only segments carry no information");
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test stitch::`
Expected: FAIL on `todo!("step 3")`.

- [ ] **Step 3: Implement stitch**

```rust
/// Shift `segs` (window-relative seconds) onto the global timeline.
///
/// Drops segments that start inside the zero padding, clamps ends to the real
/// window end, drops blank segments, and renumbers ids from `next_id`.
pub fn stitch(window: &Window, segs: Vec<Seg>, next_id: &mut u32) -> Vec<Seg> {
    let offset = window.offset as f32 / SAMPLE_RATE as f32;
    let real = window.real_len as f32 / SAMPLE_RATE as f32;
    let limit = offset + real;

    let mut out = Vec::with_capacity(segs.len());

    for seg in segs {
        if seg.start >= real {
            // Started in the padding: the model invented it.
            continue;
        }
        if seg.text.trim().is_empty() {
            continue;
        }

        let words = seg.words.map(|ws| {
            ws.into_iter()
                .filter(|w| w.start < real)
                .map(|w| crate::types::Word {
                    start: offset + w.start,
                    end: (offset + w.end).min(limit),
                    text: w.text,
                    probability: w.probability,
                })
                .collect()
        });

        out.push(Seg {
            id: *next_id,
            start: offset + seg.start,
            end: (offset + seg.end).min(limit),
            text: seg.text,
            words,
        });
        *next_id += 1;
    }

    out
}
```

- [ ] **Step 4: Add the module to lib.rs**

Add `mod stitch;` to `src/lib.rs`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cargo test stitch::`
Expected: 7 tests PASS.

Run: `cargo clippy --all-targets -- -D warnings`
Expected: no warnings.

- [ ] **Step 6: Commit**

```bash
git add src/stitch.rs src/lib.rs
git commit -m "feat: stitch per-window segments onto a global timeline"
```

---

## Task 7: Model registry

**Files:**
- Create: `src/models/mod.rs`
- Create: `src/models/registry.rs`
- Modify: `src/lib.rs` (add `mod models;`)

**Interfaces:**
- Consumes: nothing beyond std.
- Produces: `models::registry::ModelRef` (enum with `Local(PathBuf)` and `Hub { repo: String }`) and `models::registry::resolve(name: &str) -> ModelRef`.

- [ ] **Step 1: Write the failing tests**

Create `src/models/mod.rs`:

```rust
//! Model resolution, download, and cache handling.

pub mod registry;
```

Create `src/models/registry.rs`:

```rust
//! Map a user-supplied model name to either a local directory or an HF repo.

use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelRef {
    Local(PathBuf),
    Hub { repo: String },
}

pub fn resolve(_name: &str) -> ModelRef {
    todo!("step 3")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_aliases_map_to_systran_repos() {
        assert_eq!(
            resolve("large-v3"),
            ModelRef::Hub { repo: "Systran/faster-whisper-large-v3".into() }
        );
        assert_eq!(
            resolve("tiny"),
            ModelRef::Hub { repo: "Systran/faster-whisper-tiny".into() }
        );
        assert_eq!(
            resolve("medium"),
            ModelRef::Hub { repo: "Systran/faster-whisper-medium".into() }
        );
    }

    #[test]
    fn distil_alias_maps_to_the_distil_whisper_org() {
        assert_eq!(
            resolve("distil-large-v3"),
            ModelRef::Hub { repo: "distil-whisper/distil-large-v3-ct2".into() }
        );
    }

    #[test]
    fn a_name_with_a_slash_is_an_explicit_repo() {
        assert_eq!(
            resolve("someone/my-whisper-ct2"),
            ModelRef::Hub { repo: "someone/my-whisper-ct2".into() }
        );
    }

    #[test]
    fn an_existing_directory_wins_over_everything() {
        let dir = std::env::temp_dir().join("whisper_rs_t7_tiny");
        std::fs::create_dir_all(&dir).unwrap();

        let resolved = resolve(dir.to_str().unwrap());

        assert_eq!(resolved, ModelRef::Local(dir));
    }

    #[test]
    fn an_unknown_name_without_a_slash_is_still_treated_as_a_repo() {
        // Better a clear 404 from the hub than a silent wrong alias.
        assert_eq!(
            resolve("not-a-real-model"),
            ModelRef::Hub { repo: "not-a-real-model".into() }
        );
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test models::registry`
Expected: FAIL on `todo!("step 3")`.

- [ ] **Step 3: Implement resolve**

Replace the stub with:

```rust
/// Alias to Hugging Face repo. Resolution order is local path, alias, then repo.
const ALIASES: &[(&str, &str)] = &[
    ("tiny", "Systran/faster-whisper-tiny"),
    ("tiny.en", "Systran/faster-whisper-tiny.en"),
    ("base", "Systran/faster-whisper-base"),
    ("base.en", "Systran/faster-whisper-base.en"),
    ("small", "Systran/faster-whisper-small"),
    ("small.en", "Systran/faster-whisper-small.en"),
    ("medium", "Systran/faster-whisper-medium"),
    ("medium.en", "Systran/faster-whisper-medium.en"),
    ("large-v1", "Systran/faster-whisper-large-v1"),
    ("large-v2", "Systran/faster-whisper-large-v2"),
    ("large-v3", "Systran/faster-whisper-large-v3"),
    ("large", "Systran/faster-whisper-large-v3"),
    ("distil-large-v3", "distil-whisper/distil-large-v3-ct2"),
];

pub fn resolve(name: &str) -> ModelRef {
    let as_path = Path::new(name);
    if as_path.is_dir() {
        return ModelRef::Local(as_path.to_path_buf());
    }

    if let Some((_, repo)) = ALIASES.iter().find(|(alias, _)| *alias == name) {
        return ModelRef::Hub {
            repo: (*repo).to_string(),
        };
    }

    ModelRef::Hub {
        repo: name.to_string(),
    }
}
```

- [ ] **Step 4: Add the module to lib.rs**

Add `mod models;` to `src/lib.rs`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cargo test models::registry`
Expected: 5 tests PASS.

Run: `cargo clippy --all-targets -- -D warnings`
Expected: no warnings.

- [ ] **Step 6: Commit**

```bash
git add src/models src/lib.rs
git commit -m "feat: resolve model names to local paths or Hugging Face repos"
```

---

## Task 8: Model download and validation

**Files:**
- Create: `src/models/hub.rs`
- Modify: `src/models/mod.rs` (add `pub mod hub;`)

**Interfaces:**
- Consumes: `models::registry::{ModelRef, resolve}`, `error::{Error, Result}`.
- Produces: `models::hub::FetchOptions { download_root: Option<PathBuf>, local_files_only: bool }` with `Default`, `models::hub::ensure_model(name: &str, opts: &FetchOptions) -> Result<PathBuf>`, and `models::hub::validate_dir(name: &str, dir: &Path) -> Result<()>`.

Only `validate_dir` and the `local_files_only` path are unit tested — anything that downloads is left to the `#[ignore]` end-to-end test in Task 12.

- [ ] **Step 1: Write the failing tests**

Create `src/models/hub.rs`:

```rust
//! Fetch CTranslate2 Whisper models from Hugging Face, or use a local directory.

use crate::error::{Error, Result};
use crate::models::registry::{resolve, ModelRef};
use std::path::{Path, PathBuf};

/// Files CTranslate2 needs. `model.bin` and `config.json` are mandatory; the
/// rest are fetched when present, since repos differ in tokenizer layout.
const REQUIRED: &[&str] = &["model.bin", "config.json"];
const OPTIONAL: &[&str] = &[
    "tokenizer.json",
    "vocabulary.json",
    "vocabulary.txt",
    "preprocessor_config.json",
];

#[derive(Debug, Clone, Default)]
pub struct FetchOptions {
    pub download_root: Option<PathBuf>,
    pub local_files_only: bool,
}

pub fn ensure_model(_name: &str, _opts: &FetchOptions) -> Result<PathBuf> {
    todo!("step 3")
}

pub fn validate_dir(_name: &str, _dir: &Path) -> Result<()> {
    todo!("step 3")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("whisper_rs_t8_{tag}"));
        std::fs::remove_dir_all(&d).ok();
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn validate_accepts_a_directory_with_the_required_files() {
        let d = temp_dir("ok");
        std::fs::write(d.join("model.bin"), b"x").unwrap();
        std::fs::write(d.join("config.json"), b"{}").unwrap();

        validate_dir("tiny", &d).unwrap();
    }

    #[test]
    fn validate_rejects_a_directory_without_model_bin() {
        let d = temp_dir("no_bin");
        std::fs::write(d.join("config.json"), b"{}").unwrap();

        let err = validate_dir("tiny", &d).unwrap_err();

        match err {
            Error::ModelNotFound { name, path, message } => {
                assert_eq!(name, "tiny");
                assert_eq!(path, d);
                assert!(
                    message.contains("model.bin"),
                    "the message must name the missing file, got: {message}"
                );
            }
            other => panic!("expected ModelNotFound, got {other:?}"),
        }
    }

    #[test]
    fn a_valid_local_directory_is_returned_as_is() {
        let d = temp_dir("local");
        std::fs::write(d.join("model.bin"), b"x").unwrap();
        std::fs::write(d.join("config.json"), b"{}").unwrap();

        let got = ensure_model(d.to_str().unwrap(), &FetchOptions::default()).unwrap();

        assert_eq!(got, d);
    }

    #[test]
    fn local_files_only_refuses_to_download() {
        let opts = FetchOptions { local_files_only: true, ..Default::default() };

        let err = ensure_model("tiny", &opts).unwrap_err();

        match err {
            Error::ModelNotFound { message, .. } => assert!(
                message.contains("local_files_only"),
                "the message must explain why nothing was downloaded, got: {message}"
            ),
            other => panic!("expected ModelNotFound, got {other:?}"),
        }
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test models::hub`
Expected: FAIL on `todo!("step 3")`.

- [ ] **Step 3: Implement ensure_model and validate_dir**

Replace both stubs with:

```rust
/// Resolve `name` to a local directory containing a usable CTranslate2 model,
/// downloading from the hub when needed.
pub fn ensure_model(name: &str, opts: &FetchOptions) -> Result<PathBuf> {
    match resolve(name) {
        ModelRef::Local(dir) => {
            validate_dir(name, &dir)?;
            Ok(dir)
        }
        ModelRef::Hub { repo } => {
            if opts.local_files_only {
                return Err(Error::ModelNotFound {
                    name: name.to_string(),
                    path: opts.download_root.clone().unwrap_or_else(|| PathBuf::from("<hf cache>")),
                    message: "local_files_only is set, so nothing was downloaded".into(),
                });
            }
            let dir = download(name, &repo, opts)?;
            validate_dir(name, &dir)?;
            Ok(dir)
        }
    }
}

fn download(name: &str, repo: &str, opts: &FetchOptions) -> Result<PathBuf> {
    let mut builder = hf_hub::api::sync::ApiBuilder::new();
    if let Some(root) = &opts.download_root {
        builder = builder.with_cache_dir(root.clone());
    }
    let api = builder.build().map_err(|e| Error::Download {
        name: name.to_string(),
        message: e.to_string(),
    })?;
    let api_repo = api.model(repo.to_string());

    let mut dir: Option<PathBuf> = None;

    for file in REQUIRED {
        let path = api_repo.get(file).map_err(|e| Error::Download {
            name: name.to_string(),
            message: format!("{file}: {e}"),
        })?;
        if dir.is_none() {
            dir = path.parent().map(Path::to_path_buf);
        }
    }

    for file in OPTIONAL {
        // Absent optional files are normal; repos differ in tokenizer layout.
        if let Err(e) = api_repo.get(file) {
            tracing::debug!("optional file {file} not fetched for {repo}: {e}");
        }
    }

    dir.ok_or_else(|| Error::Download {
        name: name.to_string(),
        message: "downloaded files have no parent directory".into(),
    })
}

/// Fail early, and clearly, rather than letting CTranslate2 abort in C++.
pub fn validate_dir(name: &str, dir: &Path) -> Result<()> {
    if !dir.is_dir() {
        return Err(Error::ModelNotFound {
            name: name.to_string(),
            path: dir.to_path_buf(),
            message: "not a directory".into(),
        });
    }

    let missing: Vec<&str> = REQUIRED
        .iter()
        .copied()
        .filter(|f| !dir.join(f).is_file())
        .collect();

    if !missing.is_empty() {
        return Err(Error::ModelNotFound {
            name: name.to_string(),
            path: dir.to_path_buf(),
            message: format!("missing required file(s): {}", missing.join(", ")),
        });
    }

    Ok(())
}
```

- [ ] **Step 4: Export the module**

`src/models/mod.rs` becomes:

```rust
//! Model resolution, download, and cache handling.

pub mod hub;
pub mod registry;
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cargo test models::`
Expected: Task 7 and Task 8 tests PASS (9 total).

Run: `cargo clippy --all-targets -- -D warnings`
Expected: no warnings.

- [ ] **Step 6: Commit**

```bash
git add src/models
git commit -m "feat: download and validate CTranslate2 models from Hugging Face"
```

---

## Task 9: ASR trait, ct2rs backend, and language-token parsing

**Files:**
- Modify: `src/asr/mod.rs`
- Create: `src/asr/ct2.rs`

**Interfaces:**
- Consumes: `types::{Seg, Word}`, `error::{Error, Result}`.
- Produces: `asr::Asr` trait (`transcribe(&self, samples: &[f32], language: Option<&str>, word_timestamps: bool) -> Result<Vec<Seg>>`, `detect_language(&self, samples: &[f32]) -> Result<String>`), `asr::parse_language_token(raw: &str) -> Option<String>`, `asr::ct2::Ct2Asr::new(model_dir: &Path, cfg: Ct2Config) -> Result<Ct2Asr>`, and `asr::ct2::Ct2Config { device: String, device_index: i32, compute_type: String, cpu_threads: usize, num_workers: usize, beam_size: usize, patience: f32, length_penalty: f32, repetition_penalty: f32, no_repeat_ngram_size: usize, max_initial_timestamp: f32, suppress_blank: bool, temperature: f32 }`.

`parse_language_token` is pure and carries the language-detection workaround, so it is tested exhaustively without a model.

- [ ] **Step 1: Write the failing tests for parse_language_token**

Replace the contents of `src/asr/mod.rs` (keeping the Send+Sync test from Task 1) with:

```rust
//! Speech recognition backends.

pub mod ct2;

use crate::error::Result;
use crate::types::Seg;

/// A speech recognition backend. One call decodes one window.
pub trait Asr: Send + Sync {
    /// Transcribe exactly one window of samples.
    fn transcribe(
        &self,
        samples: &[f32],
        language: Option<&str>,
        word_timestamps: bool,
    ) -> Result<Vec<Seg>>;

    /// Best-effort language code for `samples`.
    fn detect_language(&self, samples: &[f32]) -> Result<String>;
}

/// Pull the language code out of raw Whisper output.
///
/// ct2rs exposes no language-detection API, so the code is recovered from the
/// `<|xx|>` token Whisper emits at the start of its output. Returns None when
/// no such token is present.
pub fn parse_language_token(raw: &str) -> Option<String> {
    let start = raw.find("<|")?;
    let rest = &raw[start + 2..];
    let end = rest.find("|>")?;
    let code = &rest[..end];

    let is_lang = (2..=3).contains(&code.len())
        && code.chars().all(|c| c.is_ascii_lowercase());

    if is_lang {
        Some(code.to_string())
    } else {
        // A timestamp or task token: keep looking after it.
        parse_language_token(&rest[end + 2..])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The Arc<Ct2Asr> design in the Python layer requires this.
    /// If this fails to compile, Ct2Asr must wrap Whisper in a Mutex.
    #[test]
    fn whisper_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ct2rs::Whisper>();
    }

    #[test]
    fn parses_a_leading_language_token() {
        assert_eq!(parse_language_token("<|pt|><|0.00|> olá"), Some("pt".into()));
    }

    #[test]
    fn parses_a_three_letter_code() {
        assert_eq!(parse_language_token("<|yue|><|0.00|> hi"), Some("yue".into()));
    }

    #[test]
    fn skips_a_leading_timestamp_token() {
        assert_eq!(
            parse_language_token("<|0.00|><|en|> hello"),
            Some("en".into()),
            "a timestamp before the language token must be skipped"
        );
    }

    #[test]
    fn skips_task_tokens() {
        assert_eq!(
            parse_language_token("<|startoftranscript|><|de|><|transcribe|> hallo"),
            Some("de".into())
        );
    }

    #[test]
    fn plain_text_has_no_language_token() {
        assert_eq!(parse_language_token("just some text"), None);
    }

    #[test]
    fn empty_input_has_no_language_token() {
        assert_eq!(parse_language_token(""), None);
    }

    #[test]
    fn an_unterminated_token_is_not_a_language() {
        assert_eq!(parse_language_token("<|pt"), None);
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test asr::`
Expected: FAIL to compile — `src/asr/ct2.rs` does not exist.

- [ ] **Step 3: Implement the ct2rs backend**

Create `src/asr/ct2.rs`. Check the exact `ct2rs::Config`, `ct2rs::WhisperOptions`, `ct2rs::Device` and `ct2rs::ComputeType` field names against `cargo doc --open -p ct2rs` before writing; the shape below is what the code needs, and only the field names may differ.

```rust
//! ct2rs (CTranslate2) Whisper backend.

use super::Asr;
use crate::error::{Error, Result};
use crate::types::{Seg, Word};
use std::path::Path;

#[derive(Debug, Clone)]
pub struct Ct2Config {
    pub device: String,
    pub device_index: i32,
    pub compute_type: String,
    pub cpu_threads: usize,
    pub num_workers: usize,
    pub beam_size: usize,
    pub patience: f32,
    pub length_penalty: f32,
    pub repetition_penalty: f32,
    pub no_repeat_ngram_size: usize,
    pub max_initial_timestamp: f32,
    pub suppress_blank: bool,
    pub temperature: f32,
}

impl Default for Ct2Config {
    fn default() -> Self {
        Self {
            device: "cpu".into(),
            device_index: 0,
            compute_type: "default".into(),
            cpu_threads: 0,
            num_workers: 1,
            beam_size: 5,
            patience: 1.0,
            length_penalty: 1.0,
            repetition_penalty: 1.0,
            no_repeat_ngram_size: 0,
            max_initial_timestamp: 1.0,
            suppress_blank: true,
            temperature: 0.0,
        }
    }
}

pub struct Ct2Asr {
    inner: ct2rs::Whisper,
    options: ct2rs::WhisperOptions,
}

impl Ct2Asr {
    pub fn new(model_dir: &Path, cfg: Ct2Config) -> Result<Self> {
        let mut config = ct2rs::Config::default();
        config.device = parse_device(&cfg.device)?;
        config.device_indices = vec![cfg.device_index];
        config.compute_type = parse_compute_type(&cfg.compute_type)?;
        config.num_threads_per_replica = cfg.cpu_threads;
        config.inter_threads = cfg.num_workers;

        let inner = ct2rs::Whisper::new(model_dir, config)
            .map_err(|e| Error::Ct2(e.to_string()))?;

        let mut options = ct2rs::WhisperOptions::default();
        options.beam_size = cfg.beam_size;
        options.patience = cfg.patience;
        options.length_penalty = cfg.length_penalty;
        options.repetition_penalty = cfg.repetition_penalty;
        options.no_repeat_ngram_size = cfg.no_repeat_ngram_size;
        options.max_initial_timestamp = cfg.max_initial_timestamp;
        options.suppress_blank = cfg.suppress_blank;
        options.sampling_temperature = cfg.temperature;

        Ok(Self { inner, options })
    }

    /// Window size the model expects, in samples.
    pub fn n_samples(&self) -> usize {
        self.inner.n_samples()
    }
}

fn parse_device(name: &str) -> Result<ct2rs::Device> {
    match name {
        "cpu" => Ok(ct2rs::Device::CPU),
        "cuda" => Ok(ct2rs::Device::CUDA),
        other => Err(Error::Ct2(format!(
            "unknown device {other:?}, expected \"cpu\" or \"cuda\""
        ))),
    }
}

fn parse_compute_type(name: &str) -> Result<ct2rs::ComputeType> {
    match name {
        "default" => Ok(ct2rs::ComputeType::DEFAULT),
        "auto" => Ok(ct2rs::ComputeType::AUTO),
        "float32" => Ok(ct2rs::ComputeType::FLOAT32),
        "float16" => Ok(ct2rs::ComputeType::FLOAT16),
        "bfloat16" => Ok(ct2rs::ComputeType::BFLOAT16),
        "int8" => Ok(ct2rs::ComputeType::INT8),
        "int8_float16" => Ok(ct2rs::ComputeType::INT8_FLOAT16),
        "int8_float32" => Ok(ct2rs::ComputeType::INT8_FLOAT32),
        "int8_bfloat16" => Ok(ct2rs::ComputeType::INT8_BFLOAT16),
        other => Err(Error::Ct2(format!("unknown compute_type {other:?}"))),
    }
}

impl Asr for Ct2Asr {
    fn transcribe(
        &self,
        samples: &[f32],
        language: Option<&str>,
        word_timestamps: bool,
    ) -> Result<Vec<Seg>> {
        let segments = self
            .inner
            .generate_segments(samples, language, &self.options)
            .map_err(|e| Error::Ct2(e.to_string()))?;

        Ok(segments
            .into_iter()
            .map(|s| Seg {
                // The real id is assigned by stitch().
                id: 0,
                start: s.start,
                end: s.end,
                text: s.text,
                words: if word_timestamps {
                    Some(
                        s.words
                            .into_iter()
                            .map(|w| Word {
                                start: w.start,
                                end: w.end,
                                text: w.text,
                                probability: w.probability,
                            })
                            .collect(),
                    )
                } else {
                    None
                },
            })
            .collect())
    }

    fn detect_language(&self, samples: &[f32]) -> Result<String> {
        // No detection API exists, so the code is read off the raw output tokens.
        let raw = self
            .inner
            .generate(samples, None, true, &self.options)
            .map_err(|e| Error::Ct2(e.to_string()))?;

        Ok(raw
            .iter()
            .find_map(|line| super::parse_language_token(line))
            .unwrap_or_else(|| "unknown".to_string()))
    }
}
```

If `ct2rs::Segment` or `ct2rs::Word` field names differ (for example `words: Option<Vec<Word>>` instead of `Vec<Word>`), adjust the mapping only. `Asr` is the contract the rest of the crate depends on.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test asr::`
Expected: 8 tests PASS (7 parser tests plus the Send+Sync assertion).

Run: `cargo clippy --all-targets -- -D warnings`
Expected: no warnings.

- [ ] **Step 5: Commit**

If the Task 1 Send+Sync check failed and `Ct2Asr` now holds a `Mutex`, say so in the commit body.

```bash
git add src/asr
git commit -m "feat: add Asr trait, ct2rs backend, and language token parsing"
```

---

## Task 10: The eager pipeline

**Files:**
- Create: `src/pipeline.rs`
- Modify: `src/lib.rs` (add `mod pipeline;`)

**Interfaces:**
- Consumes: `audio::load_16k_mono`, `vad::{VadParams, SileroBackend, detect}`, `chunk::{plan_windows, build_window}`, `types::{Info, Window, SAMPLE_RATE}`, `error::Result`.
- Produces: `pipeline::Prepared { windows: Vec<Window>, info: Info }` and `pipeline::prepare(path: &Path, vad_filter: bool, params: &VadParams) -> Result<Prepared>`. `info.language` is left empty here; the Python layer fills it after detection.

- [ ] **Step 1: Write the failing tests**

Create `src/pipeline.rs`:

```rust
//! The eager half of transcription: decode, VAD, window.

use crate::error::Result;
use crate::types::{Info, Window, SAMPLE_RATE};
use crate::vad::{Vad, VadParams};
use std::path::Path;

pub struct Prepared {
    pub windows: Vec<Window>,
    /// `language` is empty here; the caller fills it after detection.
    pub info: Info,
}

pub fn prepare(_path: &Path, _vad_filter: bool, _params: &VadParams) -> Result<Prepared> {
    todo!("step 3")
}

/// Split out so windowing is testable without touching a VAD model.
pub(crate) fn windows_from_samples(
    samples: &[f32],
    regions: &[crate::types::SpeechRegion],
    probs: &[f32],
    frame_samples: usize,
) -> (Vec<Window>, f32) {
    let ranges = crate::chunk::plan_windows(regions, probs, frame_samples);
    let windows: Vec<Window> = ranges
        .iter()
        .map(|&r| crate::chunk::build_window(samples, r))
        .collect();
    let speech_samples: usize = regions.iter().map(|r| r.len()).sum();
    (windows, speech_samples as f32 / SAMPLE_RATE as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{SpeechRegion, WINDOW_SAMPLES};

    #[test]
    fn windows_cover_the_regions_and_report_speech_duration() {
        let samples = vec![0.3f32; 40 * SAMPLE_RATE];
        let regions = vec![
            SpeechRegion { start: 0, end: 10 * SAMPLE_RATE },
            SpeechRegion { start: 20 * SAMPLE_RATE, end: 35 * SAMPLE_RATE },
        ];

        let (windows, after_vad) = windows_from_samples(&samples, &regions, &[], 512);

        assert!(!windows.is_empty());
        for w in &windows {
            assert_eq!(w.samples.len(), WINDOW_SAMPLES, "every window is padded to 30 s");
            assert!(w.real_len <= WINDOW_SAMPLES);
        }
        assert!(
            (after_vad - 25.0).abs() < 0.01,
            "10 s + 15 s of speech expected, got {after_vad}"
        );
    }

    #[test]
    fn speech_duration_equals_the_sum_of_window_real_lengths_when_regions_do_not_touch_padding() {
        // One contiguous region: real_len across windows must add up to it.
        let samples = vec![0.3f32; 70 * SAMPLE_RATE];
        let regions = vec![SpeechRegion { start: 0, end: 70 * SAMPLE_RATE }];

        let (windows, after_vad) = windows_from_samples(&samples, &regions, &[], 512);
        let total_real: usize = windows.iter().map(|w| w.real_len).sum();

        assert!(
            (total_real as f32 / SAMPLE_RATE as f32 - after_vad).abs() < 0.01,
            "window real lengths ({}) must sum to duration_after_vad ({after_vad})",
            total_real as f32 / SAMPLE_RATE as f32
        );
    }

    #[test]
    fn no_speech_sample_lands_in_two_windows() {
        let samples = vec![0.3f32; 95 * SAMPLE_RATE];
        let regions = vec![SpeechRegion { start: 0, end: 95 * SAMPLE_RATE }];

        let (windows, _) = windows_from_samples(&samples, &regions, &[], 512);

        for pair in windows.windows(2) {
            let a_end = pair[0].offset + pair[0].real_len;
            assert!(
                a_end <= pair[1].offset,
                "windows overlap: {}..{} then {}",
                pair[0].offset,
                a_end,
                pair[1].offset
            );
        }
    }

    #[test]
    fn silent_audio_yields_no_windows() {
        let samples = vec![0.0f32; 10 * SAMPLE_RATE];
        let (windows, after_vad) = windows_from_samples(&samples, &[], &[], 512);
        assert!(windows.is_empty());
        assert_eq!(after_vad, 0.0);
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test pipeline::`
Expected: FAIL to compile or FAIL on `todo!` depending on ordering; the four tests above must not pass yet.

- [ ] **Step 3: Implement prepare**

Replace the stub with:

```rust
/// Decode `path`, run VAD, and plan decoder windows.
///
/// With `vad_filter` off, the audio is windowed as one long region — the 30 s
/// ceiling still applies, so windows are cut every 30 s.
pub fn prepare(path: &Path, vad_filter: bool, params: &VadParams) -> Result<Prepared> {
    let samples = crate::audio::load_16k_mono(path)?;
    let duration = samples.len() as f32 / SAMPLE_RATE as f32;

    let (windows, after_vad) = if vad_filter {
        let mut vad = crate::vad::SileroBackend::new()?;
        let (probs, regions) = crate::vad::detect(&mut vad, &samples, params)?;
        windows_from_samples(&samples, &regions, &probs, vad.frame_samples())
    } else {
        let all = [crate::types::SpeechRegion { start: 0, end: samples.len() }];
        windows_from_samples(&samples, &all, &[], 0)
    };

    Ok(Prepared {
        windows,
        info: Info {
            language: String::new(),
            language_probability: None,
            duration,
            duration_after_vad: after_vad,
        },
    })
}
```

- [ ] **Step 4: Add the module to lib.rs**

Add `mod pipeline;` to `src/lib.rs`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cargo test`
Expected: every test from Tasks 1-10 PASS.

Run: `cargo clippy --all-targets -- -D warnings`
Expected: no warnings.

- [ ] **Step 6: Commit**

```bash
git add src/pipeline.rs src/lib.rs
git commit -m "feat: add eager decode/VAD/windowing pipeline"
```

---

## Task 11: Python bindings

**Files:**
- Create: `src/python/mod.rs`
- Create: `src/python/segment.rs`
- Create: `src/python/model.rs`
- Create: `src/python/iter.rs`
- Modify: `src/lib.rs`

**Interfaces:**
- Consumes: everything from Tasks 1-10.
- Produces: Python classes `WhisperModel`, `SegmentIterator`, `Segment`, `Word`, `TranscriptionInfo`.

- [ ] **Step 1: Write the error mapping and its test**

Create `src/python/mod.rs`:

```rust
//! The only module allowed to use pyo3 types.

pub mod iter;
pub mod model;
pub mod segment;

use crate::error::Error;
use pyo3::exceptions::{PyOSError, PyRuntimeError, PyValueError};
use pyo3::PyErr;

/// Map a crate error onto the Python exception a caller would expect.
pub fn to_pyerr(err: Error) -> PyErr {
    let message = err.to_string();
    match err {
        Error::AudioRead { .. }
        | Error::AudioFormat { .. }
        | Error::AudioEmpty { .. } => PyValueError::new_err(message),
        Error::ModelNotFound { .. } | Error::Download { .. } => PyOSError::new_err(message),
        Error::Resample(_) | Error::Vad(_) | Error::Ct2(_) => PyRuntimeError::new_err(message),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn audio_errors_become_value_errors() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let err = to_pyerr(Error::AudioEmpty { path: PathBuf::from("/tmp/x.wav") });
            assert!(err.is_instance_of::<PyValueError>(py));
        });
    }

    #[test]
    fn model_errors_become_os_errors() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let err = to_pyerr(Error::Download {
                name: "tiny".into(),
                message: "offline".into(),
            });
            assert!(err.is_instance_of::<PyOSError>(py));
        });
    }

    #[test]
    fn backend_errors_become_runtime_errors() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let err = to_pyerr(Error::Ct2("boom".into()));
            assert!(err.is_instance_of::<PyRuntimeError>(py));
        });
    }

    #[test]
    fn messages_keep_the_path_and_model_name() {
        let msg = to_pyerr(Error::ModelNotFound {
            name: "large-v3".into(),
            path: PathBuf::from("/cache/models"),
            message: "missing required file(s): model.bin".into(),
        })
        .to_string();

        assert!(msg.contains("large-v3"), "got {msg}");
        assert!(msg.contains("/cache/models"), "got {msg}");
        assert!(msg.contains("model.bin"), "got {msg}");
    }
}

use pyo3::prelude::*;
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test python::`
Expected: FAIL to compile — the `iter`, `model`, and `segment` submodules do not exist.

Note: these tests need a Python interpreter linked in. If the `extension-module` feature blocks linking during `cargo test`, remove it from the default features and gate it behind an `extension-module` feature that maturin enables:

```toml
pyo3 = { version = "0.28.3", features = ["abi3-py38"] }

[features]
extension-module = ["pyo3/extension-module"]
```

and set `features = ["extension-module"]` under `[tool.maturin]` in `pyproject.toml`. Do this now if linking fails, and note it in the commit.

- [ ] **Step 3: Implement the data classes**

Create `src/python/segment.rs`:

```rust
use pyo3::prelude::*;

#[pyclass(frozen, get_all)]
#[derive(Clone)]
pub struct Word {
    pub start: f32,
    pub end: f32,
    pub word: String,
    pub probability: f32,
}

#[pymethods]
impl Word {
    fn __repr__(&self) -> String {
        format!(
            "Word(start={:.2}, end={:.2}, word={:?}, probability={:.2})",
            self.start, self.end, self.word, self.probability
        )
    }
}

#[pyclass(frozen, get_all)]
#[derive(Clone)]
pub struct Segment {
    pub id: u32,
    pub start: f32,
    pub end: f32,
    pub text: String,
    pub words: Option<Vec<Word>>,
}

#[pymethods]
impl Segment {
    fn __repr__(&self) -> String {
        format!(
            "Segment(id={}, start={:.2}, end={:.2}, text={:?})",
            self.id, self.start, self.end, self.text
        )
    }
}

#[pyclass(frozen, get_all)]
#[derive(Clone)]
pub struct TranscriptionInfo {
    pub language: String,
    /// Always None in v1: not recoverable through the ct2rs API.
    pub language_probability: Option<f32>,
    pub duration: f32,
    pub duration_after_vad: f32,
}

#[pymethods]
impl TranscriptionInfo {
    fn __repr__(&self) -> String {
        format!(
            "TranscriptionInfo(language={:?}, duration={:.2}, duration_after_vad={:.2})",
            self.language, self.duration, self.duration_after_vad
        )
    }
}

impl From<crate::types::Seg> for Segment {
    fn from(s: crate::types::Seg) -> Self {
        Self {
            id: s.id,
            start: s.start,
            end: s.end,
            text: s.text,
            words: s.words.map(|ws| {
                ws.into_iter()
                    .map(|w| Word {
                        start: w.start,
                        end: w.end,
                        word: w.text,
                        probability: w.probability,
                    })
                    .collect()
            }),
        }
    }
}

impl From<crate::types::Info> for TranscriptionInfo {
    fn from(i: crate::types::Info) -> Self {
        Self {
            language: i.language,
            language_probability: i.language_probability,
            duration: i.duration,
            duration_after_vad: i.duration_after_vad,
        }
    }
}
```

- [ ] **Step 4: Implement the lazy iterator**

Create `src/python/iter.rs`:

```rust
use crate::asr::{ct2::Ct2Asr, Asr};
use crate::python::segment::Segment;
use crate::python::to_pyerr;
use crate::types::{Seg, Window};
use pyo3::prelude::*;
use std::collections::VecDeque;
use std::sync::Arc;

/// Lazily decodes one window per __next__ call.
#[pyclass]
pub struct SegmentIterator {
    asr: Arc<Ct2Asr>,
    windows: VecDeque<Window>,
    pending: VecDeque<Seg>,
    next_id: u32,
    language: String,
    word_timestamps: bool,
}

impl SegmentIterator {
    pub fn new(
        asr: Arc<Ct2Asr>,
        windows: Vec<Window>,
        language: String,
        word_timestamps: bool,
    ) -> Self {
        Self {
            asr,
            windows: windows.into(),
            pending: VecDeque::new(),
            next_id: 0,
            language,
            word_timestamps,
        }
    }
}

#[pymethods]
impl SegmentIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>, py: Python<'_>) -> PyResult<Option<Segment>> {
        loop {
            if let Some(seg) = slf.pending.pop_front() {
                return Ok(Some(seg.into()));
            }

            let Some(window) = slf.windows.pop_front() else {
                return Ok(None);
            };

            let asr = Arc::clone(&slf.asr);
            let language = slf.language.clone();
            let words = slf.word_timestamps;

            // Release the GIL for the whole decode: it is the slow part.
            let raw = py
                .allow_threads(|| {
                    asr.transcribe(&window.samples, Some(&language), words)
                })
                .map_err(to_pyerr)?;

            let mut next_id = slf.next_id;
            let stitched = crate::stitch::stitch(&window, raw, &mut next_id);
            slf.next_id = next_id;
            slf.pending.extend(stitched);
            // Loop again: an empty window must not end the iteration.
        }
    }
}
```

- [ ] **Step 5: Implement WhisperModel**

Create `src/python/model.rs`:

```rust
use crate::asr::ct2::{Ct2Asr, Ct2Config};
use crate::asr::Asr;
use crate::models::hub::{ensure_model, FetchOptions};
use crate::python::iter::SegmentIterator;
use crate::python::segment::TranscriptionInfo;
use crate::python::to_pyerr;
use crate::vad::VadParams;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[pyclass]
pub struct WhisperModel {
    asr: Arc<Ct2Asr>,
    config: Ct2Config,
    model_dir: PathBuf,
}

#[pymethods]
impl WhisperModel {
    #[new]
    #[pyo3(signature = (
        model,
        *,
        device = "cpu",
        device_index = 0,
        compute_type = "default",
        cpu_threads = 0,
        num_workers = 1,
        download_root = None,
        local_files_only = false,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        model: &str,
        device: &str,
        device_index: i32,
        compute_type: &str,
        cpu_threads: usize,
        num_workers: usize,
        download_root: Option<PathBuf>,
        local_files_only: bool,
    ) -> PyResult<Self> {
        let opts = FetchOptions {
            download_root,
            local_files_only,
        };

        let config = Ct2Config {
            device: device.to_string(),
            device_index,
            compute_type: compute_type.to_string(),
            cpu_threads,
            num_workers,
            ..Ct2Config::default()
        };

        let name = model.to_string();
        let cfg = config.clone();
        // Downloading and loading the model both take seconds to minutes.
        let (asr, model_dir) = py
            .allow_threads(move || -> crate::error::Result<(Ct2Asr, PathBuf)> {
                let dir = ensure_model(&name, &opts)?;
                let asr = Ct2Asr::new(&dir, cfg)?;
                Ok((asr, dir))
            })
            .map_err(to_pyerr)?;

        Ok(Self {
            asr: Arc::new(asr),
            config,
            model_dir,
        })
    }

    /// Directory the model was loaded from. Useful when debugging cache issues.
    #[getter]
    fn model_path(&self) -> String {
        self.model_dir.display().to_string()
    }

    #[pyo3(signature = (
        audio,
        *,
        language = None,
        task = "transcribe",
        beam_size = 5,
        patience = 1.0,
        length_penalty = 1.0,
        temperature = 0.0,
        repetition_penalty = 1.0,
        no_repeat_ngram_size = 0,
        max_initial_timestamp = 1.0,
        suppress_blank = true,
        word_timestamps = false,
        vad_filter = true,
        vad_parameters = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn transcribe(
        &self,
        py: Python<'_>,
        audio: PathBuf,
        language: Option<String>,
        task: &str,
        beam_size: usize,
        patience: f32,
        length_penalty: f32,
        temperature: f32,
        repetition_penalty: f32,
        no_repeat_ngram_size: usize,
        max_initial_timestamp: f32,
        suppress_blank: bool,
        word_timestamps: bool,
        vad_filter: bool,
        vad_parameters: Option<Bound<'_, PyDict>>,
    ) -> PyResult<(SegmentIterator, TranscriptionInfo)> {
        if task != "transcribe" {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "task {task:?} is not supported in v1, only \"transcribe\""
            )));
        }

        let params = vad_params_from_dict(vad_parameters.as_ref())?;

        // These four are per-call, so the backend is rebuilt when they differ
        // from the loaded configuration.
        let needs_reload = beam_size != self.config.beam_size
            || (patience - self.config.patience).abs() > f32::EPSILON
            || (length_penalty - self.config.length_penalty).abs() > f32::EPSILON
            || (temperature - self.config.temperature).abs() > f32::EPSILON
            || (repetition_penalty - self.config.repetition_penalty).abs() > f32::EPSILON
            || no_repeat_ngram_size != self.config.no_repeat_ngram_size
            || (max_initial_timestamp - self.config.max_initial_timestamp).abs() > f32::EPSILON
            || suppress_blank != self.config.suppress_blank;

        let asr = if needs_reload {
            let cfg = Ct2Config {
                beam_size,
                patience,
                length_penalty,
                temperature,
                repetition_penalty,
                no_repeat_ngram_size,
                max_initial_timestamp,
                suppress_blank,
                ..self.config.clone()
            };
            let dir = self.model_dir.clone();
            Arc::new(
                py.allow_threads(move || Ct2Asr::new(&dir, cfg))
                    .map_err(to_pyerr)?,
            )
        } else {
            Arc::clone(&self.asr)
        };

        let path: PathBuf = audio;
        let asr_for_prep = Arc::clone(&asr);
        let (windows, mut info) = py
            .allow_threads(move || -> crate::error::Result<_> {
                let prepared = crate::pipeline::prepare(Path::new(&path), vad_filter, &params)?;
                let mut info = prepared.info;

                info.language = match language {
                    Some(code) => code,
                    None => match prepared.windows.first() {
                        // One extra 30 s decode, then the code is reused for
                        // every window so the language cannot flip mid-file.
                        Some(w) => asr_for_prep.detect_language(&w.samples)?,
                        None => "unknown".to_string(),
                    },
                };

                Ok((prepared.windows, info))
            })
            .map_err(to_pyerr)?;

        // Never fabricated: the API cannot produce it.
        info.language_probability = None;

        let language = info.language.clone();
        Ok((
            SegmentIterator::new(asr, windows, language, word_timestamps),
            info.into(),
        ))
    }
}

fn vad_params_from_dict(dict: Option<&Bound<'_, PyDict>>) -> PyResult<VadParams> {
    let mut params = VadParams::default();
    let Some(dict) = dict else { return Ok(params) };

    for (key, value) in dict.iter() {
        let key: String = key.extract()?;
        match key.as_str() {
            "threshold" => params.threshold = value.extract()?,
            "neg_threshold" => params.neg_threshold = value.extract()?,
            "min_speech_duration_ms" => params.min_speech_ms = value.extract()?,
            "min_silence_duration_ms" => params.min_silence_ms = value.extract()?,
            "speech_pad_ms" => params.speech_pad_ms = value.extract()?,
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "unknown vad_parameters key {other:?}; supported: threshold, \
                     neg_threshold, min_speech_duration_ms, min_silence_duration_ms, speech_pad_ms"
                )))
            }
        }
    }

    Ok(params)
}
```

- [ ] **Step 6: Register the module**

`src/lib.rs` becomes:

```rust
mod asr;
mod audio;
mod chunk;
mod error;
mod models;
mod pipeline;
mod python;
mod stitch;
mod types;
mod vad;

use pyo3::prelude::*;

#[pymodule]
fn whisper_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<python::model::WhisperModel>()?;
    m.add_class::<python::iter::SegmentIterator>()?;
    m.add_class::<python::segment::Segment>()?;
    m.add_class::<python::segment::Word>()?;
    m.add_class::<python::segment::TranscriptionInfo>()?;
    Ok(())
}
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `cargo test`
Expected: every test PASS, including the four error-mapping tests.

Run: `cargo clippy --all-targets -- -D warnings`
Expected: no warnings.

- [ ] **Step 8: Commit**

```bash
git add src/python src/lib.rs Cargo.toml pyproject.toml
git commit -m "feat: add Python bindings with lazy segment iterator"
```

---

## Task 12: Python package, converter wrapper, and CI

**Files:**
- Create: `python/whisper_rs/__init__.py`
- Create: `python/whisper_rs/convert.py`
- Create: `tests/python/test_api.py`
- Create: `tests/e2e.rs`
- Modify: `pyproject.toml`
- Modify: `.github/workflows/CI.yml`

**Interfaces:**
- Consumes: the classes registered in Task 11.
- Produces: the installable `whisper_rs` package and `whisper_rs.convert.convert_model(model, output_dir, *, quantization=None, force=False) -> str`.

- [ ] **Step 1: Write the Python package files**

Create `python/whisper_rs/__init__.py`:

```python
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
```

Create `python/whisper_rs/convert.py`:

```python
"""Wrapper around the official ct2-transformers-converter.

Model conversion is weight I/O and tensor renaming, runs once per model, and the
official converter tracks CTranslate2 format changes -- so it stays in Python
rather than being reimplemented in Rust.

Requires the optional dependencies: pip install "whisper-rs[convert]"
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

__all__ = ["convert_model"]

_MISSING = (
    "ct2-transformers-converter was not found. Install the conversion extras:\n"
    '    pip install "whisper-rs[convert]"\n'
    "or directly:\n"
    "    pip install ctranslate2 transformers"
)


def convert_model(
    model: str,
    output_dir: str | Path,
    *,
    quantization: str | None = None,
    force: bool = False,
) -> str:
    """Convert a Hugging Face Whisper checkpoint to CTranslate2 format.

    Args:
        model: HF model id or local path, e.g. "openai/whisper-large-v3".
        output_dir: Directory to write the converted model into.
        quantization: Optional target type, e.g. "float16" or "int8".
        force: Overwrite output_dir if it already exists.

    Returns:
        The output directory as a string, ready to pass to WhisperModel.

    Raises:
        RuntimeError: The converter is not installed, or it failed.
    """
    if shutil.which("ct2-transformers-converter") is None:
        raise RuntimeError(_MISSING)

    output = Path(output_dir)
    if output.exists() and not force:
        raise RuntimeError(f"{output} already exists; pass force=True to overwrite")

    cmd = [
        "ct2-transformers-converter",
        "--model",
        str(model),
        "--output_dir",
        str(output),
    ]
    if quantization:
        cmd += ["--quantization", quantization]
    if force:
        cmd.append("--force")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ct2-transformers-converter failed (exit {result.returncode}):\n"
            f"{result.stderr.strip()}"
        )

    return str(output)


def _main(argv: list[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m whisper_rs.convert",
        description="Convert a Whisper checkpoint to CTranslate2 format.",
    )
    parser.add_argument("model", help='HF model id or local path, e.g. "openai/whisper-large-v3"')
    parser.add_argument("output_dir", help="Directory to write the converted model into")
    parser.add_argument("--quantization", default=None, help='e.g. "float16" or "int8"')
    parser.add_argument("--force", action="store_true", help="Overwrite output_dir")
    args = parser.parse_args(argv)

    try:
        path = convert_model(
            args.model,
            args.output_dir,
            quantization=args.quantization,
            force=args.force,
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
```

- [ ] **Step 2: Update pyproject.toml**

```toml
[build-system]
requires = ["maturin>=1.13,<2.0"]
build-backend = "maturin"

[project]
name = "whisper-rs"
requires-python = ">=3.8"
description = "Whisper transcription with a Rust core: CTranslate2 inference, VAD-driven windowing"
classifiers = [
    "Programming Language :: Rust",
    "Programming Language :: Python :: Implementation :: CPython",
    "Programming Language :: Python :: Implementation :: PyPy",
]
dynamic = ["version"]

[project.optional-dependencies]
convert = ["ctranslate2", "transformers"]

[tool.maturin]
python-source = "python"
module-name = "whisper_rs.whisper_rs"
features = ["pyo3/extension-module"]
```

If Task 11 Step 2 moved `extension-module` behind a crate feature, use `features = ["extension-module"]` here instead.

- [ ] **Step 3: Write the Python API tests**

Create `tests/python/test_api.py`:

```python
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
```

Add the marker registration to `pyproject.toml`:

```toml
[tool.pytest.ini_options]
markers = ["model: needs the tiny Whisper model (downloads on first run)"]
```

- [ ] **Step 4: Write the Rust end-to-end test**

Create `tests/e2e.rs`:

```rust
//! End-to-end test with a real model. Ignored by default: it downloads.
//! Run with: cargo test --test e2e -- --ignored --nocapture

use std::path::PathBuf;

/// Write a 16 kHz mono WAV: one second of silence, a tone, one more second of silence.
fn write_test_wav(path: &PathBuf) {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16_000,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut w = hound::WavWriter::create(path, spec).unwrap();
    for i in 0..(16_000 * 4) {
        let t = i as f32 / 16_000.0;
        let amp = if (1.0..3.0).contains(&t) { 0.4 } else { 0.0 };
        w.write_sample(amp * (t * 440.0 * std::f32::consts::TAU).sin())
            .unwrap();
    }
    w.finalize().unwrap();
}

#[test]
#[ignore = "downloads the tiny model"]
fn tiny_model_transcribes_a_wav_end_to_end() {
    let dir = std::env::temp_dir().join("whisper_rs_e2e");
    std::fs::create_dir_all(&dir).unwrap();
    let audio = dir.join("tone.wav");
    write_test_wav(&audio);

    // This mirrors what the Python layer does, through the public Python class.
    // Rust-side integration is exercised via the pyo3 module, so this test
    // asserts the pipeline runs and produces a coherent timeline rather than
    // asserting specific words for a tone.
    pyo3::prepare_freethreaded_python();
    pyo3::Python::with_gil(|py| {
        let module = py.import("whisper_rs").expect("build the wheel first: maturin develop");
        let model = module
            .getattr("WhisperModel")
            .unwrap()
            .call1(("tiny",))
            .expect("loading the tiny model failed");

        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("vad_filter", true).unwrap();
        let result = model
            .call_method("transcribe", (audio.to_str().unwrap(),), Some(&kwargs))
            .expect("transcribe failed");

        let info = result.get_item(1).unwrap();
        let duration: f32 = info.getattr("duration").unwrap().extract().unwrap();
        assert!((duration - 4.0).abs() < 0.2, "duration was {duration}");

        let segments = result.get_item(0).unwrap();
        let mut last_end = 0.0f32;
        for seg in segments.try_iter().unwrap() {
            let seg = seg.unwrap();
            let start: f32 = seg.getattr("start").unwrap().extract().unwrap();
            let end: f32 = seg.getattr("end").unwrap().extract().unwrap();
            assert!(start >= last_end - 0.01, "segments must not go backwards");
            assert!(end <= duration + 0.5, "segment end {end} is past the audio");
            last_end = end;
        }
    });
}
```

- [ ] **Step 5: Write the CI workflow**

Replace `.github/workflows/CI.yml`:

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
  workflow_dispatch:

jobs:
  rust:
    name: cargo test and clippy
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-latest]
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with:
          components: clippy
      - uses: Swatinem/rust-cache@v2
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install CMake
        uses: lukka/get-cmake@latest
      # The ignored tests are excluded: they download models.
      - run: cargo test --all-targets
      - run: cargo clippy --all-targets -- -D warnings

  wheel:
    name: build wheel and smoke test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - uses: Swatinem/rust-cache@v2
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install CMake
        uses: lukka/get-cmake@latest
      - run: pip install maturin pytest
      - run: maturin build --release --out dist
      - run: pip install --find-links dist whisper-rs
      # Only the tests that need no model.
      - run: pytest tests/python -v -m "not model"

  model-tests:
    name: model-backed tests
    runs-on: ubuntu-latest
    # Manual only: these download the tiny model.
    if: github.event_name == 'workflow_dispatch'
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - uses: Swatinem/rust-cache@v2
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install CMake
        uses: lukka/get-cmake@latest
      - uses: actions/cache@v4
        with:
          path: ~/.cache/huggingface
          key: hf-tiny-${{ runner.os }}
      - run: pip install maturin pytest
      - run: maturin develop --release
      - run: pytest tests/python -v -m model
      - run: cargo test --test e2e -- --ignored
```

- [ ] **Step 6: Build the wheel and run the model-free Python tests**

Run: `pip install maturin pytest && maturin develop`
Expected: builds and installs into the active virtualenv.

Run: `pytest tests/python -v -m "not model"`
Expected: 2 tests PASS (`test_module_exports_the_public_classes`, `test_missing_model_raises_oserror`).

- [ ] **Step 7: Run the model-backed tests once locally**

Run: `pytest tests/python -v -m model`
Expected: PASS. First run downloads the tiny model (about 75 MB).

This is the definition of done for v1. If `test_segments_are_produced_lazily` fails, decoding is happening in `transcribe` instead of in `__next__` — check that `prepare` is the only work done eagerly.

- [ ] **Step 8: Run the full Rust suite plus clippy**

Run: `cargo test --all-targets`
Expected: PASS (the e2e test is ignored).

Run: `cargo clippy --all-targets -- -D warnings`
Expected: no warnings.

- [ ] **Step 9: Commit**

```bash
git add python tests pyproject.toml .github/workflows/CI.yml
git commit -m "feat: add Python package, converter wrapper, and CI"
```

---

## Definition of Done

- `pip install` works from a built wheel.
- `WhisperModel("tiny")` downloads and loads a model.
- A multi-minute mp3 and wav transcribe with plausible word timestamps.
- VAD cuts silence: `info.duration_after_vad < info.duration` on audio with silence in it.
- `cargo test` (no model) and `pytest -m "not model"` are green.

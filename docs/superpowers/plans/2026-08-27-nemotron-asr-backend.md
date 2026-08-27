# Nemotron ASR Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a second ASR backend, NVIDIA Nemotron (via `parakeet-rs`'s `Nemotron` struct), exposed to Python as `NemotronModel`, reusing the existing audio/VAD/windowing/diarization pipeline.

**Architecture:** `NemotronAsr` implements the existing `asr::Asr` trait (same interface `Ct2Asr` already implements), wrapped in a `Mutex` because `parakeet_rs::Nemotron`'s methods take `&mut self` with no internal lock. `SegmentIterator` is generalized from `Arc<Ct2Asr>` to `Arc<dyn Asr>` so both backends can drive it. The onnxruntime dylib-locate/init code (`dylib.rs`), currently private to the `diarization` feature, moves to a shared top-level module so `nemotron` can reuse it without duplicating the SIGBUS-avoidance logic.

**Tech Stack:** Rust 2024, `parakeet-rs` 0.3 (`ort` 2.0.0-rc.13, `load-dynamic`), pyo3 0.28, existing `hf-hub`/`ct2rs` stack.

**Spec:** `docs/superpowers/specs/2026-08-27-whisper-rs-v3-nemotron-design.md`

## Global Constraints

- `parakeet-rs` must be added with `default-features = false, features = ["std", "load-dynamic", "api-28"]` — never `ort-defaults` (static linking), which would reproduce the v2 SIGBUS.
- The `nemotron` Cargo feature must not add a second `ort` dependency node — it enables `ort/load-dynamic` on the crate's existing optional `ort` dependency (already declared for `diarization`).
- Default build (no `nemotron`, no `diarization`) must have zero `ort`/`parakeet-rs` in its dependency graph.
- `TimestampMode::Sentences` must never be used for Nemotron (it has no punctuation prediction) — only `Tokens` and `Words`.
- Every new `Error` variant must be mapped in `src/python/mod.rs::to_pyerr`; that match has no catch-all arm, so an unmapped variant is a compile error (this is the safety net — do not add a wildcard arm to silence it).
- Nemotron's Python constructor takes no beam/temperature/patience/length-penalty/repetition-penalty knobs — those are Whisper-decoding concepts that do not apply to Nemotron's fixed greedy decoder.

---

## Task 1: Move the onnxruntime dylib-locate module out of `diarize`

`src/diarize/dylib.rs` contains generic "find and init libonnxruntime" logic with no diarization-specific content, but it is currently only compiled under `#[cfg(feature = "diarization")]` inside the `diarize` module. Task 6 needs the exact same logic under `nemotron`. Moving it to a shared top-level module now (with its existing tests carried over unchanged) avoids duplicating the SIGBUS-avoidance code — and duplicated `OnceLock<bool>` init guards would be a real bug: two independent `Once` cells racing to call `ort::init_from` from two features is exactly the kind of thing this move prevents.

**Files:**
- Create: `src/onnx.rs`
- Modify: `src/lib.rs:1-11` (module declarations)
- Modify: `src/diarize/mod.rs:1-11` (remove `dylib` submodule)
- Modify: `src/diarize/polyvoice.rs:1-10,118` (import path)
- Delete: `src/diarize/dylib.rs`

**Interfaces:**
- Produces: `crate::onnx::init_ort() -> crate::error::Result<()>`, gated `#[cfg(any(feature = "diarization", feature = "nemotron"))]`. Task 6 (`NemotronAsr::new`) calls this exact function.

- [ ] **Step 1: Copy `src/diarize/dylib.rs` to `src/onnx.rs` verbatim**

```bash
git mv src/diarize/dylib.rs src/onnx.rs
```

- [ ] **Step 2: Change the new file's gating comment and confirm its `use` lines still resolve**

`src/onnx.rs` already only uses `crate::error::{Error, Result}`, `std::path::PathBuf`, `std::sync::OnceLock`, and `pyo3` — nothing diarize-specific. No code changes needed inside the file; only its location and the module wiring around it change. Update the top doc comment's first line from:

```rust
//! Locating libonnxruntime at runtime.
```

to:

```rust
//! Locating libonnxruntime at runtime.
//!
//! Shared by the `diarization` and `nemotron` features: both need the same
//! `load-dynamic` onnxruntime to coexist with CTranslate2's static
//! `protobuf` (see `docs/superpowers/specs/2026-08-25-whisper-rs-v2-diarization-design.md`),
//! and initializing `ort` twice from two independent `OnceLock`s would be
//! the bug this module exists to prevent.
```

- [ ] **Step 3: Wire the new module into `lib.rs`**

In `src/lib.rs`, replace:

```rust
mod asr;
mod audio;
mod chunk;
#[doc(hidden)]
pub mod diarize;
mod error;
mod models;
mod pipeline;
mod python;
mod stitch;
mod types;
mod vad;
```

with:

```rust
mod asr;
mod audio;
mod chunk;
#[doc(hidden)]
pub mod diarize;
mod error;
mod models;
#[cfg(any(feature = "diarization", feature = "nemotron"))]
mod onnx;
mod pipeline;
mod python;
mod stitch;
mod types;
mod vad;
```

- [ ] **Step 4: Remove the old submodule declaration from `diarize/mod.rs`**

In `src/diarize/mod.rs`, remove the line:

```rust
#[cfg(feature = "diarization")]
pub mod dylib;
```

- [ ] **Step 5: Update `polyvoice.rs`'s import and call site**

In `src/diarize/polyvoice.rs`, change:

```rust
use super::{dylib, Diarizer, SpeakerTurn};
```

to:

```rust
use super::{Diarizer, SpeakerTurn};
use crate::onnx;
```

and change the call:

```rust
        dylib::init_ort()?;
```

to:

```rust
        onnx::init_ort()?;
```

- [ ] **Step 6: Build and run the existing diarization test suite to confirm nothing broke**

```bash
cargo test --features diarization --lib
cargo build --features diarization
```

Expected: compiles clean, all existing tests (including `onnx`'s own three unit tests, now running under the new module path) pass.

- [ ] **Step 7: Confirm the default build still excludes `onnx`**

```bash
cargo build
```

Expected: compiles clean (the `#[cfg(any(...))]` on `mod onnx;` keeps it out of the default build; nothing else references it unconditionally).

- [ ] **Step 8: Commit**

```bash
git add src/onnx.rs src/lib.rs src/diarize/mod.rs src/diarize/polyvoice.rs
git commit -m "refactor: share the onnxruntime dylib-locate module between diarization and nemotron"
```

---

## Task 2: Generalize `SegmentIterator` to `Arc<dyn Asr>`

`SegmentIterator` currently hardcodes `Arc<Ct2Asr>`. `NemotronModel` (Task 8) needs to drive the same iterator with `Arc<NemotronAsr>`. `Asr` is already object-safe (every method takes `&self`, returns `Result<...>` — no generics, no `Self` by value), so switching the field type to `Arc<dyn Asr>` is a pure widening: `Arc<Ct2Asr>` coerces to `Arc<dyn Asr>` at the call site with no change to `Ct2Asr` or the trait itself.

**Files:**
- Modify: `src/python/iter.rs:1-38` (field type, constructor)
- Modify: `src/python/model.rs` (call site passing `Arc<Ct2Asr>` into `SegmentIterator::new`)

**Interfaces:**
- Consumes: `crate::asr::Asr` (existing trait, `src/asr/mod.rs`).
- Produces: `SegmentIterator::new(asr: Arc<dyn Asr>, windows: Vec<Window>, language: String, word_timestamps: bool, turns: Vec<SpeakerTurn>) -> SegmentIterator`. Task 8 constructs `NemotronModel`'s iterator with this exact signature, passing `Arc<NemotronAsr>` coerced to `Arc<dyn Asr>`.

- [ ] **Step 1: Change the struct field and constructor signature in `src/python/iter.rs`**

Replace:

```rust
use crate::asr::{ct2::Ct2Asr, Asr};
```

with:

```rust
use crate::asr::Asr;
```

Replace:

```rust
#[pyclass]
pub struct SegmentIterator {
    asr: Arc<Ct2Asr>,
```

with:

```rust
#[pyclass]
pub struct SegmentIterator {
    asr: Arc<dyn Asr>,
```

Replace:

```rust
impl SegmentIterator {
    pub fn new(
        asr: Arc<Ct2Asr>,
```

with:

```rust
impl SegmentIterator {
    pub fn new(
        asr: Arc<dyn Asr>,
```

(The `Arc::clone(&slf.asr)` call inside `__next__` needs no change — `Arc<dyn Asr>` clones the same way.)

- [ ] **Step 2: Update the call site in `src/python/model.rs`**

`WhisperModel::transcribe` currently builds `SegmentIterator::new(asr_for_prep, ...)` or similar with `asr_for_prep: Arc<Ct2Asr>` (grep the exact local variable name and line first — it is built a few lines after the `py.detach` block that returns `(windows, info, turns)`, from the same `asr` `Arc<Ct2Asr>` used for `detect_language` earlier in the function). Change that construction site to coerce explicitly:

```rust
Ok((
    SegmentIterator::new(
        asr as Arc<dyn crate::asr::Asr>,
        windows,
        info.language.clone(),
        word_timestamps,
        turns,
    ),
    info.into(),
))
```

(Adjust variable names to match whatever the surrounding code actually calls them — the coercion (`as Arc<dyn crate::asr::Asr>`) is the only substantive change; `Arc<Ct2Asr>` values already in scope need no other modification.)

- [ ] **Step 3: Build**

```bash
cargo build
cargo build --features diarization
```

Expected: compiles clean. If the compiler reports an unsized-coercion error, confirm `Ct2Asr` still only implements `Asr` via `&self` methods (it does — no change was made to `ct2.rs`) and that the `as Arc<dyn Asr>` cast targets the trait via its fully-qualified path.

- [ ] **Step 4: Run the existing iterator tests**

```bash
cargo test iter::tests
```

Expected: `ids_thread_sequentially_across_windows_via_advance`, `turns_reach_the_words_through_advance`, and `no_turns_leaves_speakers_unset_and_splits_nothing` all still pass unchanged — they exercise `advance()`, which never touched `asr` and is unaffected by this refactor.

- [ ] **Step 5: Run the full test suite**

```bash
cargo test
cargo test --features diarization
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/python/iter.rs src/python/model.rs
git commit -m "refactor: generalize SegmentIterator to Arc<dyn Asr> so a second ASR backend can drive it"
```

---

## Task 3: Add `Error::Nemotron` and map it in `to_pyerr`

**Files:**
- Modify: `src/error.rs` (new variant)
- Modify: `src/python/mod.rs` (map it, add a test)

**Interfaces:**
- Produces: `crate::error::Error::Nemotron(String)`. Task 6 constructs this from `parakeet-rs`'s `eyre::Report` errors via `.to_string()`.

- [ ] **Step 1: Add the variant to `src/error.rs`**

Add, after the existing `Diarize` variant:

```rust
    #[error("nemotron failed: {0}")]
    Nemotron(String),
```

So the enum tail reads:

```rust
    #[error("diarization failed: {0}")]
    Diarize(String),

    #[error("nemotron failed: {0}")]
    Nemotron(String),

    #[error("{message}")]
    OnnxRuntimeMissing { message: String },
```

- [ ] **Step 2: Write the failing test in `src/python/mod.rs`**

Add to the `tests` module, after `backend_errors_become_runtime_errors`:

```rust
    #[test]
    fn nemotron_errors_become_runtime_errors() {
        Python::initialize();
        Python::attach(|py| {
            let err = to_pyerr(Error::Nemotron("boom".into()));
            assert!(err.is_instance_of::<PyRuntimeError>(py));
        });
    }
```

- [ ] **Step 3: Run it to verify it fails to compile (the match is not yet exhaustive over the new variant in the intended arm)**

```bash
cargo test --lib nemotron_errors_become_runtime_errors
```

Expected: compile error, `to_pyerr`'s match is non-exhaustive (`Error::Nemotron` not covered) — this is `to_pyerr`'s deliberate lack of a catch-all doing its job.

- [ ] **Step 4: Map the variant**

In `src/python/mod.rs`, change:

```rust
        Error::Resample(_) | Error::Vad(_) | Error::Ct2(_) | Error::Diarize(_) => {
            PyRuntimeError::new_err(message)
        }
```

to:

```rust
        Error::Resample(_) | Error::Vad(_) | Error::Ct2(_) | Error::Diarize(_) | Error::Nemotron(_) => {
            PyRuntimeError::new_err(message)
        }
```

- [ ] **Step 5: Run the test again to verify it passes**

```bash
cargo test --lib nemotron_errors_become_runtime_errors
cargo test --lib
```

Expected: PASS, full suite green.

- [ ] **Step 6: Commit**

```bash
git add src/error.rs src/python/mod.rs
git commit -m "feat: add Error::Nemotron, mapped to RuntimeError"
```

---

## Task 4: Add the `nemotron` model registry alias

**Files:**
- Modify: `src/models/registry.rs`

**Interfaces:**
- Produces: `resolve("nemotron") == ModelRef::Hub { repo: "nvidia/nemotron-3.5-asr-streaming-0.6b".into() }`.

- [ ] **Step 1: Write the failing test**

Add to the `tests` module in `src/models/registry.rs`, after `distil_alias_maps_to_the_distil_whisper_org`:

```rust
    #[test]
    fn nemotron_alias_maps_to_the_nvidia_repo() {
        assert_eq!(
            resolve("nemotron"),
            ModelRef::Hub { repo: "nvidia/nemotron-3.5-asr-streaming-0.6b".into() }
        );
    }
```

- [ ] **Step 2: Run it to verify it fails**

```bash
cargo test --lib nemotron_alias_maps_to_the_nvidia_repo
```

Expected: FAIL — `resolve("nemotron")` currently falls through to `ModelRef::Hub { repo: "nemotron".into() }` (the "no alias, no local dir, treat as explicit repo" branch), not the nvidia repo.

- [ ] **Step 3: Add the alias**

In `src/models/registry.rs`, add a new entry to `ALIASES`:

```rust
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
    ("nemotron", "nvidia/nemotron-3.5-asr-streaming-0.6b"),
];
```

- [ ] **Step 4: Run it to verify it passes**

```bash
cargo test --lib nemotron_alias_maps_to_the_nvidia_repo
cargo test --lib registry
```

Expected: PASS, and the pre-existing registry tests are unaffected.

- [ ] **Step 5: Commit**

```bash
git add src/models/registry.rs
git commit -m "feat: add nemotron model alias to the registry"
```

---

## Task 5: Add the `nemotron` Cargo feature and dependency

**Files:**
- Modify: `Cargo.toml`

**Interfaces:**
- Produces: the `nemotron` feature, `dep:parakeet-rs`. Task 6 is the first code to compile behind it.

- [ ] **Step 1: Add the dependency**

In `Cargo.toml`, in `[dependencies]`, add (near the existing `ort` line):

```toml
parakeet-rs = { version = "0.3", default-features = false, features = ["std", "load-dynamic", "api-28"], optional = true }
```

- [ ] **Step 2: Add the feature**

In `[features]`, add:

```toml
nemotron = ["dep:parakeet-rs", "ort/load-dynamic"]
```

So the full `[features]` table reads:

```toml
[features]
extension-module = ["pyo3/extension-module"]
silero-vad = ["wavekat-vad/silero"]
diarization = ["dep:polyvoice", "polyvoice/pipeline-full", "dep:ort", "ort/load-dynamic"]
nemotron = ["dep:parakeet-rs", "ort/load-dynamic"]
```

Note `dep:ort` is not re-listed under `nemotron`: `ort` becomes present in the dependency graph only when something needs it. `parakeet-rs` itself pulls in `ort` as one of its own dependencies (not this crate's optional one) — `ort/load-dynamic` here targets *this crate's* `ort` dependency edge specification, which Cargo unifies with `parakeet-rs`'s `ort` dependency **only if `dep:ort` is also active**, e.g. via `diarization`. Verify this in Step 4 below; if unification does not occur with `nemotron` alone (because this crate's own `ort` dependency is `optional = true` and un-activated), add `"dep:ort"` to the `nemotron` feature list as well so `nemotron` alone still gets a shared, dynamically-linked `ort`.

- [ ] **Step 3: Resolve and inspect the lockfile**

```bash
cargo build --features nemotron
```

Expected: resolves and downloads `parakeet-rs` and its own dependency tree (`ort` rc.13, `tokenizers`, `ndarray` 0.17, `realfft`, `hound`, `eyre`). Compiles clean with no code yet added behind the feature (the feature only affects the dependency graph until Task 6 lands).

- [ ] **Step 4: Verify only one `ort` load-dynamic configuration is active, and that it's shared**

```bash
cargo tree --features nemotron -e features -p ort
```

Inspect the output. If it shows exactly one `ort` node with `load-dynamic` active (and, if `parakeet-rs` also pulled its own separate `ort` unification target, confirm via `cargo tree --features nemotron -i ort` that there is only one version of `ort` in the resolved graph — Cargo unifies by version, so `parakeet-rs`'s `ort = "2.0.0-rc.13"` and this crate's own `ort = "2.0.0-rc.13"` must resolve to the same locked version). If `cargo tree -i ort` shows two distinct `ort` versions, add `"dep:ort"` explicitly to the `nemotron` feature list (per Step 2's note) and re-run this check until only one `ort` version appears.

```bash
cargo build --features "diarization nemotron"
cargo tree --features "diarization nemotron" -i ort
```

Expected: still exactly one `ort` version, now with both `diarization` and `nemotron` contributing to `load-dynamic` on it — this is the check the spec's "Shared `ort` init" section depends on.

- [ ] **Step 5: Confirm the default build is untouched**

```bash
cargo tree | grep -i "parakeet\|ort v" || echo "clean"
```

Expected: `clean` — no `parakeet-rs`, no `ort`, in the default (no-features) dependency tree.

- [ ] **Step 6: Commit**

```bash
git add Cargo.toml Cargo.lock
git commit -m "build: add optional parakeet-rs dependency behind a nemotron feature"
```

---

## Task 6: Implement `NemotronAsr` (`src/asr/nemotron.rs`)

**Files:**
- Create: `src/asr/nemotron.rs`
- Modify: `src/asr/mod.rs` (module declaration)
- Modify: `src/lib.rs` (expose `asr` module publicly, hidden, for the coexistence-test helper added in Task 7)

**Interfaces:**
- Consumes: `crate::asr::Asr` trait (`src/asr/mod.rs`), `crate::error::{Error, Result}`, `crate::types::{Seg, Word}`, `crate::onnx::init_ort` (Task 1).
- Produces:
  ```rust
  pub struct NemotronConfig {
      pub target_lang: Option<String>,
  }
  impl Default for NemotronConfig { /* target_lang: None */ }

  pub struct NemotronAsr { /* private fields */ }
  impl NemotronAsr {
      pub fn new(model_dir: &std::path::Path, config: NemotronConfig) -> Result<Self>;
  }
  impl Asr for NemotronAsr {
      fn transcribe(&self, samples: &[f32], language: Option<&str>, word_timestamps: bool) -> Result<Vec<Seg>>;
      fn detect_language(&self, samples: &[f32]) -> Result<(String, f32)>;
  }
  ```
  Task 8 (`NemotronModel`) constructs `NemotronAsr::new` and wraps it in `Arc<NemotronAsr>`.

- [ ] **Step 1: Declare the module, gated**

In `src/asr/mod.rs`, add:

```rust
pub mod ct2;
pub mod detect;
#[cfg(feature = "nemotron")]
pub mod nemotron;
```

- [ ] **Step 2: Write `src/asr/nemotron.rs`'s skeleton and the language-tag parsing helper first (pure, testable without weights)**

Nemotron's multilingual variant prefixes its token stream with a language tag piece (e.g. `<en-US>`) when running in `auto` mode; `detect_language` needs to recognize this shape. This is the same tag-shape `parakeet-rs`'s own `is_lang_tag` checks internally (private to that crate — reimplemented here since it is not exported):

```rust
//! `parakeet-rs`'s Nemotron (0.6B) backend.
//!
//! Entirely behind the `nemotron` feature: without it, `parakeet-rs` (and
//! `ort`, unless `diarization` also pulls it in) is absent from the
//! dependency graph.

use super::Asr;
use crate::error::{Error, Result};
use crate::types::{Seg, Word};
use parakeet_rs::{Nemotron, NemotronMode, TimestampMode};
use std::path::Path;
use std::sync::Mutex;

#[derive(Debug, Clone, Default)]
pub struct NemotronConfig {
    /// Ignored for `NemotronMode::EnglishOnly`, which has no language
    /// conditioning at all.
    pub target_lang: Option<String>,
}

pub struct NemotronAsr {
    inner: Mutex<Nemotron>,
    mode: NemotronMode,
}

/// A SentencePiece language-tag piece, e.g. `<en-US>` or `<en>`: `<`, two or
/// five inner characters matching `xx` or `xx-XX`, `>`. Mirrors the shape
/// `parakeet-rs`'s own (private) `is_lang_tag` checks, reimplemented here
/// because that helper is not exported.
fn parse_lang_tag(piece: &str) -> Option<&str> {
    let bytes = piece.as_bytes();
    if bytes.len() < 4 || bytes[0] != b'<' || bytes[bytes.len() - 1] != b'>' {
        return None;
    }
    let inner = &piece[1..piece.len() - 1];
    let inner_bytes = inner.as_bytes();
    let shape_ok = match inner_bytes.len() {
        2 => inner_bytes[0].is_ascii_lowercase() && inner_bytes[1].is_ascii_lowercase(),
        5 => {
            inner_bytes[0].is_ascii_lowercase()
                && inner_bytes[1].is_ascii_lowercase()
                && inner_bytes[2] == b'-'
                && inner_bytes[3].is_ascii_uppercase()
                && inner_bytes[4].is_ascii_uppercase()
        }
        _ => false,
    };
    shape_ok.then_some(inner)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_two_letter_tag_parses() {
        assert_eq!(parse_lang_tag("<en>"), Some("en"));
    }

    #[test]
    fn a_locale_tag_parses() {
        assert_eq!(parse_lang_tag("<en-US>"), Some("en-US"));
    }

    #[test]
    fn ordinary_text_is_not_a_tag() {
        assert_eq!(parse_lang_tag("hello"), None);
        assert_eq!(parse_lang_tag("<hi"), None);
        assert_eq!(parse_lang_tag("hi>"), None);
        assert_eq!(parse_lang_tag(""), None);
    }

    #[test]
    fn wrong_case_is_not_a_tag() {
        assert_eq!(parse_lang_tag("<EN>"), None);
        assert_eq!(parse_lang_tag("<en-us>"), None);
    }
}
```

- [ ] **Step 3: Run the pure tests**

```bash
cargo test --features nemotron --lib asr::nemotron
```

Expected: `a_two_letter_tag_parses`, `a_locale_tag_parses`, `ordinary_text_is_not_a_tag`, `wrong_case_is_not_a_tag` all PASS. This is real coverage of the one piece of this file's logic that does not require model weights (per the spec's testing section, which flags this branch as needing confirmation — it turned out testable in isolation).

- [ ] **Step 4: Add `NemotronAsr::new`**

Append to `src/asr/nemotron.rs`:

```rust
impl NemotronAsr {
    pub fn new(model_dir: &Path, config: NemotronConfig) -> Result<Self> {
        crate::onnx::init_ort()?;

        let mut nemotron = Nemotron::from_pretrained(model_dir, None)
            .map_err(|e| Error::Nemotron(format!("failed to load Nemotron model: {e}")))?;
        let mode = nemotron.mode();

        if mode == NemotronMode::Multilingual {
            if let Some(lang) = &config.target_lang {
                nemotron
                    .set_target_lang(lang)
                    .map_err(|e| Error::Nemotron(format!("unsupported target_lang {lang:?}: {e}")))?;
            }
        }

        Ok(Self { inner: Mutex::new(nemotron), mode })
    }
}
```

- [ ] **Step 5: Implement `Asr::transcribe`**

Append:

```rust
impl Asr for NemotronAsr {
    fn transcribe(
        &self,
        samples: &[f32],
        language: Option<&str>,
        word_timestamps: bool,
    ) -> Result<Vec<Seg>> {
        let mut nemotron = self
            .inner
            .lock()
            .map_err(|_| Error::Nemotron("model lock poisoned".into()))?;

        if self.mode == NemotronMode::Multilingual {
            if let Some(lang) = language {
                nemotron
                    .set_target_lang(lang)
                    .map_err(|e| Error::Nemotron(format!("unsupported target_lang {lang:?}: {e}")))?;
            }
        }

        let mode = if word_timestamps { TimestampMode::Words } else { TimestampMode::Tokens };
        let result = nemotron
            .transcribe_audio_with_timestamps(samples, Some(mode))
            .map_err(|e| Error::Nemotron(format!("transcription failed: {e}")))?;

        if result.tokens.is_empty() {
            return Ok(Vec::new());
        }

        let start = result.tokens.first().map(|t| t.start).unwrap_or(0.0);
        let end = result.tokens.last().map(|t| t.end).unwrap_or(start);

        let words = word_timestamps.then(|| {
            result
                .tokens
                .iter()
                .map(|t| Word {
                    start: t.start,
                    end: t.end,
                    text: t.text.clone(),
                    probability: 1.0,
                    speaker: None,
                })
                .collect::<Vec<_>>()
        });

        Ok(vec![Seg {
            id: 0,
            start,
            end,
            text: result.text,
            words,
            speaker: None,
        }])
    }

    fn detect_language(&self, samples: &[f32]) -> Result<(String, f32)> {
        if self.mode == NemotronMode::EnglishOnly {
            return Ok(("en".to_string(), 1.0));
        }

        let mut nemotron = self
            .inner
            .lock()
            .map_err(|_| Error::Nemotron("model lock poisoned".into()))?;

        let result = nemotron
            .transcribe_audio_with_timestamps(samples, Some(TimestampMode::Tokens))
            .map_err(|e| Error::Nemotron(format!("language detection pass failed: {e}")))?;

        match result.tokens.first().and_then(|t| parse_lang_tag(&t.text)) {
            Some(lang) => Ok((lang.to_string(), 1.0)),
            None => Ok(("unknown".to_string(), 0.0)),
        }
    }
}
```

- [ ] **Step 6: Build**

```bash
cargo build --features nemotron
```

Expected: compiles clean. If `parakeet_rs::Nemotron`'s actual method names/signatures differ from what this step assumes (`from_pretrained`, `mode`, `set_target_lang`, `transcribe_audio_with_timestamps`, `TimedToken { text, start, end }`, `TranscriptionResult { text, tokens }`), fix the call sites to match the installed `parakeet-rs` 0.3.x API — consult `~/.cargo/registry/src/*/parakeet-rs-0.3.*/src/nemotron.rs` and `timestamps.rs` directly (this plan's Task 6 design was verified by reading `parakeet-rs` 0.3.7's source on GitHub during spec review; a patch release could have shifted names).

- [ ] **Step 7: Run the full nemotron-gated test suite**

```bash
cargo test --features nemotron --lib
```

Expected: all pass, including the pure `parse_lang_tag` tests from Step 3.

- [ ] **Step 8: Commit**

```bash
git add src/asr/nemotron.rs src/asr/mod.rs
git commit -m "feat: implement NemotronAsr, the Nemotron backend for the Asr trait"
```

---

## Task 7: Expose a `nemotron_for_test` helper and extend the coexistence test

Mirrors the existing `diarize_for_test` helper (`src/lib.rs`), which exists because `diarize`/`asr` are otherwise private modules unreachable from the `tests/` integration-test crate.

**Files:**
- Modify: `src/lib.rs`
- Modify: `tests/coexistence.rs`

**Interfaces:**
- Consumes: `NemotronAsr::new` (Task 6).
- Produces: `whisper_rs::nemotron_for_test(dir: &Path) -> Result<impl asr::Asr>`, gated `#[cfg(feature = "nemotron")]`.

- [ ] **Step 1: Make `asr` reachable, hidden, like `diarize` already is**

In `src/lib.rs`, change:

```rust
mod asr;
```

to:

```rust
#[doc(hidden)]
pub mod asr;
```

- [ ] **Step 2: Add the test helper**

In `src/lib.rs`, after the existing `diarize_for_test` function, add:

```rust
/// Construct a Nemotron backend for integration tests.
///
/// `asr` is a private-in-spirit module (only `#[doc(hidden)] pub` for this
/// reason); the coexistence test in `tests/` cannot reach `NemotronAsr`
/// directly otherwise.
#[cfg(feature = "nemotron")]
#[doc(hidden)]
pub fn nemotron_for_test(model_dir: &std::path::Path) -> crate::error::Result<impl asr::Asr> {
    asr::nemotron::NemotronAsr::new(model_dir, asr::nemotron::NemotronConfig::default())
}
```

- [ ] **Step 3: Build to confirm the new pub surface compiles**

```bash
cargo build --features nemotron
cargo build --features "diarization nemotron"
```

Expected: compiles clean.

- [ ] **Step 4: Extend `tests/coexistence.rs` with the three-way test**

The file currently opens with `#![cfg(feature = "diarization")]`, which would exclude a `nemotron`-only build from even seeing the new test. Change the file-level gate and add the new test. Replace:

```rust
#![cfg(feature = "diarization")]

use whisper_rs::diarize::Diarizer;
```

with:

```rust
#![cfg(any(feature = "diarization", feature = "nemotron"))]

#[cfg(feature = "diarization")]
use whisper_rs::diarize::Diarizer;
#[cfg(feature = "nemotron")]
use whisper_rs::asr::Asr;
```

Then append, at the end of the file:

```rust
/// CTranslate2, onnxruntime-via-polyvoice, and onnxruntime-via-parakeet-rs
/// in one process. Only meaningful with both features on: this is the
/// specific combination the v3 design's "Shared `ort` init" section depends
/// on (`onnx::init_ort` must be safely callable from both `PolyvoiceDiarizer::new`
/// and `NemotronAsr::new` in the same process without a second, racing
/// `ort::init_from`).
#[cfg(all(feature = "diarization", feature = "nemotron"))]
#[test]
#[ignore = "needs CT2_MODEL_DIR, NEMOTRON_MODEL_DIR, and an installed onnxruntime"]
fn ctranslate2_diarization_and_nemotron_coexist() {
    let ct2_dir = std::path::PathBuf::from(
        std::env::var("CT2_MODEL_DIR").expect("set CT2_MODEL_DIR"),
    );
    let nemotron_dir = std::path::PathBuf::from(
        std::env::var("NEMOTRON_MODEL_DIR").expect("set NEMOTRON_MODEL_DIR"),
    );

    let whisper = ct2rs::Whisper::new(&ct2_dir, Default::default()).expect("whisper load");
    let diarizer = whisper_rs::diarize_for_test(4).expect("diarizer construction");
    let nemotron = whisper_rs::nemotron_for_test(&nemotron_dir).expect("nemotron construction");

    let silence = vec![0.0f32; 16_000];
    let _ = diarizer.diarize(&silence);
    let _ = nemotron.transcribe(&silence, None, false);

    let window = vec![0.0f32; whisper.n_samples()];
    let out = whisper
        .generate(&window, Some("en"), false, &Default::default())
        .expect("whisper still works after both onnx backends ran");

    eprintln!("three-way coexistence OK, whisper returned {out:?}");
}
```

- [ ] **Step 5: Build the test crate (without running the ignored test — no models available here)**

```bash
cargo test --features "diarization nemotron" --test coexistence --no-run
```

Expected: compiles clean. Do not remove the `#[ignore]` or attempt to run it without real model directories and an installed onnxruntime — this mirrors exactly how the existing `ctranslate2_and_onnxruntime_coexist` test is (and must remain) skipped in normal CI runs.

- [ ] **Step 6: Confirm the single-feature builds still compile the file correctly**

```bash
cargo test --features diarization --test coexistence --no-run
cargo test --features nemotron --test coexistence --no-run
```

Expected: both compile clean — the original test builds under `diarization` alone (its own `#[cfg(feature = "diarization")]` import gate and lack of any gate on itself means it's still only meaningful there, but the file itself must not fail to compile under `nemotron`-only, where `Diarizer`/`diarize_for_test` are unavailable).

- [ ] **Step 7: Commit**

```bash
git add src/lib.rs tests/coexistence.rs
git commit -m "test: add a three-way CTranslate2 + diarization + nemotron coexistence test"
```

---

## Task 8: `NemotronModel` Python class

**Files:**
- Create: `src/python/nemotron_model.rs`
- Modify: `src/python/mod.rs` (module declaration)
- Modify: `src/lib.rs` (`#[pymodule]` registration)

**Interfaces:**
- Consumes: `crate::asr::nemotron::{NemotronAsr, NemotronConfig}` (Task 6), `crate::models::hub::{ensure_model, FetchOptions}`, `crate::models::registry` (Task 4, via `ensure_model`), `crate::pipeline::prepare`, `crate::python::iter::SegmentIterator` (Task 2's `Arc<dyn Asr>` signature), `crate::python::segment::TranscriptionInfo`, `crate::vad::VadParams`.
- Produces: `#[pyclass] NemotronModel`, registered in the `whisper_rs` pymodule as `NemotronModel`.

- [ ] **Step 1: Write `src/python/nemotron_model.rs`**

This mirrors `WhisperModel` (`src/python/model.rs`) structurally — same eager prepare/VAD/language-detect/diarize split, same tri-state `word_timestamps`/`diarize` validation — with the beam-search-family parameters removed and `target_lang` added:

```rust
//! `NemotronModel`: the Nemotron backend exposed to Python.
//!
//! Structurally parallel to `WhisperModel` (`model.rs`) — same eager
//! prepare/VAD/language-detect/diarize split — but with none of Whisper's
//! beam-search decoding knobs, which do not apply to Nemotron's fixed
//! greedy decoder.

use crate::asr::nemotron::{NemotronAsr, NemotronConfig};
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

const DEFAULT_MAX_SPEAKERS: usize = 8;

/// A loaded Nemotron model, ready to transcribe audio files.
///
/// # Example
/// ```python
/// import whisper_rs
///
/// model = whisper_rs.NemotronModel("nemotron")
/// segments, info = model.transcribe("audio.wav")
/// for segment in segments:
///     print(segment.start, segment.end, segment.text)
/// ```
#[pyclass]
pub struct NemotronModel {
    asr: Arc<NemotronAsr>,
    model_dir: PathBuf,
}

#[pymethods]
impl NemotronModel {
    #[new]
    #[pyo3(signature = (
        model,
        *,
        download_root = None,
        local_files_only = false,
        target_lang = None,
    ))]
    fn new(
        py: Python<'_>,
        model: &str,
        download_root: Option<PathBuf>,
        local_files_only: bool,
        target_lang: Option<String>,
    ) -> PyResult<Self> {
        let opts = FetchOptions { download_root, local_files_only };
        let config = NemotronConfig { target_lang };

        let name = model.to_string();
        let (asr, model_dir) = py
            .detach(move || -> crate::error::Result<(NemotronAsr, PathBuf)> {
                let dir = ensure_model(&name, &opts)?;
                let asr = NemotronAsr::new(&dir, config)?;
                Ok((asr, dir))
            })
            .map_err(to_pyerr)?;

        Ok(Self { asr: Arc::new(asr), model_dir })
    }

    #[getter]
    fn model_path(&self) -> String {
        self.model_dir.display().to_string()
    }

    #[pyo3(signature = (
        audio,
        *,
        language = None,
        word_timestamps = None,
        vad_filter = true,
        vad_parameters = None,
        diarize = false,
        max_speakers = DEFAULT_MAX_SPEAKERS,
        num_speakers = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn transcribe(
        &self,
        py: Python<'_>,
        audio: PathBuf,
        language: Option<String>,
        word_timestamps: Option<bool>,
        vad_filter: bool,
        vad_parameters: Option<Bound<'_, PyDict>>,
        diarize: bool,
        max_speakers: usize,
        num_speakers: Option<usize>,
    ) -> PyResult<(SegmentIterator, TranscriptionInfo)> {
        if !diarize {
            if max_speakers != DEFAULT_MAX_SPEAKERS {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "max_speakers={max_speakers} has no effect without diarize=True. \
                     Pass diarize=True, or leave max_speakers unset."
                )));
            }
            if let Some(k) = num_speakers {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "num_speakers={k} has no effect without diarize=True. \
                     Pass diarize=True, or leave num_speakers unset."
                )));
            }
        }

        let word_timestamps = match (word_timestamps, diarize) {
            (Some(false), true) => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "word_timestamps=False cannot be combined with diarize=True: \
                     speakers are assigned per word, so word timestamps are required. \
                     Pass word_timestamps=True, or leave it unset to have it enabled \
                     automatically.",
                ))
            }
            (Some(explicit), _) => explicit,
            (None, diarize) => diarize,
        };

        let params = crate::python::model::vad_params_from_dict(vad_parameters.as_ref())?;

        let asr = Arc::clone(&self.asr);
        let asr_for_prep = Arc::clone(&asr);
        let path: PathBuf = audio;
        let (windows, info, turns) = py
            .detach(move || -> crate::error::Result<_> {
                let prepared = crate::pipeline::prepare(Path::new(&path), vad_filter, &params)?;
                let mut info = prepared.info;

                match language {
                    Some(code) => info.language = code,
                    None => match prepared.windows.first() {
                        Some(w) => {
                            let (code, probability) = asr_for_prep.detect_language(&w.samples)?;
                            info.language = code;
                            info.language_probability = Some(probability);
                        }
                        None => info.language = "unknown".to_string(),
                    },
                }

                let turns = if diarize {
                    #[cfg(feature = "diarization")]
                    {
                        let diarizer = crate::diarize::polyvoice::PolyvoiceDiarizer::new(
                            max_speakers,
                            num_speakers,
                        )?;
                        crate::diarize::Diarizer::diarize(&diarizer, &prepared.samples)?
                    }
                    #[cfg(not(feature = "diarization"))]
                    {
                        return Err(crate::error::Error::Nemotron(
                            "diarize=True requires the diarization feature".into(),
                        ));
                    }
                } else {
                    Vec::new()
                };

                info.num_speakers = diarize.then(|| {
                    turns.iter().map(|t| t.speaker).collect::<std::collections::HashSet<_>>().len()
                });

                Ok((prepared.windows, info, turns))
            })
            .map_err(to_pyerr)?;

        Ok((
            SegmentIterator::new(
                asr as Arc<dyn Asr>,
                windows,
                info.language.clone(),
                word_timestamps,
                turns,
            ),
            info.into(),
        ))
    }
}
```

**Note on `vad_params_from_dict`:** this plan assumes `WhisperModel`'s `transcribe` builds `VadParams` via a free function `vad_params_from_dict` in `src/python/model.rs` — grep for it (`grep -n "fn vad_params_from_dict" src/python/model.rs`) before writing this file. If it is a private (non-`pub`) free function, change its declaration to `pub(crate) fn vad_params_from_dict` so `nemotron_model.rs` can call it; do not duplicate its body. Same check for however `WhisperModel` builds the diarization `turns`/`num_speakers` block — this step's version above is written from the spec, not copy-pasted from `model.rs`'s exact private helpers, so reconcile field-for-field against the real `WhisperModel::transcribe` body (`src/python/model.rs`, the section after `let (windows, info, turns) = ...`) and prefer extracting any shared logic (the diarization block, the `num_speakers` computation) into a `pub(crate)` helper both `model.rs` and `nemotron_model.rs` call, rather than duplicating it — this is the "existing code has problems that affect the work" case the writing-plans skill calls out: two independent copies of the diarization-wiring block would drift.

- [ ] **Step 2: Declare the module**

In `src/python/mod.rs`, add:

```rust
pub mod iter;
pub mod model;
#[cfg(feature = "nemotron")]
pub mod nemotron_model;
pub mod segment;
```

- [ ] **Step 3: Register the class**

In `src/lib.rs`, in the `#[pymodule]` function, add:

```rust
#[pymodule]
fn whisper_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    init_tracing();
    m.add_class::<python::model::WhisperModel>()?;
    #[cfg(feature = "nemotron")]
    m.add_class::<python::nemotron_model::NemotronModel>()?;
    m.add_class::<python::iter::SegmentIterator>()?;
    m.add_class::<python::segment::Segment>()?;
    m.add_class::<python::segment::Word>()?;
    m.add_class::<python::segment::TranscriptionInfo>()?;
    Ok(())
}
```

- [ ] **Step 4: Build**

```bash
cargo build --features nemotron
cargo build --features "diarization nemotron"
cargo build --features nemotron --features extension-module
```

Expected: compiles clean on all three. Fix any field/type mismatches surfaced against the real `WhisperModel::transcribe`/`vad_params_from_dict` per Step 1's note.

- [ ] **Step 5: Run the full nemotron test suite**

```bash
cargo test --features nemotron --lib
cargo test --features "diarization nemotron" --lib
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/python/nemotron_model.rs src/python/mod.rs src/lib.rs
git commit -m "feat: add NemotronModel, the Python entry point for the Nemotron backend"
```

---

## Task 9: Python packaging (`nemotron` extra) and `pyproject.toml`

**Files:**
- Modify: `pyproject.toml` (or wherever the `[diarization]` extra / maturin feature mapping currently lives — grep first)

**Interfaces:** none new — this is packaging metadata only.

- [ ] **Step 1: Locate the existing `diarization` extra wiring**

```bash
grep -rn "diarization" pyproject.toml Cargo.toml 2>/dev/null
```

Find how `pip install whisper-rs[diarization]` currently maps to the Cargo `diarization` feature (maturin's `[tool.maturin]` config, a `features` list, or an extras-to-Cargo-feature build script). This plan does not assume its exact shape — read it before editing.

- [ ] **Step 2: Add the equivalent `nemotron` extra**

Add a `nemotron` extra following whatever pattern Step 1 found for `diarization`, mapping to the Cargo `nemotron` feature. If `diarization` is wired as a maturin feature flag passed at build time (e.g. `maturin build --features diarization`) rather than a true pip extras mechanism (Cargo features are compile-time, so a single wheel cannot conditionally include both unless both are compiled in), mirror that exact mechanism — do not invent a different packaging approach for `nemotron` than the one already chosen for `diarization`.

- [ ] **Step 3: Build the wheel locally to confirm it doesn't break**

```bash
maturin build --features nemotron 2>&1 | tail -30
```

Expected: builds (or fails with a clear, expected error if system dependencies like a C++ toolchain are the blocker — not a Cargo/pyproject configuration error).

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "build: add nemotron packaging extra alongside diarization"
```

---

## Task 10: End-to-end Python test (model-gated)

**Files:**
- Modify: `tests/python/` (find the existing `@pytest.mark.model`-style test file for `WhisperModel`, e.g. `test_transcribe.py` or similar — grep first)

**Interfaces:** none new — exercises `NemotronModel` end to end.

- [ ] **Step 1: Find the existing model-gated Whisper test for its exact fixture/marker pattern**

```bash
grep -rln "pytest.mark.model" tests/python/
```

Read one such test in full to copy its audio-fixture setup (how it gets real speech audio — likely `say` on macOS, per the diarization spec's testing section, or a checked-in short WAV).

- [ ] **Step 2: Write the failing test**

In a new file `tests/python/test_nemotron.py` (or alongside the existing model tests if convention favors one file — match whatever Step 1 found):

```python
import pytest
import whisper_rs


@pytest.mark.model
def test_nemotron_transcribes_real_speech(short_speech_wav):
    # `short_speech_wav` fixture: reuse whatever fixture name/conftest.py
    # entry the existing WhisperModel model-gated tests use for real audio
    # (see Step 1) -- do not invent a second audio-generation path.
    model = whisper_rs.NemotronModel("nemotron")
    segments, info = model.transcribe(str(short_speech_wav), word_timestamps=True)
    segments = list(segments)

    assert len(segments) > 0
    assert any(seg.text.strip() for seg in segments)
    first_words = segments[0].words
    assert first_words is not None
    assert all(w.end >= w.start for w in first_words)
```

- [ ] **Step 3: Run it to verify it fails for the right reason (no model downloaded / feature not built) or is properly skipped**

```bash
pytest tests/python/test_nemotron.py -v -m model
```

Expected: either skipped (if `pytest.mark.model` is configured to skip by default without `--run-model-tests` or similar — match whatever the existing tests do) or fails cleanly on a missing extension build, not an import error in the test file itself.

- [ ] **Step 4: Build the extension with the feature and run for real (local verification only — this step is not expected to run unattended in CI without network + weights)**

```bash
maturin develop --features nemotron
pytest tests/python/test_nemotron.py -v -m model --run-model-tests  # flag name: match Step 1's convention
```

Expected: PASS, given network access to download `nvidia/nemotron-3.5-asr-streaming-0.6b` and a real speech fixture.

- [ ] **Step 5: Commit**

```bash
git add tests/python/test_nemotron.py
git commit -m "test: add an end-to-end NemotronModel transcription test"
```

---

## Definition of Done (from the spec — verify each before considering this plan complete)

1. `pip install whisper-rs[nemotron]` produces a working `NemotronModel` — Task 9.
2. `NemotronModel("nemotron").transcribe(audio)` returns text with word timestamps — Task 6, Task 10.
3. Multilingual checkpoint: `mode() == Multilingual` detected, `target_lang` honoured, tag-derived `detect_language` — Task 6 (`detect_language`'s `Multilingual` branch; this specific checkpoint's live behaviour is only verified when Task 10's test is run against a real multilingual repo, which the default `"nemotron"` alias does not point at — note this gap if the multilingual checkpoint's HF repo name differs from the English-only one).
4. Default build has no `parakeet-rs`; `nemotron` alone still links onnxruntime dynamically — Task 5, Steps 4-5.
5. `diarization` + `nemotron` together do not reproduce the SIGBUS — Task 7's three-way coexistence test (must actually be run manually with real model dirs before claiming this; `#[ignore]`d tests do not run in normal `cargo test`).
6. `NemotronModel` composes with `diarize=True` with no Nemotron-specific code in `diarize/` — Task 8 (reuses `diarize::polyvoice::PolyvoiceDiarizer` unchanged).

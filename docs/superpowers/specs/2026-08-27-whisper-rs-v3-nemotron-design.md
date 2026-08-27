# whisper-rs v3: Nemotron ASR backend

Status: approved design, not yet implemented.
Extends `2026-08-24-whisper-rs-design.md` (v1) and
`2026-08-25-whisper-rs-v2-diarization-design.md` (v2). v2 explicitly scoped
Nemotron out ("Nemotron, or any second ASR engine. Out of scope entirely.")
— this is that follow-up.

## Goal

A second ASR backend, NVIDIA Nemotron (0.6B, FastConformer streaming model,
via `parakeet-rs`'s `Nemotron` struct), usable from Python as `NemotronModel`
alongside the existing `WhisperModel`. Not a replacement for Whisper — a
second engine behind the same `Asr` trait, reusing the existing audio/VAD/
windowing pipeline.

## Non-goals

- **Streaming (`transcribe_chunk`) API.** v1/v2's pipeline is whole-file,
  eager, batch. Nemotron's streaming chunk API is not exposed.
- **`Parakeet`/`ParakeetTDT`/other `parakeet-rs` model variants.** Only the
  `Nemotron` struct.
- **Diarization changes.** The existing `diarize` feature is backend-agnostic
  at the `Diarizer` trait level already; nothing here changes it. Combining
  `diarize=True` with `NemotronModel` is in scope only insofar as it falls
  out of both using the same `Info`/`Word`/`Seg` types — no new work.
- **Accurate multilingual language auto-detection.** See "detect_language"
  below — the multilingual variant's detection is a best-effort reading of
  the model's own leading language tag, not a real detector pass like v1's
  Whisper encoder trick.
- **GPU / CUDA / CoreML execution providers.** CPU only, matching the
  `diarization` feature's current scope.

## The constraint this design must not re-break

v2 spent an entire task establishing that CTranslate2 and onnxruntime can
only coexist in one process when onnxruntime is loaded via `ort`'s
`load-dynamic` feature (a dylib resolved at runtime), never statically
linked — static linking produces a `protobuf` symbol collision and a
`SIGBUS`. This is why `diarization`'s Cargo feature forces
`polyvoice/load-dynamic`.

`parakeet-rs` depends on `ort` directly, and its `default` features
(`cpu`, `ort-defaults`, `api-28`) pull in `ort/default` — static linking.
The `nemotron` feature below must disable that default and force
`ort/load-dynamic` explicitly, the same way `diarization` does, so the two
features share one dynamically-loaded onnxruntime instance rather than each
trying to statically link their own copy (which would collide with each
other *and* with CTranslate2).

`ort` is already a direct optional dependency of this crate (added for
`diarization`) — `nemotron` reuses that same `ort` node via Cargo feature
unification; it does not add a second one.

## Approach

`parakeet-rs::Nemotron` auto-detects, from the loaded ONNX graph, whether the
checkpoint is the English-only variant (vocab 1024, no language
conditioning) or the Multilingual 3.5 variant (vocab ~13k, `prompt_index`
input, `set_target_lang`). Both drop into the same `Nemotron` type — this
design supports both by reading `Nemotron::mode()` at load time rather than
picking one ahead of time.

Its transcription API returns `TranscriptionResult { text, tokens: Vec<TimedToken> }`
where each `TimedToken { text, start, end }` is already in seconds. Grouping
is controlled by `TimestampMode`:

- `Tokens` — raw subword pieces (`transcribe_audio_with_timestamps` default).
- `Words` — subwords merged into words.
- `Sentences` — grouped by punctuation. **Not used here**: Nemotron 0.6B is
  a CTC-style greedy decoder with no punctuation prediction (unlike
  `ParakeetTDT`), so `Sentences` mode would not produce meaningful
  boundaries. Confirmed by `parakeet-rs`'s own doc comment on
  `TimestampMode::Sentences` ("only works with models that predict
  punctuation, e.g. Parakeet TDT").

Consequence: **one Nemotron call produces one flat transcript with no
internal sentence segmentation.** This is not a limitation introduced by
this design — it's the model. It composes cleanly with the existing pipeline
because `pipeline::prepare` already hands the ASR backend one VAD/silence-
bounded window at a time and expects `Vec<Seg>` back; Nemotron's backend
simply always returns a `Vec` of length 0 or 1 per window (0 for a window
whose decode produced no tokens, 1 otherwise), rather than Whisper's
internal multi-segment split within a 30 s window.

## Components

### `src/asr/nemotron.rs`

```rust
pub struct NemotronConfig {
    pub target_lang: Option<String>,   // ignored for EnglishOnly mode
}

pub struct NemotronAsr {
    inner: Mutex<parakeet_rs::Nemotron>,
    mode: parakeet_rs::NemotronMode,
    config: NemotronConfig,
}

impl NemotronAsr {
    pub fn new(model_dir: &Path, config: NemotronConfig) -> Result<Self>;
}

impl Asr for NemotronAsr {
    fn transcribe(&self, samples: &[f32], language: Option<&str>, word_timestamps: bool) -> Result<Vec<Seg>>;
    fn detect_language(&self, samples: &[f32]) -> Result<(String, f32)>;
}
```

**Why `Mutex`, unlike `Ct2Asr`.** `ct2rs::Whisper` is verified `Send + Sync`
on its own (v1's `whisper_is_send_and_sync` test) — its interior mutability
is behind CTranslate2's own thread-safe session. `parakeet_rs::Nemotron`
holds its streaming decoder state (`encoder_cache`, `state_1`, `state_2`,
`accumulated_tokens`, ...) as plain fields with no internal lock, and every
transcription method takes `&mut self`. The `Asr` trait requires
`Send + Sync` behind a `&self` call (`Arc<dyn Asr>`/`Arc<NemotronAsr>` in the
Python layer), so `NemotronAsr` supplies the lock itself: `.lock().unwrap()`
around each `transcribe`/`detect_language` call. This serializes concurrent
calls on one model instance — acceptable, since v1/v2 already serialize
Python-side transcription per model object; it does not preclude loading two
independent `NemotronModel` instances for concurrency.

**`transcribe`:**

1. If `mode == Multilingual` and `language.is_some()`, call
   `inner.set_target_lang(language)` before transcribing (ignore for
   `EnglishOnly`, which has no language conditioning — matches `available_languages()`
   being empty for that mode).
2. Call `inner.transcribe_audio_with_timestamps(samples, Some(mode))` where
   `mode = if word_timestamps { TimestampMode::Words } else { TimestampMode::Tokens }`.
   `Tokens` mode is still requested (not skipped) when word timestamps are
   off, because segment `start`/`end` are still needed and both modes carry
   per-piece timing — only the granularity of `Seg::words` differs.
3. Empty `tokens` → empty `Vec<Seg>` (silence-only window, matches how
   `ct2rs` can also emit nothing for silence).
4. Otherwise, one `Seg`: `start = tokens[0].start`, `end = tokens.last().end`,
   `text = result.text` (already assembled/detokenized by `parakeet-rs`),
   `words = Some(...)` when `word_timestamps`, built from the `Words`-mode
   tokens with `probability` set to `1.0` (`TimedToken` carries no
   log-probability at `Words`/`Sentences` granularity — only `TokenInfo`,
   from the separate `transcribe_audio_with_tokens` call, has `logprob`, and
   pulling that in for probability alone is not worth a second forward pass).
   `speaker: None` (diarization assigns this later, same as v1).
5. Segment `id` is left at the sentinel used elsewhere pre-`stitch`
   (`0` — `stitch` / `number()` assign real ids); this mirrors how `Ct2Asr`
   hands `stitch` unnumbered segments today.

**`detect_language`:**

- `EnglishOnly` mode: no detection needed or possible — return
  `Ok(("en".to_string(), 1.0))` unconditionally. This mirrors v1's rule
  against fabricating values, but here `"en"` is not fabricated: it is the
  only language the loaded checkpoint can produce.
- `Multilingual` mode: `parakeet-rs` exposes no detect-only call — the model
  only reports what language it used by prefixing the token stream with a
  language-tag piece (e.g. `<en-US>`) when `target_lang("auto")` (prompt
  index 101, the default for a freshly-`from_shared` instance) is active.
  This backend runs one real `transcribe_audio_with_timestamps(samples, Some(TimestampMode::Tokens))`
  pass and reads the first token: if it matches the crate's
  `<xx-XX>`/`<xx>` tag shape, that is the detected language and
  `probability = 1.0` (the model does not expose a real confidence score for
  this, so `1.0` communicates "read from the model's own tag", not
  "measured probability" — documented on the method, not left implicit).
  If no leading tag is found (should not happen at `target_lang("auto")` per
  the model card, but is not guaranteed by the crate's public API), fall
  back to `Ok(("unknown".to_string(), 0.0))` rather than panicking.
  **This duplicates the transcription work `WhisperModel.transcribe` would
  otherwise avoid via Whisper's cheap encoder-only detector** (v1's
  `asr::detect`) — there is no equivalent cheap path for Nemotron. Recorded
  as a known cost, not fixed in v3: `NemotronModel.transcribe(language=...)`
  with an explicit language still skips it entirely, same tri-state
  behaviour as v1.

### `src/models/registry.rs`

New alias:

```rust
("nemotron", "nvidia/nemotron-3.5-asr-streaming-0.6b"),
```

Resolution order (local path / alias / explicit repo) is unchanged — this is
one more row in `ALIASES`, following the existing `Systran/...` pattern. The
HF repo layout `parakeet-rs::NemotronHandle::from_pretrained` needs
(`encoder.onnx` + `encoder.onnx.data`, `decoder_joint.onnx`,
`tokenizer.model`) determines what `ensure_model` downloads — no new fetch
mechanism, this reuses `hf-hub` exactly like the Whisper path (unlike v2's
`polyvoice` models, which needed a GitHub-Releases fetch path because they
are not on the HF Hub).

### `src/python/model.rs`

New `#[pyclass] NemotronModel`, structurally parallel to `WhisperModel` but
narrower — Nemotron has no beam search, no temperature/patience/
length-penalty/repetition-penalty knobs (those are Whisper decoding
concepts; Nemotron's decoder is a fixed greedy RNN-T-style joint network).
Kept constructor surface:

```python
NemotronModel(
    model,                      # alias, HF repo, or local dir
    *,
    download_root=None,
    local_files_only=False,
    target_lang=None,           # multilingual variant only; ignored otherwise
)
```

`transcribe()` has the same signature and eager/lazy split as
`WhisperModel.transcribe` (VAD + windowing + optional diarization eager,
ASR decode lazy per window) — it reuses `pipeline::prepare` unchanged, just
constructs `Arc<NemotronAsr>` instead of `Arc<Ct2Asr>` before handing it to
`SegmentIterator`. `word_timestamps`'s tri-state resolution against
`diarize` is identical to v1/v2 (Nemotron's `Words` mode is exactly what
diarization's per-word assignment needs).

Both classes are registered in `lib.rs`'s `#[pymodule]` fn.

## Feature gating and packaging

```toml
[dependencies]
parakeet-rs = { version = "0.3", default-features = false, features = ["std", "load-dynamic", "api-28"], optional = true }

[features]
nemotron = ["dep:parakeet-rs", "ort/load-dynamic"]
```

`parakeet-rs`'s own Cargo features (`cpu`, `ort-defaults`, `std`, `api-28`,
`load-dynamic`, ...) forward to its internal `ort` — but crucially it is
*the same* `ort` crate node this workspace already declares as a direct
optional dependency for `diarization`. Disabling `parakeet-rs`'s
`ort-defaults` (which pulls `ort/default`, static) and enabling
`load-dynamic` on both `polyvoice` (transitively) and `parakeet-rs`
(directly) keeps exactly one dynamically-loaded onnxruntime in the graph
regardless of which of `diarization`/`nemotron`/both are enabled.

Without the `nemotron` feature, `parakeet-rs` (and, if `diarization` is also
off, `ort`) is entirely absent — default build stays byte-for-byte v1/v2.

Python packaging: `pip install whisper-rs[nemotron]` — same onnxruntime pip
wheel + `ORT_DYLIB_PATH` resolution v2 already built for `diarization`,
reused verbatim (`ort::init_from` is process-global — this must be called
at most once regardless of how many of the two ONNX-backed features are
active; see "Shared ort init" below).

### Shared `ort` init

v2's `ort::init_from(&dylib).commit()` call happens once, lazily, the first
time any ONNX-backed path runs (diarization or, now, Nemotron). This design
does not change that call site's location — it changes what triggers it.
Both `PolyvoiceDiarizer::new` and `NemotronAsr::new` must go through the
same one-time init (a `std::sync::Once` or equivalent already implied by v2;
this document does not re-litigate v2's dylib-resolution code, only notes
that `NemotronAsr::new` is a second caller of it).

## Error handling

New `Error` variant:

- `Error::Nemotron(String)` — backend failure (model load, transcription),
  mapped to `RuntimeError` in Python, following the `Error::Diarize(String)`
  precedent. `parakeet-rs` errors are `eyre::Report`; converted via
  `.to_string()` at the boundary, same pattern v1 uses for other
  string-message variants.

`Error::OnnxRuntimeMissing` (added in v2) is reused as-is — it is already
feature-agnostic ("the dylib could not be found"), not diarization-specific.

`to_pyerr` stays exhaustive; adding `Error::Nemotron` without mapping it is
a compile error, per v1/v2's existing pattern.

## Testing

- **Registry alias test**: `resolve("nemotron")` maps to the HF repo,
  mirroring the existing alias tests in `models/registry.rs`.
- **`NemotronAsr` is not directly unit-testable without the model weights**
  (600 MB+ ONNX graph) — like `Ct2Asr`, correctness here is established by
  an integration test gated behind the `nemotron` feature and a real model
  download, marked to skip in environments without network/weights access
  (matching v1/v2's `@pytest.mark.model` convention on the Python side, and
  a `#[cfg(feature = "nemotron")]` + explicit opt-in env var on the Rust
  side for anything that downloads).
- **Coexistence test extended**: v2's permanent CTranslate2 + onnxruntime
  coexistence test gains a third participant when both `diarization` and
  `nemotron` are enabled together — load a CTranslate2 Whisper model, run a
  `polyvoice` diarization pass, and run a Nemotron transcription pass, all
  in one process. This is the regression test for the constraint section
  above; it must pass before both features are documented as usable
  together.
- **`NemotronConfig`/mode-selection logic** (English-only vs multilingual
  branching in `transcribe`/`detect_language`) is pure enough to unit-test
  with a fake `Transcriber`-shaped stub if `parakeet_rs::Nemotron` cannot be
  constructed without real weights — deferred to the implementation plan to
  confirm whether `parakeet-rs` exposes any seam for this; if not, this
  logic is covered only by the integration test above.

## Definition of done

1. `pip install whisper-rs[nemotron]` produces a working `NemotronModel`.
2. `NemotronModel("nemotron").transcribe(audio)` returns text for real
   English speech, with `word_timestamps=True` giving per-word timing.
3. The multilingual checkpoint, loaded via an explicit HF repo name, detects
   `mode() == Multilingual`, honours `target_lang`, and `detect_language`
   returns a tag-derived language when `language` is not pinned.
4. Default build (no `nemotron` feature) has no `parakeet-rs` in its
   dependency graph, and adding `nemotron` alone (without `diarization`)
   still only links onnxruntime dynamically, never statically.
5. `diarization` and `nemotron` enabled together do not reproduce the v2
   SIGBUS — covered by the extended coexistence test.
6. `NemotronModel` composes with `diarize=True` with no NemotronModel-
   specific code in `diarize/`.

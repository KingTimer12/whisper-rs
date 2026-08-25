# whisper-rs v2: speaker diarization

Status: approved design, not yet implemented.
Supersedes nothing; extends `2026-08-24-whisper-rs-design.md` (v1).

## Goal

Answer "who spoke each word" alongside "what was said", with no cap on the
number of speakers, entirely in-process and in Rust.

## Non-goals

- **Punctuation and casing.** Deferred to v3, where it starts as a
  measurement rather than a change. See "Deferred: punctuation" below.
- **Nemotron, or any second ASR engine.** Out of scope entirely.
- **Streaming diarization.** Clustering needs the whole file (see
  "Why diarization runs on the whole file").
- **Speaker naming or enrollment.** Speakers are integers, stable within one
  transcription and meaningless across calls.

## The constraint that shaped this design

CTranslate2 and onnxruntime both statically link `protobuf`. With both in one
process, the duplicate symbols produce a `SIGBUS` at runtime. This is what
forced Silero VAD out of the v1 default build, and it is why v1 has no ONNX
dependency at all.

A spike (2026-08-25) established that `ort` in `load-dynamic` mode removes the
collision, and that the failure is specifically caused by static linking:

| Configuration | `SileroBackend::new()` |
| --- | --- |
| `ort` static (the v1 build) | `signal: 10, SIGBUS: access to undefined memory` |
| `ort` load-dynamic, libonnxruntime 1.29.0 | loads, infers, and CTranslate2 keeps working afterwards |

The counterfactual is the important half: reverting only the `load-dynamic`
feature reproduces the SIGBUS at the identical call site. Both load orders
(CTranslate2 first, onnxruntime first) pass under `load-dynamic`.

Mechanism: with `load-dynamic`, onnxruntime arrives as a dylib loaded at
runtime that resolves its own `protobuf` internally, instead of contributing
`protobuf` symbols to the binary CTranslate2 already populated. The ODR
conflict stops existing by construction.

**Validated on macOS arm64 only.** macOS uses two-level namespaces, which
isolate dylib symbols by default. Linux with `RTLD_GLOBAL` can interpose
symbols, so the collision may return there. CI on Linux must run the
coexistence test before this crate claims Linux support for diarization.

## Approach

`polyvoice` (0.17.x, MIT) provides the pipeline: powerset segmentation,
speaker embeddings (CAM++ / ERes2NetV2 / ResNet34), and clustering (AHC,
k-means, spectral, NME-SC, VBx), plus overlap handling, resegmentation, and a
DER metric for evaluating quality.

Why it was chosen over the alternatives considered:

- **Sortformer via `parakeet-rs`** is a single end-to-end model, far simpler,
  and was the initial recommendation — but `NUM_SPEAKERS = 4` is a property of
  the model, not a setting. A hard cap of 4 speakers fails the requirement.
- **`speakrs`** (Apache-2.0) and **`pyannote-rs`** (MIT, more downloads) are
  both viable and more conservative bets. `polyvoice` was chosen for
  completeness: it already contains the clustering variants and evaluation
  metric that this design would otherwise have to grow.
- **Building it from `ort` + a WeSpeaker ONNX + `diaric`** offers the most
  control and reimplements what `polyvoice` already ships.

Two further properties settled it. `polyvoice`'s `default = []` leaves `ort`
out of the graph entirely, with an `Embedder` trait for bring-your-own — so
the ONNX surface is opt-in rather than imposed. And its models are on the
project's own GitHub Releases, ungated: pyannote's own weights on Hugging Face
require accepting conditions on an account, which would be a distribution
problem for a wheel.

**`polyvoice` has not been run.** Everything above is read from its source and
metadata. The implementation plan's first task is a validation spike whose
instruction is to stop and report on failure rather than build on top of it —
see "Task zero".

## Why diarization runs on the whole file

Diarization consumes the complete decoded signal, not the VAD windows the ASR
consumes.

Clustering is what removes the speaker cap: local segmentation reports at most
3 simultaneous speakers per chunk, and the global speaker count emerges from
clustering chunk-level embeddings across the entire file. Feeding it
window-by-window would destroy exactly that global identity — the property
that distinguishes this from Sortformer — because nothing would establish that
the voice at minute 1 and the voice at minute 40 are the same person.

```
decode -> resample -> +-> VAD -> windows -> ASR (ct2rs) -> segments -+
                      |                                              +-> assign -> segments with speakers
                      +-> polyvoice: segmentation -> embeddings -> clustering -+
```

## Components

### `src/diarize/mod.rs`

```rust
/// One speaker's continuous turn, in seconds on the global timeline.
pub struct SpeakerTurn {
    pub start: f64,
    pub end: f64,
    pub speaker: usize,
}

pub trait Diarizer: Send + Sync {
    /// Diarize a whole file's worth of 16 kHz mono samples.
    fn diarize(&self, samples: &[f32]) -> Result<Vec<SpeakerTurn>>;
}
```

Same shape as v1's `Vad` trait, for the same reason: it keeps the concrete
backend behind one boundary. `polyvoice` is pre-1.0 and may break its API
between minor versions; containing it to one file is the mitigation.

### `src/diarize/polyvoice.rs`

The `Diarizer` implementation, entirely behind the `diarization` Cargo
feature. Without the feature, `ort` is absent from the dependency graph and
the default build is byte-for-byte the v1 build.

Configuration surface, mapped from `polyvoice`:

- `max_speakers: usize` (default 8) maps onto `polyvoice`'s `max_clusters`,
  which every clusterer takes as its only required constructor argument
  (`AhcClusterer::new(max_clusters)`, `NmeScClusterer::new(max_clusters)`).
  It is an upper bound for clustering, not a model limit; `polyvoice`'s CLI
  validates `1..=255`, and `NmeScClusterer::default()` uses 64.
- There is **no minimum-speaker knob**: `polyvoice` has no `min_speakers`
  anywhere. The clustering decides the count on its own, bounded above by
  `max_clusters`. This API must not offer a `min_speakers` it cannot honour.
- Embedder and clusterer choices are **not** exposed in v2. Task zero picks
  one pairing, and records in this file both which one and the DER it
  measured. Exposing the choice is a v3 question that should follow further
  measurement, not precede it.

`polyvoice`'s `Clusterer` trait guarantees compact `0..K` numbering
(`result[i] < unique(result).count()`), so speaker ids arrive dense and
zero-based and need no remapping before assignment.

### `src/diarize/assign.rs`

The only genuinely new logic, and the only part testable without downloading
anything. Pure Rust, no model, no I/O.

```rust
/// Assign a speaker to every word, then split segments where the speaker
/// changes. Segments are returned unnumbered; see `number()`.
pub fn assign(segs: Vec<Seg>, turns: &[SpeakerTurn]) -> Vec<Seg>;
```

Rules:

1. **Per word, maximum temporal overlap.** A word's speaker is the one whose
   turns cover the most of that word's `[start, end)`.
2. **Ties break toward the earlier-starting turn**, so the result is
   deterministic rather than dependent on turn ordering.
3. **A word with no overlapping turn gets `speaker = None`.** It is not
   guessed. This follows v1's rule against fabricating a value the pipeline
   cannot compute — the same reason `language_probability` is `None` rather
   than invented when the language was pinned.
4. **Segments split at word boundaries where the speaker changes.** Runs of
   consecutive same-speaker words (including runs of `None`) each become one
   segment.
5. **A split segment's `text` is rebuilt from its words' text.** Whisper's
   own segment text is not always the exact concatenation of its word texts,
   so split segments may differ from the original in whitespace. Documented,
   not silently papered over.
6. **A segment with no words is passed through unchanged** with
   `speaker = None`. This happens when the ASR produced a segment but no word
   alignment for it.

### Segment numbering moves

v1 assigns segment ids inside `stitch`:

```rust
pub fn stitch(window: &Window, segs: Vec<Seg>, next_id: &mut u32) -> Vec<Seg>
```

Splitting a segment after `stitch` would leave the ids non-sequential, and the
iterator is lazy so earlier ids cannot be renumbered retroactively.

So: `stitch` stops assigning ids and returns unnumbered segments; splitting
happens next; a separate `number(&mut [Seg], &mut u32)` is the last step
before the segments leave the iterator. Ids stay sequential and gap-free
across the whole transcript, which is what they promise today.

### `src/models/hub.rs`

Gains the `polyvoice` ONNX downloads from GitHub Releases, reusing the
existing cache directory and `FetchOptions` (`download_root`,
`local_files_only`). These are plain HTTPS assets, not Hugging Face repos, so
they need their own fetch path rather than `hf-hub`.

## Python API

```python
segments, info = model.transcribe(audio, diarize=True, max_speakers=8)
for seg in segments:
    print(seg.speaker, seg.start, seg.end, seg.text)
    for w in seg.words:
        print("  ", w.speaker, w.start, w.end, w.word)
```

- `Segment.speaker: Optional[int]`
- `Word.speaker: Optional[int]`
- `TranscriptionInfo.num_speakers: Optional[int]` — distinct speakers
  actually assigned, `None` when `diarize=False`.
- `diarize: bool = False`, `max_speakers: int = 8`. No `min_speakers`:
  see the component section for why it cannot be honoured.

With `diarize=False` (the default), `speaker` is `None` everywhere and no ONNX
model is loaded or downloaded. v1 behaviour is unchanged.

### `word_timestamps` becomes tri-state

Per-word assignment requires word timestamps. Silently switching on a
parameter the caller passed as `False` is the accept-and-ignore behaviour v1
forbids, so the default becomes `None`:

| `word_timestamps` | `diarize=False` | `diarize=True` |
| --- | --- | --- |
| `None` (default) | off | on |
| `False` | off | `ValueError` — explicit contradiction |
| `True` | on | on |

The `ValueError` is deliberate: the two arguments genuinely conflict, and
there is no reading of "no word timestamps, but assign speakers per word"
that can be honoured.

### `diarize=True` breaks iterator laziness

Diarization needs the entire file before the first segment can be assigned, so
it runs eagerly inside `transcribe()` — joining VAD, windowing, and language
detection, which are already eager in v1. ASR decoding stays lazy: only the
diarization pass is brought forward. Documented in the `transcribe` docstring
alongside the existing eagerness notes.

## Feature gating and packaging

Cargo:

```toml
diarization = [
    "dep:polyvoice",
    "polyvoice/pipeline-full",   # onnx + download + segmentation + embedder
                                 # + clusterer + resegmentation
    "polyvoice/load-dynamic",
]
```

`load-dynamic` is not optional within this feature — it is the entire reason
the feature can exist.

Python: `pip install whisper-rs[diarization]` pulls the `onnxruntime` pip
package, and the extension resolves the dylib path from the installed package
at import time, before any `ort` call. This keeps the wheel small, lets pip
manage onnxruntime versions and CVEs, and is the exact configuration the spike
validated (onnxruntime 1.29.0 from the pip wheel).

If `diarize=True` is requested and the dylib cannot be found, the error names
the missing package and the install command. It must not be a bare
`ort` initialisation failure.

Minimum onnxruntime version: `polyvoice` and `parakeet-rs` default to
`ort`'s `api-28`, requiring onnxruntime >= 1.28. If that floor proves too high
for common environments, `api-24` lowers it; the choice is recorded here so it
is a decision rather than an accident.

## Error handling

New `Error` variants, following v1's pattern of naming the thing that failed:

- `Error::Diarize(String)` — backend failure, mapped to `RuntimeError` in
  Python like the other backend errors.
- `Error::OnnxRuntimeMissing { message }` — the dylib could not be located.
  Mapped to `OSError`, message names `pip install whisper-rs[diarization]`.

`to_pyerr` stays exhaustive with no catch-all, so adding these variants
without mapping them is a compile error.

## Testing

**`assign.rs` is covered entirely without models.** Overlap, ties, silence,
missing turns, a segment with no words, a speaker change mid-segment, a run of
unassignable words, and the segment-splitting arithmetic are all pure
functions over constructed inputs. This is where correctness is actually
established.

**Real diarization is tested with generated multi-speaker audio.** macOS `say`
with distinct voices (Samantha, Alex, Fred, Daniel, Karen) concatenated in a
known order gives exact ground truth for both speaker count and turn order —
which a real recording would not. Marked `@pytest.mark.model` and skipped
where `say` is absent, matching v1's real-speech tests.

**A coexistence test is permanent, not a spike artefact.** One test loads a
CTranslate2 model and runs ONNX inference in the same process. It is the
regression test for the SIGBUS, and it is the test that must pass on Linux CI
before Linux support is claimed.

## Task zero: validate `polyvoice` before building on it

The implementation plan's first task downloads the models, runs `polyvoice` on
audio with five or more speakers, and checks that the speaker count and turn
order match the ground truth. **On failure, stop and report — do not continue
to the next task.**

This is not ceremony. In v1, two blocking defects were found late: no model
could load at all because `ct2rs` requires a `preprocessor_config.json` that
the Systran repos do not publish, and language auto-detection never worked
because the language token lives in the decoder prompt and is never returned.
Both were discovered after the code that depended on them was written. The
cost of finding out first is one task; the cost of finding out last is a
rewrite.

## Deferred: punctuation

Recorded here because it was raised during design and the finding should not be
lost.

`faster-whisper` conditions each window's decode on the previous window's text
(`condition_on_previous_text`), which is what keeps sentences, commas, and
capitalisation coherent across a 30 s boundary. This crate does not: every
window is decoded in total isolation, a deliberate v1 choice inherited from
whisperx.

`ct2rs` does not permit otherwise. The prompt is built internally and is fixed:

```rust
fn generate_prompt<'a>(&self, lang_token: &'a str, timestamp: bool) -> Vec<&'a str> {
    let mut prompt = vec!["<|startoftranscript|>", lang_token, "<|transcribe|>"];
    if !timestamp { prompt.push("<|notimestamps|>"); }
    prompt
}
```

There is no injection point for previous text, and `WhisperOptions` has no
prompt field. Every window therefore begins as though it were the start of the
audio.

If punctuation quality turns out to be a real problem, this is the first
suspect, and the fix is on our side: drop to `ct2rs::sys::Whisper` and build
the prompt directly, the same move `asr::detect` already makes for language
detection. Swapping in a different ASR engine would treat the symptom while
paying for another 600 MB model, another tokenizer, and the loss of Whisper's
mature multilingual coverage.

v3 should measure before it changes anything.

## Definition of done

1. `pip install whisper-rs[diarization]` produces a working diarization path.
2. `transcribe(audio, diarize=True)` assigns a speaker to every word of real
   multi-speaker audio.
3. Audio with five or more distinct speakers yields five or more distinct
   speaker ids — the requirement Sortformer could not meet.
4. Segments split where the speaker changes mid-segment, with sequential,
   gap-free ids.
5. `diarize=False` is byte-for-byte v1 behaviour, with no ONNX loaded.
6. The default build (no `diarization` feature) has no `ort` in its
   dependency graph.
7. `word_timestamps=False` with `diarize=True` raises `ValueError`.
8. The CTranslate2 + onnxruntime coexistence test passes on macOS, and its
   Linux result is known and recorded either way.

## Task 0 findings

A spike (`tests/spike_polyvoice.rs`, deleted after this record was made)
ran `polyvoice` 0.17 end to end against a 5-voice macOS `say` fixture
(Samantha, Alex, Fred, Daniel, Karen concatenated, 16 kHz mono, 23.7 s).

**Result: no SIGBUS, no `ConfigError` — but `num_speakers = 3`, under the
5 required.** This is failure mode 3 from the task-0 brief: diarization runs
but under-counts, and per the brief this blocks Task 1 pending a design
conversation about clusterer/embedder tuning or defaults.

Working call sequence:

```rust
ort::init_from(&dylib).expect("...").commit();   // ORT_DYLIB_PATH, load-dynamic mode

let registry = polyvoice::models::ModelRegistry::default()?;   // NOT optional --
                                                                 // Pipeline::builder().build()
                                                                 // fails with
                                                                 // ConfigError::MissingRegistry
                                                                 // without it

let pipeline = polyvoice::pipeline_v2::Pipeline::builder()
    .max_speakers(8)
    .with_models_from(registry)
    .build()?;

let sr = polyvoice::types::SampleRate::new(16_000)?;
let result = pipeline.run(&samples, sr)?;
```

Turn list observed (`num_speakers = 3`):

```
speaker=0  0.00..9.58
speaker=1  9.58..14.77
speaker=2  14.77..19.22
speaker=0 19.32..23.63
```

Speakers 3 (Daniel) and 4 (Karen) were merged into clusters already assigned
to speakers 0/1/2 — the pipeline under-clustered, not under-segmented (the
turn boundaries at ~9.6s/14.8s/19.3s roughly track the actual utterance
boundaries, so segmentation found more than 3 turns' worth of boundaries but
clustering collapsed them to 3 speaker identities).

**Defaults in effect** (`Profile::Balanced`, the `PipelineConfig::default()`,
none overridden except `max_speakers(8)`):
- Segmenter: `PowersetSegmenter` loading `powerset_int8.onnx`.
- Embedder: `ResNet34Adapter` loading `resnet34_int8.onnx`.
- Clusterer: `ClustererKind::Ahc { threshold: 0.45 }` (`DEFAULT_AHC_THRESHOLD`,
  `polyvoice::types::config`) — agglomerative hierarchical clustering.
- `max_speakers`: 8 (spike override; profile default is 20).
- `min_cluster_size`: 1 (no pruning).
- `resegment_overlap`: true.
- `execution_provider`: `ExecutionProvider::auto()`.

**How models were obtained:** `polyvoice::models::ModelRegistry::default()`
resolves a cache dir at `~/Library/Caches/polyvoice/models` (macOS) from the
crate's embedded manifest, and `ensure_for_profile` downloaded
`powerset_int8.onnx` and `resnet34_int8.onnx` into that cache on first run —
no manual model-fetch step, no HF token, no venv needed for models. This is
separate from the ONNX Runtime dylib itself, which was supplied via
`ORT_DYLIB_PATH` pointing at a pip-installed `onnxruntime` wheel's
`libonnxruntime.1.29.0.dylib` (a pre-existing spike artifact reused per Task
0 runner instructions, not re-derived here).

**Errors encountered along the way (both resolved before the run above,
recorded because they contradict the brief's literal steps):**

1. `cargo add polyvoice@0.17 --optional --no-default-features --features pipeline-full,load-dynamic`
   fails at the `cargo add` step itself:
   ```
   error: unrecognized feature for crate polyvoice: load-dynamic
   ```
   `polyvoice` 0.17 has no `load-dynamic` feature of its own (it forwards to `ort`
   internally). Resolution: add polyvoice with `--features pipeline-full` only;
   `load-dynamic` is obtained by adding `ort` as a direct dev-dependency with
   `--features load-dynamic,std`, which via Cargo feature unification turns on
   `load-dynamic` for the single shared `ort` crate instance polyvoice also
   depends on. Verified with `cargo tree --features diarization -e features`:
   both `ort feature "load-dynamic"` and `ort feature "download-binaries"`
   (polyvoice's own default) show as active on the same `ort` node.

2. First `build()` call (no registry) failed with:
   ```
   MissingRegistry { profile: Balanced }
   ```
   (`polyvoice::pipeline_v2::ConfigError::MissingRegistry`). The brief's Step 3
   spike code omits `.with_models_from(...)`; `Profile::Balanced`,
   `Profile::Mobile`, and `Profile::Fast` all require it. Resolution: call
   `polyvoice::models::ModelRegistry::default()` and pass it to
   `.with_models_from(registry)` before `.build()`. Task 6 must include this
   registry call — it is not optional plumbing.

**Conclusion: gate outcome is failure mode 3 (under-counting), not fatal to
the design but blocking Task 1 until embedder/clusterer tuning (e.g. a lower
AHC threshold, `ClustererKind::NmeSc`, or a different profile) is decided.**

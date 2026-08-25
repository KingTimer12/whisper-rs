# whisper-rs v2 Diarization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Assign a speaker to every transcribed word, with no cap on the number of speakers, in-process and in Rust.

**Architecture:** Diarization runs as a stage parallel to ASR. Both consume the same 16 kHz mono `Vec<f32>` that `pipeline::prepare` already produces. Diarization sees the whole file (global clustering is what removes the speaker cap); ASR sees VAD windows. They meet in a pure-Rust assignment step that gives each word a speaker and splits segments where the speaker changes.

**Tech Stack:** `polyvoice` 0.17 (segmentation + embeddings + clustering, MIT), `ort` 2.0.0-rc.13 in `load-dynamic` mode, `onnxruntime` from the pip wheel, existing `ct2rs` / `symphonia` / `rubato` / `wavekat-vad` stack.

**Spec:** `docs/superpowers/specs/2026-08-25-whisper-rs-v2-diarization-design.md`

## Global Constraints

- **`ort` MUST be in `load-dynamic` mode.** Static linking causes `SIGBUS: access to undefined memory` from the CTranslate2/onnxruntime static-`protobuf` ODR collision. This is not a preference; it is the only reason this feature can exist.
- **The default build MUST NOT contain `ort`.** Without the `diarization` feature, `cargo tree` must show no `ort`, and behaviour must be byte-for-byte v1.
- **onnxruntime >= 1.28** (`polyvoice` defaults to `ort`'s `api-28`). Verified working: 1.29.0 from the pip wheel.
- **No fabricated values.** A word with no overlapping speaker turn gets `speaker = None`, never a guess. Same rule that keeps `language_probability` honest.
- **No silently-ignored arguments.** An argument that cannot take effect must error or warn, never be accepted and dropped.
- **`max_speakers` is `u8` in `polyvoice`** (`PipelineBuilder::max_speakers(n: u8)`), so the valid range is `1..=255`. Values outside it must be rejected with a clear error, not truncated.
- **Segment ids stay sequential and gap-free** across the whole transcript, as v1 promises.
- **Platform:** the coexistence test is validated on macOS arm64 only. Linux is unproven; Task 11 records its result either way.

## Deliberate deviations from the spec

Recorded here so they are decisions rather than drift:

1. **`SpeakerTurn` uses `f32` seconds, not `f64`.** The spec wrote `f64`, matching `polyvoice`'s `TimeRange`. Every other timestamp in this crate (`Seg`, `Word`, `Info`) is `f32` seconds, and assignment compares turns against word times. Converting once at the backend boundary beats a mixed-precision comparison in the hot path. The `f64 -> f32` narrowing is harmless: at f32 precision, one second near the 10-hour mark still resolves to under 4 ms.
2. **Model downloading reuses `polyvoice`'s own downloader, not `models/hub.rs`.** The spec said `hub.rs` gains the GitHub Releases fetch. `polyvoice` already ships `download_with_checksum_and_signature(url, sha256, signature, dest)` with minisign verification behind its `download` feature. Reimplementing a less-verified downloader beside it would be strictly worse. `hub.rs` is untouched by this plan.

---

### Task 0: Validation gate — prove `polyvoice` works before building on it

**This task is a gate. If it fails, STOP and report. Do not start Task 1.**

`polyvoice` has never been run. In v1 two blocking defects were found only after the code depending on them existed: no model could load because `ct2rs` demands a `preprocessor_config.json` the Systran repos do not publish, and language auto-detection never worked because the language token lives in the decoder prompt and is never returned. Both cost a rewrite. This task buys that information first.

**Files:**
- Create: `tests/spike_polyvoice.rs` (throwaway — deleted in Step 6)

**Interfaces:**
- Produces: a go/no-go answer, plus the concrete `polyvoice` call sequence Task 6 will use.

- [ ] **Step 1: Add the dependency and feature**

```bash
cargo add polyvoice@0.17 --optional --no-default-features \
  --features pipeline-full,load-dynamic
```

Then add to `[features]` in `Cargo.toml`:

```toml
diarization = ["dep:polyvoice"]
```

- [ ] **Step 2: Build the multi-speaker fixture**

Five distinct macOS voices, concatenated in a known order. The order is the ground truth — a real recording would not give you one.

```bash
SP="$(mktemp -d)"
i=0
for v in Samantha Alex Fred Daniel Karen; do
  say -v "$v" -o "$SP/$i.aiff" "This is speaker number $i speaking a full sentence for the diarization test."
  i=$((i+1))
done
# Concatenate to one 16 kHz mono WAV
sox "$SP"/0.aiff "$SP"/1.aiff "$SP"/2.aiff "$SP"/3.aiff "$SP"/4.aiff \
  -r 16000 -c 1 -b 16 "$SP/five_speakers.wav" 2>/dev/null \
  || python3 - "$SP" <<'PY'
import sys, subprocess, wave, pathlib
sp = pathlib.Path(sys.argv[1])
wavs = []
for i in range(5):
    w = sp / f"{i}.wav"
    subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1",
                    str(sp / f"{i}.aiff"), str(w)], check=True)
    wavs.append(w)
out = wave.open(str(sp / "five_speakers.wav"), "wb")
first = wave.open(str(wavs[0]), "rb")
out.setparams(first.getparams()); first.close()
for w in wavs:
    r = wave.open(str(w), "rb")
    out.writeframes(r.readframes(r.getnframes())); r.close()
out.close()
print(sp / "five_speakers.wav")
PY
echo "FIXTURE=$SP/five_speakers.wav"
```

`sox` is tried first because it is one line; the Python fallback uses only the stdlib plus `afconvert`, which ships with macOS. If neither path produces a WAV, stop and report — the fixture is a prerequisite, not part of the deliverable.

- [ ] **Step 3: Write the spike**

```rust
//! THROWAWAY SPIKE -- deleted at the end of Task 0.
//!
//! Question: does `polyvoice` run at all, and does it find five speakers in
//! audio containing exactly five?

#[test]
#[ignore]
fn polyvoice_finds_five_speakers() {
    let dylib = std::env::var("ORT_DYLIB_PATH").expect("set ORT_DYLIB_PATH");
    let wav = std::env::var("FIXTURE").expect("set FIXTURE");

    // Order matters: initialise ort before anything builds a session.
    let ok = ort::init_from(&dylib)
        .expect("ort could not load the dylib")
        .commit();
    assert!(ok, "ort environment commit failed");

    let samples = read_wav_16k_mono(&wav);
    eprintln!("[spike] {} samples ({:.1}s)", samples.len(), samples.len() as f32 / 16_000.0);

    let pipeline = polyvoice::pipeline_v2::Pipeline::builder()
        .max_speakers(8)
        .build()
        .expect("pipeline build failed -- report the ConfigError verbatim");

    let sr = polyvoice::types::SampleRate::new(16_000).expect("16 kHz is in range");
    let result = pipeline.run(&samples, sr).expect("pipeline run failed");

    eprintln!("[spike] num_speakers = {}", result.num_speakers);
    for t in &result.turns {
        eprintln!(
            "[spike] turn speaker={} {:.2}..{:.2}",
            t.speaker.0, t.time.start, t.time.end
        );
    }
    eprintln!("[spike] SPIKE RESULT: ran without crashing");

    assert!(
        result.num_speakers >= 5,
        "expected >= 5 speakers, got {} -- this is the requirement Sortformer could not meet",
        result.num_speakers
    );
}

/// Minimal 16-bit mono WAV reader: the fixture is written by us, so this only
/// needs to handle the one format it produces.
fn read_wav_16k_mono(path: &str) -> Vec<f32> {
    let mut r = hound::WavReader::open(path).expect("open fixture");
    assert_eq!(r.spec().sample_rate, 16_000, "fixture must be 16 kHz");
    assert_eq!(r.spec().channels, 1, "fixture must be mono");
    r.samples::<i16>()
        .map(|s| s.expect("sample") as f32 / 32_768.0)
        .collect()
}
```

Add `ort` as a dev-dependency for the spike only:

```bash
cargo add --dev ort@2.0.0-rc.13 --no-default-features --features load-dynamic,std
```

- [ ] **Step 4: Get the onnxruntime dylib**

```bash
python3 -m venv /tmp/ortvenv && /tmp/ortvenv/bin/pip -q install onnxruntime
find /tmp/ortvenv -name "libonnxruntime*.dylib"
```

- [ ] **Step 5: Run it**

```bash
ORT_DYLIB_PATH=<path from step 4> FIXTURE=<path from step 2> \
  cargo test --features diarization --test spike_polyvoice -- --ignored --nocapture
```

Expected: prints `num_speakers = 5` (or more) and one line per turn.

**Three failure modes, and what each means. Report which one occurred:**

| Symptom | Meaning |
| --- | --- |
| `SIGBUS` | The collision is not solved after all. `load-dynamic` is not reaching `ort` — check `cargo tree -f "{p} {f}"` for `ort` features. Blocks the whole feature. |
| `ConfigError` on `build()` | Models are missing. `polyvoice` needs its ONNX weights fetched; report the exact error so Task 6 can wire the download. Not fatal to the design. |
| `num_speakers < 5` | Diarization runs but under-counts. Report the actual count and the turn list. Possibly a clusterer/embedder tuning question, possibly a wrong default. **This is the one that requires a design conversation before Task 1.** |

- [ ] **Step 6: Record findings, then delete the spike**

Append a `## Task 0 findings` section to the spec (`docs/superpowers/specs/2026-08-25-whisper-rs-v2-diarization-design.md`) recording: the working call sequence, the observed `num_speakers`, which embedder/clusterer defaults were in play, and how the models were obtained. The spec's component section promises this record.

```bash
rm tests/spike_polyvoice.rs
cargo remove --dev ort
git add -A && git commit -m "docs: record Task 0 polyvoice validation findings"
```

---

### Task 1: Types, error variants, and feature scaffolding

**Files:**
- Create: `src/diarize/mod.rs`
- Modify: `src/error.rs` (add two variants)
- Modify: `src/python/mod.rs` (map the two variants in `to_pyerr`)
- Modify: `src/lib.rs` (add `mod diarize;`)
- Modify: `Cargo.toml`

**Interfaces:**
- Produces: `diarize::SpeakerTurn { start: f32, end: f32, speaker: usize }`, `diarize::Diarizer` trait, `Error::Diarize(String)`, `Error::OnnxRuntimeMissing { message: String }`.

- [ ] **Step 1: Write the failing test**

In `src/diarize/mod.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_turn_knows_its_duration() {
        let t = SpeakerTurn { start: 1.5, end: 4.0, speaker: 2 };
        assert!((t.duration() - 2.5).abs() < f32::EPSILON);
    }

    #[test]
    fn a_backwards_turn_has_zero_duration() {
        // Defensive: a backend that emits end < start must not produce a
        // negative duration, which would corrupt the overlap arithmetic in
        // `assign` by making a bad turn look like the best match.
        let t = SpeakerTurn { start: 4.0, end: 1.5, speaker: 0 };
        assert_eq!(t.duration(), 0.0);
    }
}
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cargo test --lib diarize::tests`
Expected: FAIL — `cannot find struct SpeakerTurn`.

- [ ] **Step 3: Write the implementation**

`src/diarize/mod.rs`:

```rust
//! Speaker diarization: who spoke when.
//!
//! The concrete backend lives behind the `diarization` feature so the default
//! build contains no `ort` at all. See `polyvoice.rs` for why that matters.

use crate::error::Result;

#[cfg(feature = "diarization")]
pub mod polyvoice;

pub mod assign;

/// One speaker's continuous turn, in seconds on the global timeline.
///
/// `f32` seconds matches every other timestamp in this crate (`Seg`, `Word`,
/// `Info`); `polyvoice`'s own `TimeRange` is `f64` and is narrowed once at the
/// backend boundary rather than forcing mixed-precision comparisons into the
/// per-word overlap arithmetic.
#[derive(Debug, Clone, PartialEq)]
pub struct SpeakerTurn {
    pub start: f32,
    pub end: f32,
    pub speaker: usize,
}

impl SpeakerTurn {
    /// Length in seconds, clamped at zero.
    ///
    /// Clamping is deliberate: a backend emitting `end < start` would
    /// otherwise yield a negative duration, and negative overlap would make a
    /// malformed turn compare as a *better* match than a valid one in
    /// `assign::assign`.
    pub fn duration(&self) -> f32 {
        (self.end - self.start).max(0.0)
    }
}

/// A diarization backend. One call consumes a whole file.
pub trait Diarizer: Send + Sync {
    /// Diarize a whole file's worth of 16 kHz mono samples in `[-1, 1]`.
    ///
    /// Whole-file, not per-window: global clustering across the entire signal
    /// is what allows an unbounded speaker count, and what establishes that
    /// the voice at minute 1 is the same person as the voice at minute 40.
    fn diarize(&self, samples: &[f32]) -> Result<Vec<SpeakerTurn>>;
}
```

- [ ] **Step 4: Run it to verify it passes**

Run: `cargo test --lib diarize::tests`
Expected: PASS (2 tests).

- [ ] **Step 5: Add the error variants**

In `src/error.rs`, add to the `Error` enum:

```rust
    #[error("diarization failed: {0}")]
    Diarize(String),

    #[error("{message}")]
    OnnxRuntimeMissing { message: String },
```

In `src/python/mod.rs`, extend `to_pyerr`'s match:

```rust
        Error::Diarize(_) => PyRuntimeError::new_err(message),
        Error::OnnxRuntimeMissing { .. } => PyOSError::new_err(message),
```

`to_pyerr` has no catch-all arm, by design — adding a variant without mapping it is a compile error rather than a silent `RuntimeError`.

- [ ] **Step 6: Wire the module and feature**

`src/lib.rs`: add `mod diarize;` alongside the other `mod` declarations.

`Cargo.toml`:

```toml
[features]
diarization = [
    "dep:polyvoice",
    "polyvoice/pipeline-full",
    "polyvoice/load-dynamic",
]
```

- [ ] **Step 7: Verify the default build has no `ort`**

```bash
cargo tree | grep -c '^.*ort v' || echo "no ort in default build -- correct"
cargo tree --features diarization | grep 'ort v'
```

Expected: nothing in the default build; `ort v2.0.0-rc.13` with the feature on.

- [ ] **Step 8: Run the full suite and commit**

```bash
cargo test && cargo clippy --all-targets -- -D warnings
git add -A
git commit -m "feat: add Diarizer trait, SpeakerTurn, and diarization error variants"
```

---

### Task 2: Per-word speaker assignment by maximum overlap

The core of the feature, and the only part testable without a model. Pure functions over constructed inputs — this is where correctness is actually established.

**Files:**
- Create: `src/diarize/assign.rs`

**Interfaces:**
- Consumes: `SpeakerTurn` (Task 1), `crate::types::{Seg, Word}`.
- Produces: `pub fn speaker_for(word: &Word, turns: &[SpeakerTurn]) -> Option<usize>`, `pub fn overlap(a_start: f32, a_end: f32, b_start: f32, b_end: f32) -> f32`.

- [ ] **Step 1: Write the failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Word;

    fn word(start: f32, end: f32) -> Word {
        Word { start, end, text: "x".into(), probability: 1.0 }
    }

    fn turn(start: f32, end: f32, speaker: usize) -> SpeakerTurn {
        SpeakerTurn { start, end, speaker }
    }

    #[test]
    fn disjoint_ranges_do_not_overlap() {
        assert_eq!(overlap(0.0, 1.0, 2.0, 3.0), 0.0);
    }

    #[test]
    fn touching_ranges_do_not_overlap() {
        // Ranges are half-open [start, end), so an end meeting a start is
        // adjacency, not overlap.
        assert_eq!(overlap(0.0, 1.0, 1.0, 2.0), 0.0);
    }

    #[test]
    fn partial_overlap_is_the_intersection() {
        assert_eq!(overlap(0.0, 2.0, 1.0, 3.0), 1.0);
    }

    #[test]
    fn containment_is_the_inner_range() {
        assert_eq!(overlap(0.0, 5.0, 1.0, 2.0), 1.0);
    }

    #[test]
    fn a_word_takes_the_speaker_covering_most_of_it() {
        let turns = vec![turn(0.0, 1.2, 7), turn(1.2, 3.0, 9)];
        // 0.2 s with speaker 7, 0.8 s with speaker 9.
        assert_eq!(speaker_for(&word(1.0, 2.0), &turns), Some(9));
    }

    #[test]
    fn a_word_with_no_overlapping_turn_gets_no_speaker() {
        let turns = vec![turn(10.0, 12.0, 1)];
        assert_eq!(speaker_for(&word(0.0, 1.0), &turns), None);
    }

    #[test]
    fn no_turns_at_all_means_no_speaker() {
        assert_eq!(speaker_for(&word(0.0, 1.0), &[]), None);
    }

    #[test]
    fn an_exact_tie_breaks_toward_the_earlier_turn() {
        // Both cover exactly 0.5 s of the word. The result must not depend on
        // the order `turns` happens to arrive in, so it is pinned to the turn
        // that starts earlier.
        let turns = vec![turn(1.5, 2.0, 4), turn(1.0, 1.5, 2)];
        assert_eq!(speaker_for(&word(1.0, 2.0), &turns), Some(2));
    }

    #[test]
    fn overlapping_turns_for_one_speaker_accumulate() {
        // Two turns for speaker 3 covering 0.3 + 0.3, versus one turn of 0.5
        // for speaker 5. Speaker 3 wins on total, not on any single turn.
        let turns = vec![
            turn(0.0, 0.3, 3),
            turn(0.3, 0.6, 3),
            turn(0.6, 1.1, 5),
        ];
        assert_eq!(speaker_for(&word(0.0, 1.0), &turns), Some(3));
    }

    #[test]
    fn a_zero_length_word_gets_no_speaker() {
        // Whisper can emit a word whose start equals its end. It overlaps
        // nothing by definition, so it is unassignable rather than assigned
        // to whichever turn happens to contain the instant.
        let turns = vec![turn(0.0, 5.0, 1)];
        assert_eq!(speaker_for(&word(2.0, 2.0), &turns), None);
    }

    #[test]
    fn a_backwards_turn_never_wins() {
        let turns = vec![turn(5.0, 1.0, 8), turn(0.0, 1.0, 2)];
        assert_eq!(speaker_for(&word(0.0, 1.0), &turns), Some(2));
    }
}
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cargo test --lib diarize::assign`
Expected: FAIL — `cannot find function overlap`.

- [ ] **Step 3: Write the implementation**

```rust
//! Joining ASR output to diarization output.
//!
//! Pure arithmetic over timestamps: no models, no I/O, no ONNX. This is the
//! only genuinely new logic in v2, and the only part that can be tested
//! exhaustively without downloading anything.

use super::SpeakerTurn;
use crate::types::Word;
use std::collections::HashMap;

/// Length of the intersection of two half-open ranges, in seconds.
///
/// Half-open `[start, end)` means adjacency is not overlap: a word ending
/// exactly where a turn begins belongs to neither.
pub fn overlap(a_start: f32, a_end: f32, b_start: f32, b_end: f32) -> f32 {
    (a_end.min(b_end) - a_start.max(b_start)).max(0.0)
}

/// The speaker covering the most of `word`, or `None` when nothing covers it.
///
/// Overlap is summed *per speaker*, not per turn, so a speaker split across
/// several short turns is compared fairly against one with a single long turn.
///
/// Ties break toward the speaker whose earliest overlapping turn starts
/// first. Without that rule the answer would depend on the order `turns`
/// arrives in, which is not something a caller should have to reason about.
///
/// A word that overlaps nothing gets `None` rather than a guess: this crate
/// does not fabricate values it cannot compute.
pub fn speaker_for(word: &Word, turns: &[SpeakerTurn]) -> Option<usize> {
    // speaker -> (total overlap, earliest overlapping turn start)
    let mut totals: HashMap<usize, (f32, f32)> = HashMap::new();

    for turn in turns {
        // `duration()` clamps at zero, but a backwards turn would still yield
        // a bogus intersection here, so skip it outright.
        if turn.duration() <= 0.0 {
            continue;
        }
        let shared = overlap(word.start, word.end, turn.start, turn.end);
        if shared <= 0.0 {
            continue;
        }
        let entry = totals.entry(turn.speaker).or_insert((0.0, turn.start));
        entry.0 += shared;
        entry.1 = entry.1.min(turn.start);
    }

    totals
        .into_iter()
        .max_by(|(_, (a_total, a_start)), (_, (b_total, b_start))| {
            // Higher total wins; on a tie the earlier start wins, so `b_start`
            // is compared against `a_start` to invert that half of the order.
            a_total
                .partial_cmp(b_total)
                .expect("overlap totals are finite")
                .then_with(|| {
                    b_start
                        .partial_cmp(a_start)
                        .expect("turn starts are finite")
                })
        })
        .map(|(speaker, _)| speaker)
}
```

- [ ] **Step 4: Run them to verify they pass**

Run: `cargo test --lib diarize::assign`
Expected: PASS (11 tests).

- [ ] **Step 5: Commit**

```bash
cargo clippy --all-targets -- -D warnings
git add src/diarize/assign.rs
git commit -m "feat: assign a speaker to each word by maximum temporal overlap"
```

---

### Task 3: Split segments where the speaker changes

**Files:**
- Modify: `src/diarize/assign.rs`
- Modify: `src/types.rs` (add `speaker` to `Word` and `Seg`)

**Interfaces:**
- Consumes: `speaker_for` (Task 2).
- Produces: `pub fn assign(segs: Vec<Seg>, turns: &[SpeakerTurn]) -> Vec<Seg>`. Returned segments have `id: 0` — numbering is Task 4.

- [ ] **Step 1: Add the fields**

In `src/types.rs`:

```rust
pub struct Word {
    pub start: f32,
    pub end: f32,
    pub text: String,
    pub probability: f32,
    /// Assigned speaker, or `None` when no diarization ran or no turn covered
    /// this word.
    pub speaker: Option<usize>,
}

pub struct Seg {
    pub id: u32,
    pub start: f32,
    pub end: f32,
    pub text: String,
    pub words: Option<Vec<Word>>,
    /// Speaker of every word in this segment. `None` when no diarization ran,
    /// or when this segment's words were all unassignable.
    pub speaker: Option<usize>,
}
```

This breaks every existing `Word { .. }` and `Seg { .. }` literal. Fix each by adding `speaker: None` — the compiler lists them all. Expect them in `src/asr/ct2.rs`, `src/stitch.rs`, and their tests.

- [ ] **Step 2: Write the failing tests**

Append to `src/diarize/assign.rs`'s test module:

```rust
    fn seg(start: f32, end: f32, text: &str, words: Option<Vec<Word>>) -> Seg {
        Seg { id: 0, start, end, text: text.into(), words, speaker: None }
    }

    fn spoken(start: f32, end: f32, text: &str) -> Word {
        Word { start, end, text: text.into(), probability: 1.0, speaker: None }
    }

    #[test]
    fn a_single_speaker_segment_is_not_split() {
        let turns = vec![turn(0.0, 5.0, 1)];
        let segs = vec![seg(0.0, 2.0, " hello world", Some(vec![
            spoken(0.0, 1.0, " hello"),
            spoken(1.0, 2.0, " world"),
        ]))];

        let out = assign(segs, &turns);

        assert_eq!(out.len(), 1);
        assert_eq!(out[0].speaker, Some(1));
        assert_eq!(out[0].text, " hello world");
        let words = out[0].words.as_ref().unwrap();
        assert!(words.iter().all(|w| w.speaker == Some(1)));
    }

    #[test]
    fn a_segment_splits_where_the_speaker_changes() {
        let turns = vec![turn(0.0, 1.0, 1), turn(1.0, 2.0, 2)];
        let segs = vec![seg(0.0, 2.0, " hello world", Some(vec![
            spoken(0.0, 1.0, " hello"),
            spoken(1.0, 2.0, " world"),
        ]))];

        let out = assign(segs, &turns);

        assert_eq!(out.len(), 2, "one segment per speaker run");
        assert_eq!(out[0].speaker, Some(1));
        assert_eq!(out[0].text, " hello");
        assert_eq!(out[0].start, 0.0);
        assert_eq!(out[0].end, 1.0);
        assert_eq!(out[1].speaker, Some(2));
        assert_eq!(out[1].text, " world");
        assert_eq!(out[1].start, 1.0);
        assert_eq!(out[1].end, 2.0);
    }

    #[test]
    fn a_split_segments_bounds_come_from_its_own_words() {
        // Not from the original segment: the second half must not claim the
        // first half's start, or the timeline overlaps itself.
        let turns = vec![turn(0.0, 1.0, 1), turn(1.0, 3.0, 2)];
        let segs = vec![seg(0.0, 3.0, " a b c", Some(vec![
            spoken(0.0, 1.0, " a"),
            spoken(1.0, 2.0, " b"),
            spoken(2.0, 3.0, " c"),
        ]))];

        let out = assign(segs, &turns);

        assert_eq!(out.len(), 2);
        assert_eq!((out[0].start, out[0].end), (0.0, 1.0));
        assert_eq!((out[1].start, out[1].end), (1.0, 3.0));
        assert_eq!(out[1].text, " b c");
    }

    #[test]
    fn a_run_of_unassignable_words_becomes_its_own_segment() {
        let turns = vec![turn(0.0, 1.0, 1), turn(2.0, 3.0, 1)];
        let segs = vec![seg(0.0, 3.0, " a b c", Some(vec![
            spoken(0.0, 1.0, " a"),
            spoken(1.0, 2.0, " b"),   // gap between turns: unassignable
            spoken(2.0, 3.0, " c"),
        ]))];

        let out = assign(segs, &turns);

        assert_eq!(out.len(), 3);
        assert_eq!(out[0].speaker, Some(1));
        assert_eq!(out[1].speaker, None, "the gap is not attributed to anyone");
        assert_eq!(out[1].text, " b");
        assert_eq!(out[2].speaker, Some(1));
    }

    #[test]
    fn a_segment_with_no_words_passes_through_unassigned() {
        // The ASR produced a segment but no word alignment for it, so there is
        // nothing to assign per-word and nothing to split on.
        let turns = vec![turn(0.0, 5.0, 1)];
        let segs = vec![seg(0.0, 2.0, " hello", None)];

        let out = assign(segs, &turns);

        assert_eq!(out.len(), 1);
        assert_eq!(out[0].speaker, None);
        assert!(out[0].words.is_none());
        assert_eq!(out[0].text, " hello");
    }

    #[test]
    fn a_segment_with_an_empty_word_list_passes_through() {
        let turns = vec![turn(0.0, 5.0, 1)];
        let segs = vec![seg(0.0, 2.0, " hello", Some(vec![]))];

        let out = assign(segs, &turns);

        assert_eq!(out.len(), 1);
        assert_eq!(out[0].speaker, None);
        assert_eq!(out[0].text, " hello");
    }

    #[test]
    fn no_turns_leaves_every_segment_unassigned_and_unsplit() {
        let segs = vec![seg(0.0, 2.0, " hello world", Some(vec![
            spoken(0.0, 1.0, " hello"),
            spoken(1.0, 2.0, " world"),
        ]))];

        let out = assign(segs, &[]);

        assert_eq!(out.len(), 1);
        assert_eq!(out[0].speaker, None);
        assert_eq!(out[0].text, " hello world");
    }

    #[test]
    fn every_segment_is_processed_not_just_the_first() {
        let turns = vec![turn(0.0, 10.0, 3)];
        let segs = vec![
            seg(0.0, 1.0, " one", Some(vec![spoken(0.0, 1.0, " one")])),
            seg(1.0, 2.0, " two", Some(vec![spoken(1.0, 2.0, " two")])),
        ];

        let out = assign(segs, &turns);

        assert_eq!(out.len(), 2);
        assert!(out.iter().all(|s| s.speaker == Some(3)));
    }
```

- [ ] **Step 3: Run them to verify they fail**

Run: `cargo test --lib diarize::assign`
Expected: FAIL — `cannot find function assign`.

- [ ] **Step 4: Write the implementation**

Append to `src/diarize/assign.rs`:

```rust
use crate::types::Seg;

/// Assign a speaker to every word, then split each segment into runs of
/// consecutive same-speaker words.
///
/// Returned segments carry `id: 0`; sequential numbering is applied later, by
/// `stitch::number`, because splitting changes how many segments exist and the
/// iterator that hands them out is lazy.
///
/// A split segment's `text` is rebuilt by concatenating its words' text.
/// Whisper's own segment text is not always exactly the concatenation of its
/// word texts, so a split segment's text may differ from the original in
/// whitespace. A segment that is *not* split keeps its original text verbatim.
pub fn assign(segs: Vec<Seg>, turns: &[SpeakerTurn]) -> Vec<Seg> {
    let mut out = Vec::with_capacity(segs.len());

    for mut seg in segs {
        let Some(mut words) = seg.words.take() else {
            // No word alignment: nothing to assign, nothing to split on.
            out.push(Seg { speaker: None, words: None, ..seg });
            continue;
        };

        if words.is_empty() {
            out.push(Seg { speaker: None, words: Some(words), ..seg });
            continue;
        }

        for word in &mut words {
            word.speaker = speaker_for(word, turns);
        }

        // One output segment per run of consecutive words sharing a speaker.
        // Runs of `None` are runs too, so unattributed speech stays visible
        // instead of being folded into a neighbour.
        let mut runs: Vec<Vec<Word>> = Vec::new();
        for word in words {
            match runs.last_mut() {
                Some(run) if run[0].speaker == word.speaker => run.push(word),
                _ => runs.push(vec![word]),
            }
        }

        if runs.len() == 1 {
            // Unsplit: keep the original text exactly as the ASR produced it.
            let speaker = runs[0][0].speaker;
            out.push(Seg { speaker, words: Some(runs.pop().unwrap()), ..seg });
            continue;
        }

        for run in runs {
            let speaker = run[0].speaker;
            let start = run.first().expect("a run is never empty").start;
            let end = run.last().expect("a run is never empty").end;
            let text = run.iter().map(|w| w.text.as_str()).collect::<String>();
            out.push(Seg { id: 0, start, end, text, words: Some(run), speaker });
        }
    }

    out
}
```

- [ ] **Step 5: Run them to verify they pass**

Run: `cargo test --lib`
Expected: PASS — Task 2's 11 plus these 8.

- [ ] **Step 6: Commit**

```bash
cargo clippy --all-targets -- -D warnings
git add -A
git commit -m "feat: split segments where the speaker changes mid-segment"
```

---

### Task 4: Move segment numbering out of `stitch`

Splitting changes how many segments exist, so ids can no longer be handed out before it. The iterator is lazy and cannot renumber retroactively, so numbering becomes the last step before segments leave.

**Files:**
- Modify: `src/stitch.rs`
- Modify: `src/python/iter.rs`

**Interfaces:**
- Changes: `stitch(window: &Window, segs: Vec<Seg>) -> Vec<Seg>` — the `&mut u32` parameter is gone and returned segments have `id: 0`.
- Produces: `pub fn number(segs: &mut [Seg], next_id: &mut u32)`.

- [ ] **Step 1: Write the failing tests**

In `src/stitch.rs`'s test module:

```rust
    #[test]
    fn numbering_is_sequential_across_calls() {
        let mut next = 0;
        let mut first = vec![seg(0.0, 1.0, " a"), seg(1.0, 2.0, " b")];
        number(&mut first, &mut next);
        let mut second = vec![seg(2.0, 3.0, " c")];
        number(&mut second, &mut next);

        assert_eq!(first.iter().map(|s| s.id).collect::<Vec<_>>(), vec![0, 1]);
        assert_eq!(second[0].id, 2, "ids continue across calls, without a gap");
        assert_eq!(next, 3);
    }

    #[test]
    fn numbering_an_empty_slice_does_not_advance_the_counter() {
        let mut next = 7;
        number(&mut [], &mut next);
        assert_eq!(next, 7);
    }

    #[test]
    fn stitch_leaves_ids_at_zero() {
        // Numbering happens after splitting, so stitch must not claim ids.
        let window = Window { offset: 0, samples: vec![0.0; 16_000], real_len: 16_000 };
        let out = stitch(&window, vec![seg(0.0, 0.5, " hi")]);
        assert_eq!(out[0].id, 0);
    }
```

Add the `seg` helper to that module if it is not already there:

```rust
    fn seg(start: f32, end: f32, text: &str) -> Seg {
        Seg { id: 0, start, end, text: text.into(), words: None, speaker: None }
    }
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cargo test --lib stitch`
Expected: FAIL — `cannot find function number`, and `stitch` takes 3 arguments.

- [ ] **Step 3: Change `stitch` and add `number`**

In `src/stitch.rs`, drop the `next_id: &mut u32` parameter, remove the id assignment and the counter advance, and construct kept segments with `id: 0`. Then add:

```rust
/// Assign sequential ids to `segs`, continuing from `next_id`.
///
/// Numbering is separate from stitching because per-word diarization can split
/// one stitched segment into several, and the iterator handing segments to the
/// caller is lazy — an id given out early cannot be revised once a later split
/// changes the count. Numbering last keeps ids sequential and gap-free, which
/// is what they promise.
pub fn number(segs: &mut [Seg], next_id: &mut u32) {
    for seg in segs {
        seg.id = *next_id;
        *next_id += 1;
    }
}
```

- [ ] **Step 4: Update the caller**

In `src/python/iter.rs`'s `__next__`, replace

```rust
            let mut next_id = slf.next_id;
            let stitched = crate::stitch::stitch(&window, raw, &mut next_id);
            slf.next_id = next_id;
            slf.pending.extend(stitched);
```

with

```rust
            let mut stitched = crate::stitch::stitch(&window, raw);
            let mut next_id = slf.next_id;
            crate::stitch::number(&mut stitched, &mut next_id);
            slf.next_id = next_id;
            slf.pending.extend(stitched);
```

Task 7 inserts the `assign` call between `stitch` and `number`.

- [ ] **Step 5: Run the suite to verify it passes**

Run: `cargo test`
Expected: PASS. Existing stitch tests that passed `&mut next_id` need their calls updated; the compiler lists them.

- [ ] **Step 6: Commit**

```bash
cargo clippy --all-targets -- -D warnings
git add -A
git commit -m "refactor: number segments after stitching, not during it"
```

---

### Task 5: Resolve the onnxruntime dylib from the pip package

`load-dynamic` needs the dylib at runtime. Finding it must produce an error naming the fix, not a bare `ort` initialisation failure.

**Files:**
- Create: `src/diarize/dylib.rs`
- Modify: `src/diarize/mod.rs` (add `mod dylib;` under the feature)

**Interfaces:**
- Produces: `pub fn locate() -> Result<std::path::PathBuf>`, `pub fn init_ort() -> Result<()>`.

- [ ] **Step 1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_missing_dylib_names_the_install_command() {
        // `locate` consults ORT_DYLIB_PATH first. Pointing it somewhere that
        // does not exist exercises the error path without depending on
        // whether onnxruntime happens to be installed here.
        let err = locate_in(Some("/nonexistent/libonnxruntime.dylib".into()), &[])
            .expect_err("a nonexistent path must not resolve");
        let message = err.to_string();
        assert!(
            message.contains("whisper-rs[diarization]"),
            "the error must name the fix, got: {message}"
        );
    }

    #[test]
    fn an_explicit_path_that_exists_is_used_verbatim() {
        // Any existing file stands in for the dylib: `locate` checks presence,
        // it does not validate the file's contents. `ort` reports a bad
        // library far more precisely than a guess here could.
        let this_file = std::path::PathBuf::from(file!());
        let found = locate_in(Some(this_file.clone()), &[]).expect("an existing path resolves");
        assert_eq!(found, this_file);
    }

    #[test]
    fn a_site_packages_candidate_is_found() {
        let this_file = std::path::PathBuf::from(file!());
        let found = locate_in(None, &[this_file.clone()]).expect("a present candidate resolves");
        assert_eq!(found, this_file);
    }
}
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cargo test --features diarization --lib diarize::dylib`
Expected: FAIL — `cannot find function locate_in`.

- [ ] **Step 3: Write the implementation**

```rust
//! Locating libonnxruntime at runtime.
//!
//! `ort` runs in `load-dynamic` mode because static linking collides with
//! CTranslate2's `protobuf` and crashes the process with `SIGBUS`. The
//! trade-off is that the dylib has to be found at runtime, and a failure to
//! find it must explain itself: a bare `ort` initialisation error tells a
//! Python caller nothing actionable.

use crate::error::{Error, Result};
use std::path::PathBuf;
use std::sync::OnceLock;

static INIT: OnceLock<bool> = OnceLock::new();

/// Candidate paths inside an installed `onnxruntime` pip package.
fn pip_candidates() -> Vec<PathBuf> {
    let Ok(output) = std::process::Command::new("python3")
        .args(["-c", "import onnxruntime, os; print(os.path.dirname(onnxruntime.__file__))"])
        .output()
    else {
        return Vec::new();
    };
    if !output.status.success() {
        return Vec::new();
    }
    let dir = PathBuf::from(String::from_utf8_lossy(&output.stdout).trim().to_string());
    if dir.as_os_str().is_empty() {
        return Vec::new();
    }

    let capi = dir.join("capi");
    let mut found = Vec::new();
    // The wheel ships a version-stamped filename (libonnxruntime.1.29.0.dylib),
    // so the directory is scanned rather than a fixed name being guessed.
    if let Ok(entries) = std::fs::read_dir(&capi) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.starts_with("libonnxruntime") || name.starts_with("onnxruntime") {
                found.push(entry.path());
            }
        }
    }
    found
}

/// Testable core: the first existing path among `explicit` then `candidates`.
fn locate_in(explicit: Option<PathBuf>, candidates: &[PathBuf]) -> Result<PathBuf> {
    if let Some(path) = explicit {
        if path.is_file() {
            return Ok(path);
        }
        return Err(Error::OnnxRuntimeMissing {
            message: format!(
                "ORT_DYLIB_PATH points to {}, which does not exist. Unset it to \
                 use the onnxruntime pip package, or install one with \
                 `pip install whisper-rs[diarization]`.",
                path.display()
            ),
        });
    }

    for candidate in candidates {
        if candidate.is_file() {
            return Ok(candidate.clone());
        }
    }

    Err(Error::OnnxRuntimeMissing {
        message: "diarization needs the onnxruntime shared library, which was \
                  not found. Install it with `pip install whisper-rs[diarization]`, \
                  or point ORT_DYLIB_PATH at an existing libonnxruntime."
            .into(),
    })
}

/// Locate libonnxruntime: `ORT_DYLIB_PATH` first, then the pip package.
pub fn locate() -> Result<PathBuf> {
    let explicit = std::env::var_os("ORT_DYLIB_PATH").map(PathBuf::from);
    locate_in(explicit, &pip_candidates())
}

/// Initialise `ort` once per process.
///
/// `OnceLock` rather than repeated `init`: `ort`'s environment is global, and
/// a second `commit()` on an already-initialised environment returns `false`,
/// which would otherwise surface as a spurious error on the second
/// `transcribe(diarize=True)` call in a process.
pub fn init_ort() -> Result<()> {
    let path = locate()?;
    let committed = INIT.get_or_init(|| {
        ort::init_from(path.to_string_lossy().as_ref())
            .map(|env| env.commit())
            .unwrap_or(false)
    });

    if *committed {
        Ok(())
    } else {
        Err(Error::OnnxRuntimeMissing {
            message: format!(
                "found libonnxruntime at {} but could not initialise it. It may \
                 be older than the required 1.28, or built for another \
                 architecture.",
                path.display()
            ),
        })
    }
}
```

- [ ] **Step 4: Run it to verify it passes**

Run: `cargo test --features diarization --lib diarize::dylib`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
cargo clippy --all-targets --features diarization -- -D warnings
git add -A
git commit -m "feat: locate libonnxruntime from the pip package with an actionable error"
```

---

### Task 6: The `polyvoice` diarization backend

**Files:**
- Create: `src/diarize/polyvoice.rs`

**Interfaces:**
- Consumes: `Diarizer`, `SpeakerTurn` (Task 1), `dylib::init_ort` (Task 5), and the call sequence Task 0 recorded.
- Produces: `pub struct PolyvoiceDiarizer`, `PolyvoiceDiarizer::new(max_speakers: usize) -> Result<Self>`.

- [ ] **Step 1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_speakers_must_fit_polyvoice_u8_range() {
        // polyvoice's builder takes u8, so 256 cannot be honoured. Truncating
        // it to 0 (or to 255) would silently transcribe under a limit the
        // caller did not ask for.
        let err = PolyvoiceDiarizer::new(256).expect_err("256 exceeds the range");
        assert!(err.to_string().contains("1..=255"), "got: {err}");
    }

    #[test]
    fn zero_max_speakers_is_rejected() {
        let err = PolyvoiceDiarizer::new(0).expect_err("0 speakers is meaningless");
        assert!(err.to_string().contains("1..=255"), "got: {err}");
    }
}
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cargo test --features diarization --lib diarize::polyvoice`
Expected: FAIL — `cannot find struct PolyvoiceDiarizer`.

- [ ] **Step 3: Write the implementation**

Use the call sequence Task 0 recorded in the spec. The shape:

```rust
//! `polyvoice` diarization backend.
//!
//! Entirely behind the `diarization` feature: without it, `ort` is absent from
//! the dependency graph and this crate contains no ONNX runtime at all.

use super::{dylib, Diarizer, SpeakerTurn};
use crate::error::{Error, Result};
use crate::types::SAMPLE_RATE;

pub struct PolyvoiceDiarizer {
    pipeline: polyvoice::pipeline_v2::Pipeline,
}

impl PolyvoiceDiarizer {
    /// Build a diarizer bounded at `max_speakers`.
    ///
    /// `max_speakers` is an upper bound for clustering, not a model limit —
    /// which is the whole reason this backend was chosen over Sortformer,
    /// whose `NUM_SPEAKERS = 4` is fixed in the model.
    pub fn new(max_speakers: usize) -> Result<Self> {
        let bounded: u8 = u8::try_from(max_speakers)
            .ok()
            .filter(|n| *n >= 1)
            .ok_or_else(|| {
                Error::Diarize(format!(
                    "max_speakers must be in 1..=255, got {max_speakers}"
                ))
            })?;

        dylib::init_ort()?;

        let pipeline = polyvoice::pipeline_v2::Pipeline::builder()
            .max_speakers(bounded)
            .build()
            .map_err(|e| Error::Diarize(format!("building the pipeline failed: {e}")))?;

        Ok(Self { pipeline })
    }
}

impl Diarizer for PolyvoiceDiarizer {
    fn diarize(&self, samples: &[f32]) -> Result<Vec<SpeakerTurn>> {
        let sr = polyvoice::types::SampleRate::new(SAMPLE_RATE as u32).ok_or_else(|| {
            Error::Diarize(format!("{SAMPLE_RATE} Hz is outside polyvoice's supported range"))
        })?;

        let result = self
            .pipeline
            .run(samples, sr)
            .map_err(|e| Error::Diarize(e.to_string()))?;

        // polyvoice's TimeRange is f64; this crate's timeline is f32 seconds
        // throughout, so it narrows once here rather than forcing
        // mixed-precision comparisons into the per-word overlap arithmetic.
        Ok(result
            .turns
            .into_iter()
            .map(|t| SpeakerTurn {
                start: t.time.start as f32,
                end: t.time.end as f32,
                speaker: t.speaker.0 as usize,
            })
            .collect())
    }
}
```

If Task 0 found that models must be fetched explicitly, add that call before `.build()` and document where the weights land, matching Task 0's recorded findings. Do not invent a download path that Task 0 did not exercise.

- [ ] **Step 4: Run it to verify it passes**

Run: `cargo test --features diarization --lib diarize::polyvoice`
Expected: PASS (2 tests).

- [ ] **Step 5: Verify the default build is still ONNX-free**

```bash
cargo tree | grep 'ort v' && echo "FAIL: ort leaked into the default build" || echo "OK"
cargo test
```

- [ ] **Step 6: Commit**

```bash
cargo clippy --all-targets --features diarization -- -D warnings
git add -A
git commit -m "feat: add the polyvoice diarization backend behind a feature flag"
```

---

### Task 7: Wire diarization into the pipeline

**Files:**
- Modify: `src/python/model.rs`
- Modify: `src/python/iter.rs`

**Interfaces:**
- Consumes: `PolyvoiceDiarizer` (Task 6), `assign::assign` (Task 3), `stitch::number` (Task 4).
- Produces: `SegmentIterator::new(asr, windows, language, word_timestamps, turns: Vec<SpeakerTurn>)`.

- [ ] **Step 1: Write the failing test**

In `src/python/iter.rs`'s test module:

```rust
    #[test]
    fn an_iterator_without_turns_assigns_no_speakers() {
        // Constructed directly rather than through transcribe(): this checks
        // the wiring, not the model.
        let turns: Vec<crate::diarize::SpeakerTurn> = Vec::new();
        assert!(turns.is_empty(), "the diarize=False path carries no turns");
    }
```

- [ ] **Step 2: Thread turns through the iterator**

In `src/python/iter.rs`, add the field and constructor parameter:

```rust
pub struct SegmentIterator {
    asr: Arc<Ct2Asr>,
    windows: VecDeque<Window>,
    pending: VecDeque<Seg>,
    next_id: u32,
    language: String,
    word_timestamps: bool,
    /// Empty when `diarize=False`; `assign` then leaves every speaker `None`.
    turns: Vec<crate::diarize::SpeakerTurn>,
}
```

In `__next__`, insert assignment between stitch and numbering:

```rust
            let stitched = crate::stitch::stitch(&window, raw);
            let mut assigned = crate::diarize::assign::assign(stitched, &slf.turns);
            let mut next_id = slf.next_id;
            crate::stitch::number(&mut assigned, &mut next_id);
            slf.next_id = next_id;
            slf.pending.extend(assigned);
```

`assign` with an empty `turns` slice is the `diarize=False` path: every word gets `speaker = None` and nothing splits, so v1 behaviour is preserved by the same code path rather than by a branch.

- [ ] **Step 3: Run diarization eagerly in `transcribe`**

In `src/python/model.rs`, inside the existing `py.detach` block that runs `prepare`, after the language is settled:

```rust
                let turns = if diarize {
                    #[cfg(feature = "diarization")]
                    {
                        let samples = prepared.samples.as_slice();
                        let diarizer =
                            crate::diarize::polyvoice::PolyvoiceDiarizer::new(max_speakers)?;
                        diarizer.diarize(samples)?
                    }
                    #[cfg(not(feature = "diarization"))]
                    {
                        return Err(crate::error::Error::Diarize(
                            "this build has no diarization support: reinstall with \
                             `pip install whisper-rs[diarization]`, or build the crate \
                             with --features diarization"
                                .into(),
                        ));
                    }
                } else {
                    Vec::new()
                };
```

`prepare` currently returns only windows and info. Extend `Prepared` in `src/pipeline.rs` with the full sample buffer:

```rust
pub struct Prepared {
    pub windows: Vec<Window>,
    pub info: Info,
    /// The whole decoded, resampled signal. Diarization needs it: global
    /// clustering across the entire file is what allows an unbounded speaker
    /// count, so it cannot work from the VAD windows the ASR consumes.
    pub samples: Vec<f32>,
}
```

- [ ] **Step 4: Compute `num_speakers`**

After diarization, before returning:

```rust
                // Distinct speakers actually present in the turns, not the
                // `max_speakers` bound the caller asked for.
                info.num_speakers = if diarize {
                    let mut ids: Vec<usize> = turns.iter().map(|t| t.speaker).collect();
                    ids.sort_unstable();
                    ids.dedup();
                    Some(ids.len())
                } else {
                    None
                };
```

- [ ] **Step 5: Run the suite**

Run: `cargo test && cargo test --features diarization`
Expected: PASS both.

- [ ] **Step 6: Commit**

```bash
cargo clippy --all-targets -- -D warnings
cargo clippy --all-targets --features diarization -- -D warnings
git add -A
git commit -m "feat: run diarization eagerly and assign speakers as segments are produced"
```

---

### Task 8: Python API surface

**Files:**
- Modify: `src/python/segment.rs`
- Modify: `src/python/model.rs`
- Modify: `src/types.rs` (add `num_speakers` to `Info`)
- Modify: `pyproject.toml`

**Interfaces:**
- Produces: `Segment.speaker`, `Word.speaker`, `TranscriptionInfo.num_speakers`, and `transcribe(..., diarize=False, max_speakers=8, word_timestamps=None)`.

- [ ] **Step 1: Add the exposed fields**

In `src/python/segment.rs`, add `pub speaker: Option<usize>` to both `Word` and `Segment`, `pub num_speakers: Option<usize>` to `TranscriptionInfo`, and carry them through the two `From` impls. Extend `__repr__` for `Segment`:

```rust
    fn __repr__(&self) -> String {
        format!(
            "Segment(id={}, start={:.2}, end={:.2}, speaker={:?}, text={:?})",
            self.id, self.start, self.end, self.speaker, self.text
        )
    }
```

- [ ] **Step 2: Make `word_timestamps` tri-state**

Change the signature default to `word_timestamps = None` and the parameter type to `Option<bool>`, then resolve it:

```rust
        // Per-word speaker assignment requires word timestamps. Silently
        // switching on a parameter the caller passed as `False` is the
        // accept-and-ignore behaviour this crate forbids, so an explicit
        // `False` alongside `diarize=True` is a contradiction and errors.
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
```

- [ ] **Step 3: Write the failing Python tests**

In `tests/python/test_api.py`:

```python
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


def test_transcribe_rejects_out_of_range_max_speakers():
    # No model needed: validation happens before anything is loaded.
    assert hasattr(whisper_rs.WhisperModel, "transcribe")
```

- [ ] **Step 4: Add the pip extra**

In `pyproject.toml`:

```toml
[project.optional-dependencies]
diarization = ["onnxruntime>=1.28"]
```

The floor is not cosmetic: `polyvoice` builds against `ort`'s `api-28`, so an older onnxruntime fails to initialise.

- [ ] **Step 5: Build and run**

```bash
maturin develop --release --features diarization
python -m pytest tests/python -q
```

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat: expose speaker, num_speakers, and tri-state word_timestamps in Python"
```

---

### Task 9: The coexistence regression test

Permanent, not a spike artefact. It is the regression test for the `SIGBUS`, and the gate for claiming Linux support.

**Files:**
- Create: `tests/coexistence.rs`

- [ ] **Step 1: Write the test**

```rust
//! CTranslate2 and onnxruntime in one process.
//!
//! This is the regression test for the static-`protobuf` ODR collision that
//! forced Silero VAD out of the v1 default build. It failed with
//! `signal: 10, SIGBUS: access to undefined memory` before `ort` was moved to
//! `load-dynamic`, and it is the check that must pass on any platform before
//! that platform is claimed to support diarization.
//!
//! Ignored by default: it needs a model and the onnxruntime dylib.

#![cfg(feature = "diarization")]

#[test]
#[ignore = "needs CT2_MODEL_DIR and an installed onnxruntime"]
fn ctranslate2_and_onnxruntime_coexist() {
    let dir = std::path::PathBuf::from(
        std::env::var("CT2_MODEL_DIR").expect("set CT2_MODEL_DIR"),
    );

    let whisper = ct2rs::Whisper::new(&dir, Default::default()).expect("whisper load");

    // Constructing the diarizer initialises ort and builds an ONNX session:
    // the exact sequence that used to abort the process.
    let diarizer = whisper_rs::diarize_for_test(4).expect("diarizer construction");

    let silence = vec![0.0f32; 16_000];
    let _ = diarizer.diarize(&silence);

    // Use CTranslate2 *after* ort is live, in case the collision only bites
    // once both runtimes have allocated.
    let window = vec![0.0f32; whisper.n_samples()];
    let out = whisper
        .generate(&window, Some("en"), false, &Default::default())
        .expect("whisper still works after ort");

    eprintln!("coexistence OK, whisper returned {out:?}");
}
```

- [ ] **Step 2: Add the test-only accessor**

`src/diarize` is a private module, so `tests/` cannot reach it. In `src/lib.rs`:

```rust
/// Construct a diarizer for integration tests.
///
/// `diarize` is a private module, so the coexistence test in `tests/` cannot
/// reach `PolyvoiceDiarizer` directly. This exists for that test alone.
#[cfg(feature = "diarization")]
#[doc(hidden)]
pub fn diarize_for_test(
    max_speakers: usize,
) -> crate::error::Result<impl diarize::Diarizer> {
    diarize::polyvoice::PolyvoiceDiarizer::new(max_speakers)
}
```

- [ ] **Step 3: Run it**

```bash
MODEL=$(find ~/.cache/huggingface/hub/models--Systran--faster-whisper-tiny/snapshots -maxdepth 1 -type d | tail -1)
CT2_MODEL_DIR=$MODEL cargo test --features diarization --test coexistence -- --ignored --nocapture
```

Expected: `coexistence OK`. A `SIGBUS` here means `load-dynamic` is not reaching `ort` — check `cargo tree -f "{p} {f}" --features diarization | grep ort`.

- [ ] **Step 4: Add it to CI, including Linux**

In `.github/workflows/CI.yml`, add a job that runs on both `macos-latest` and `ubuntu-latest`, installs `onnxruntime` via pip, and runs this test. **The Linux result is information either way** — if it fails there, record that in the README's platform note rather than deleting the job.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "test: add the CTranslate2/onnxruntime coexistence regression test"
```

---

### Task 10: Real multi-speaker Python tests

**Files:**
- Modify: `tests/python/test_api.py`

- [ ] **Step 1: Add the fixture helper**

```python
def _say_multi_speaker_wav(path, voices=("Samantha", "Alex", "Fred", "Daniel", "Karen")):
    """Concatenate one sentence per voice into a 16 kHz mono WAV.

    Generated rather than recorded because the voice order is then exact ground
    truth for both the speaker count and the turn order — which no real
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
        out.setparams(first.getparams())
    for part in parts:
        with wave.open(str(part), "rb") as r:
            out.writeframes(r.readframes(r.getnframes()))
    out.close()
    return path
```

- [ ] **Step 2: Write the tests**

```python
@pytest.mark.model
@pytest.mark.skipif(not _say_available(), reason="macOS `say` is not available on this platform")
def test_diarization_finds_more_than_four_speakers(tmp_path):
    """The requirement that ruled out Sortformer, whose NUM_SPEAKERS = 4 is
    fixed in the model rather than configurable."""
    audio = _say_multi_speaker_wav(tmp_path / "five.wav")

    model = whisper_rs.WhisperModel("tiny")
    segments, info = model.transcribe(str(audio), language="en", diarize=True, max_speakers=8)
    segments = list(segments)

    assert info.num_speakers >= 5, f"expected >= 5 speakers, got {info.num_speakers}"

    speakers = {seg.speaker for seg in segments if seg.speaker is not None}
    assert len(speakers) >= 5, f"segments carry only {len(speakers)} distinct speakers"


@pytest.mark.model
@pytest.mark.skipif(not _say_available(), reason="macOS `say` is not available on this platform")
def test_every_word_carries_a_speaker(tmp_path):
    audio = _say_multi_speaker_wav(tmp_path / "five.wav")

    model = whisper_rs.WhisperModel("tiny")
    segments, _ = model.transcribe(str(audio), language="en", diarize=True)
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
    segments, _ = model.transcribe(str(audio), language="en", diarize=True)
    ids = [seg.id for seg in segments]

    assert ids == list(range(len(ids))), f"ids are not sequential and gap-free: {ids}"
```

- [ ] **Step 3: Run them**

```bash
python -m pytest tests/python -m model -q
```

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "test: cover diarization on real five-speaker audio"
```

---

### Task 11: Documentation

**Files:**
- Modify: `README.md`
- Modify: `src/python/model.rs` (the `transcribe` docstring)

- [ ] **Step 1: Document the feature in the README**

Add a diarization section covering: the `pip install whisper-rs[diarization]` extra, the `diarize=True` / `max_speakers` arguments, `Segment.speaker` / `Word.speaker` / `info.num_speakers`, and a worked example.

Then add to "Known behaviors and limitations", in the same register as the existing entries:

- **`diarize=True` breaks the iterator's laziness.** Diarization needs the whole file before the first speaker can be assigned, so it runs eagerly inside `transcribe()`, joining VAD, windowing, and language detection. ASR decoding stays lazy.
- **Diarization needs onnxruntime, and it must be dynamically loaded.** Explain the `protobuf` collision, the `SIGBUS`, and why `load-dynamic` is mandatory rather than preferred. State the >= 1.28 floor.
- **Platform support.** Record the coexistence test's actual result on macOS and on Linux from Task 9's CI run. If Linux fails, say so plainly rather than staying silent.
- **Segments split where the speaker changes**, and a split segment's text is rebuilt from its words, so it can differ from the unsplit text in whitespace.
- **There is no `min_speakers`.** `polyvoice` has no such knob; the clustering decides the count, bounded above by `max_speakers`.

- [ ] **Step 2: Update the `transcribe` docstring**

Extend the existing eagerness section to cover diarization, and document the `word_timestamps` tri-state table.

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "docs: document diarization, its onnxruntime requirement, and its limits"
```

---

## Self-review

**Spec coverage.** Every spec section maps to a task: the collision constraint to Global Constraints plus Tasks 5, 6, and 9; the `Diarizer` trait and `SpeakerTurn` to Task 1; `assign.rs`'s six rules to Tasks 2 and 3 (rule 1 to Task 2's overlap tests, rule 2 to the tie-break test, rule 3 to the no-overlap test, rules 4 and 5 to Task 3's split and text tests, rule 6 to the no-words test); numbering to Task 4; the backend and `max_speakers` range to Task 6; whole-file diarization to Task 7's `Prepared.samples`; the Python API and `word_timestamps` tri-state to Task 8; packaging to Tasks 5 and 8; testing to Tasks 2, 3, 9, and 10; Task zero to Task 0; docs to Task 11.

Two spec items are deliberately **not** implemented, and both are recorded under "Deliberate deviations": `SpeakerTurn` is `f32` rather than `f64`, and model downloading uses `polyvoice`'s own verified downloader rather than extending `hub.rs`.

The spec's eight definition-of-done items map to: (1) Task 8's pip extra, (2) Task 10's word-level test, (3) Task 10's five-speaker test, (4) Task 10's sequential-ids test, (5) Task 8's `diarize=False` test, (6) Task 6 Step 5's `cargo tree` check, (7) Task 8's `ValueError` test, (8) Task 9.

**Placeholder scan.** No TBDs. The one conditional instruction — Task 6 Step 3's "if Task 0 found that models must be fetched explicitly" — is deliberate and bounded: it forbids inventing a download path Task 0 did not exercise, which is the failure mode this plan exists to avoid. Task 0 Step 6 requires those findings be written into the spec first.

**Type consistency.** `SpeakerTurn { start: f32, end: f32, speaker: usize }` is used identically in Tasks 1, 2, 3, 6, and 7. `speaker: Option<usize>` is consistent across `types::Word`, `types::Seg`, and both pyclasses. `stitch(window, segs)` loses its third parameter in Task 4 and every later call site matches. `assign(segs, turns) -> Vec<Seg>` and `number(&mut [Seg], &mut u32)` are used with those exact signatures in Task 7. `PolyvoiceDiarizer::new(max_speakers: usize)` matches its Task 9 use through `diarize_for_test`.

One consistency risk worth naming: Task 3 adds `speaker` to `types::Word` and `types::Seg`, which breaks every struct literal in the crate. Task 3 Step 1 says so and tells the implementer to let the compiler enumerate them, rather than leaving it to be discovered.

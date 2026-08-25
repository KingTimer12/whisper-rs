//! Speaker diarization: who spoke when.
//!
//! The concrete backend lives behind the `diarization` feature so the default
//! build contains no `ort` at all. See `polyvoice.rs` (added in a later task)
//! for why that matters.

use crate::error::Result;

pub mod assign;
#[cfg(feature = "diarization")]
pub mod dylib;

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

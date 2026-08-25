//! Plain data types shared across modules. No logic, no pyo3.
//!
//! Nothing here is consumed yet: later tasks (audio pipeline, VAD, ct2rs
//! wrapper) wire these up. Allow dead_code until then so clippy stays clean.
#![allow(dead_code)]

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
    /// The detector's probability for the detected language. `None` when the
    /// caller pinned the language (nothing was detected, so there is no score
    /// to report) or the audio held no speech at all.
    pub language_probability: Option<f32>,
    pub duration: f32,
    pub duration_after_vad: f32,
}

//! Voice activity detection: parameters, region extraction, backend wrapper.
#![allow(dead_code)]

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

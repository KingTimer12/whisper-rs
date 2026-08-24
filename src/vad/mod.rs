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

/// Silero VAD via wavekat-vad. 16 kHz only.
///
/// Note: `wavekat_vad::backends::silero::SileroVad::process` expects `&[i16]`
/// samples (not `&[f32]`), and requires the frame to match its fixed chunk
/// size exactly (512 samples at 16 kHz) or it returns `VadError::InvalidFrameSize`.
/// So this wrapper converts f32 samples in [-1.0, 1.0] to i16 and pads the
/// trailing partial frame with silence before calling `process`.
pub struct SileroBackend {
    inner: wavekat_vad::backends::silero::SileroVad,
    frame: usize,
}

impl SileroBackend {
    pub fn new() -> Result<Self> {
        let inner = wavekat_vad::backends::silero::SileroVad::new(crate::types::SAMPLE_RATE as u32)
            .map_err(|e| crate::error::Error::Vad(e.to_string()))?;
        // Silero's native window at 16 kHz.
        Ok(Self { inner, frame: 512 })
    }
}

fn f32_to_i16(sample: f32) -> i16 {
    (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16
}

impl Vad for SileroBackend {
    fn probabilities(&mut self, samples: &[f32]) -> Result<Vec<f32>> {
        use wavekat_vad::VoiceActivityDetector;

        let mut out = Vec::with_capacity(samples.len() / self.frame + 1);
        for chunk in samples.chunks(self.frame) {
            // The model needs a full frame; pad the tail with silence.
            let frame_i16: Vec<i16> = if chunk.len() == self.frame {
                chunk.iter().copied().map(f32_to_i16).collect()
            } else {
                let mut padded = chunk.to_vec();
                padded.resize(self.frame, 0.0);
                padded.iter().copied().map(f32_to_i16).collect()
            };
            let p = self
                .inner
                .process(&frame_i16, crate::types::SAMPLE_RATE as u32)
                .map_err(|e| crate::error::Error::Vad(e.to_string()))?;
            out.push(p);
        }
        Ok(out)
    }

    fn frame_samples(&self) -> usize {
        self.frame
    }
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
}

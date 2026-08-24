//! rubato-backed resampling to SAMPLE_RATE.
//!
//! Written against rubato 5.0, which replaced the 0.16 `SincFixedIn` type
//! with a single `Async` resampler configured via `FixedAsync`, and moved
//! buffer I/O onto the `audioadapter` crate's `Adapter`/`AdapterMut` traits
//! instead of plain `&[Vec<f32>]` slices.

use crate::error::{Error, Result};
use crate::types::SAMPLE_RATE;
use rubato::audioadapter_buffers::direct::SequentialSlice;
use rubato::{Async, FixedAsync, Resampler, SincInterpolationParameters, SincInterpolationType, WindowFunction};

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
        f_cutoff: Some(0.95),
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };

    let chunk = 1024usize;
    let ratio = SAMPLE_RATE as f64 / from_rate as f64;
    let mut resampler = Async::<f32>::new_sinc(ratio, 2.0, &params, chunk, 1, FixedAsync::Input)
        .map_err(|e| Error::Resample(e.to_string()))?;

    let input_len = samples.len();
    let input = SequentialSlice::new(&samples, 1, input_len).map_err(|e| Error::Resample(e.to_string()))?;

    // `process_all` resamples the whole clip in one call: it internally chunks the input,
    // pads and flushes the filter as needed, and trims the resampler's startup delay itself.
    // This is correct for any input length (including inputs shorter than one chunk) and for
    // any ratio (upsampling or downsampling), unlike manually chunking + padding + trimming by
    // a naive `round(len * ratio)` estimate, which ignores the filter's group delay.
    let produced = resampler
        .process_all(&input, input_len, None)
        .map_err(|e| Error::Resample(e.to_string()))?;

    Ok(produced.take_data())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(rate: u32, secs: f32) -> Vec<f32> {
        let frames = (rate as f32 * secs) as usize;
        (0..frames)
            .map(|i| {
                let t = i as f32 / rate as f32;
                (t * 440.0 * std::f32::consts::TAU).sin() * 0.5
            })
            .collect()
    }

    /// Upsampling (8 kHz -> 16 kHz) must roughly double the sample count, not
    /// under-crop the tail the way a naive `round(len * ratio)` truncation can
    /// for short/padded inputs at ratio > 1.
    #[test]
    fn upsamples_8k_to_16k() {
        let samples = sine(8_000, 1.0);
        let out = to_16k(samples.clone(), 8_000).unwrap();

        let expected = samples.len() * 2;
        let diff = (out.len() as i64 - expected as i64).abs();
        assert!(diff < (expected as i64 / 100).max(2), "expected ~{expected}, got {}", out.len());
    }

    /// An input shorter than one internal processing chunk (1024 frames) must
    /// still resample to a length proportional to the ratio, without panicking
    /// or degenerating to an empty/garbage buffer.
    #[test]
    fn resamples_input_shorter_than_one_chunk() {
        let from_rate = 44_100u32;
        let samples: Vec<f32> = (0..500)
            .map(|i| {
                let t = i as f32 / from_rate as f32;
                (t * 440.0 * std::f32::consts::TAU).sin() * 0.5
            })
            .collect();

        let out = to_16k(samples.clone(), from_rate).unwrap();

        let ratio = SAMPLE_RATE as f64 / from_rate as f64;
        let expected = (samples.len() as f64 * ratio).round() as i64;
        let diff = (out.len() as i64 - expected).abs();
        assert!(diff <= 5, "expected ~{expected}, got {}", out.len());
    }
}

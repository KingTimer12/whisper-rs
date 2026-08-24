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

    let mut out: Vec<f32> = Vec::with_capacity((samples.len() as f64 * ratio) as usize + chunk);
    let mut pos = 0usize;

    while pos < samples.len() {
        let end = (pos + chunk).min(samples.len());
        let mut block = samples[pos..end].to_vec();
        // The final block must be padded to the fixed chunk size.
        block.resize(chunk, 0.0);

        let input = SequentialSlice::new(&block, 1, chunk).map_err(|e| Error::Resample(e.to_string()))?;
        let produced = resampler
            .process(&input, None)
            .map_err(|e| Error::Resample(e.to_string()))?;
        out.extend_from_slice(&produced.take_data());

        pos = end;
    }

    // Trim the tail produced by zero padding the last block.
    let expected = (samples.len() as f64 * ratio).round() as usize;
    out.truncate(expected.min(out.len()));

    Ok(out)
}

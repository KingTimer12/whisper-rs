//! Audio loading: any supported container to 16 kHz mono f32.
//!
//! Nothing outside tests calls this yet: Task 10 wires it into the pyo3
//! surface. Allow dead_code until then so clippy stays clean, matching the
//! pattern used in error.rs and types.rs.
#![allow(dead_code)]

pub mod decode;
pub mod resample;

use crate::error::Result;
use std::path::Path;

/// Decode `path` and return 16 kHz mono f32 samples in [-1, 1].
pub fn load_16k_mono(path: &Path) -> Result<Vec<f32>> {
    let decoded = decode::decode_file(path)?;
    resample::to_16k(decoded.samples, decoded.sample_rate)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Write a mono WAV of a 440 Hz sine at `rate` Hz for `secs` seconds.
    fn write_sine_wav(path: &Path, rate: u32, secs: f32, channels: u16) {
        let spec = hound::WavSpec {
            channels,
            sample_rate: rate,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        let mut writer = hound::WavWriter::create(path, spec).unwrap();
        let frames = (rate as f32 * secs) as usize;
        for i in 0..frames {
            let t = i as f32 / rate as f32;
            let v = (t * 440.0 * std::f32::consts::TAU).sin() * 0.5;
            for _ in 0..channels {
                writer.write_sample(v).unwrap();
            }
        }
        writer.finalize().unwrap();
    }

    #[test]
    fn loads_16k_mono_wav_unchanged_in_length() {
        let dir = std::env::temp_dir().join("whisper_rs_t2_a");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("mono16k.wav");
        write_sine_wav(&path, 16_000, 1.0, 1);

        let samples = load_16k_mono(&path).unwrap();

        assert!(
            (samples.len() as i64 - 16_000).abs() <= 1,
            "expected ~16000 samples, got {}",
            samples.len()
        );
        assert!(samples.iter().all(|s| s.abs() <= 1.0), "samples must be normalised");
    }

    #[test]
    fn resamples_44100_to_16k() {
        let dir = std::env::temp_dir().join("whisper_rs_t2_b");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("mono44k.wav");
        write_sine_wav(&path, 44_100, 1.0, 1);

        let samples = load_16k_mono(&path).unwrap();

        // One second in, so within 1% of 16000 samples.
        let diff = (samples.len() as f32 - 16_000.0).abs();
        assert!(diff < 160.0, "expected ~16000 samples, got {}", samples.len());
    }

    #[test]
    fn downmixes_stereo_to_mono() {
        let dir = std::env::temp_dir().join("whisper_rs_t2_c");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("stereo16k.wav");
        write_sine_wav(&path, 16_000, 1.0, 2);

        let samples = load_16k_mono(&path).unwrap();

        assert!(
            (samples.len() as i64 - 16_000).abs() <= 1,
            "stereo must collapse to one channel, got {} samples",
            samples.len()
        );
    }

    #[test]
    fn missing_file_is_an_audio_read_error() {
        let err = load_16k_mono(Path::new("/nonexistent/nope.wav")).unwrap_err();
        assert!(matches!(err, crate::error::Error::AudioRead { .. }), "got {err:?}");
    }
}

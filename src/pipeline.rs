//! The eager half of transcription: decode, VAD, window.
#![allow(dead_code)]

use crate::error::Result;
use crate::types::{Info, Window, SAMPLE_RATE};
use crate::vad::VadParams;
use std::path::Path;

pub struct Prepared {
    pub windows: Vec<Window>,
    /// `language` is empty here; the caller fills it after detection.
    pub info: Info,
}

/// Decode `path`, run VAD, and plan decoder windows.
///
/// With `vad_filter` off, the audio is windowed as one long region — the 30 s
/// ceiling still applies, so windows are cut every 30 s.
pub fn prepare(path: &Path, vad_filter: bool, params: &VadParams) -> Result<Prepared> {
    let samples = crate::audio::load_16k_mono(path)?;
    let duration = samples.len() as f32 / SAMPLE_RATE as f32;

    let (windows, after_vad) = if vad_filter {
        let mut vad = crate::vad::default_backend()?;
        let (probs, regions) = crate::vad::detect(vad.as_mut(), &samples, params)?;
        let frame_samples = vad.frame_samples();
        windows_from_samples(&samples, &regions, &probs, frame_samples)
    } else {
        let all = [crate::types::SpeechRegion { start: 0, end: samples.len() }];
        windows_from_samples(&samples, &all, &[], 0)
    };

    Ok(Prepared {
        windows,
        info: Info {
            language: String::new(),
            language_probability: None,
            duration,
            duration_after_vad: after_vad,
        },
    })
}

/// Split out so windowing is testable without touching a VAD model.
pub(crate) fn windows_from_samples(
    samples: &[f32],
    regions: &[crate::types::SpeechRegion],
    probs: &[f32],
    frame_samples: usize,
) -> (Vec<Window>, f32) {
    let ranges = crate::chunk::plan_windows(regions, probs, frame_samples);
    let windows: Vec<Window> = ranges
        .iter()
        .map(|&r| crate::chunk::build_window(samples, r))
        .collect();
    let speech_samples: usize = regions.iter().map(|r| r.len()).sum();
    (windows, speech_samples as f32 / SAMPLE_RATE as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{SpeechRegion, WINDOW_SAMPLES};

    #[test]
    fn windows_cover_the_regions_and_report_speech_duration() {
        let samples = vec![0.3f32; 40 * SAMPLE_RATE];
        let regions = vec![
            SpeechRegion { start: 0, end: 10 * SAMPLE_RATE },
            SpeechRegion { start: 20 * SAMPLE_RATE, end: 35 * SAMPLE_RATE },
        ];

        let (windows, after_vad) = windows_from_samples(&samples, &regions, &[], 512);

        assert!(!windows.is_empty());
        for w in &windows {
            assert_eq!(w.samples.len(), WINDOW_SAMPLES, "every window is padded to 30 s");
            assert!(w.real_len <= WINDOW_SAMPLES);
        }
        assert!(
            (after_vad - 25.0).abs() < 0.01,
            "10 s + 15 s of speech expected, got {after_vad}"
        );
    }

    #[test]
    fn speech_duration_equals_the_sum_of_window_real_lengths_when_regions_do_not_touch_padding() {
        // One contiguous region: real_len across windows must add up to it.
        let samples = vec![0.3f32; 70 * SAMPLE_RATE];
        let regions = vec![SpeechRegion { start: 0, end: 70 * SAMPLE_RATE }];

        let (windows, after_vad) = windows_from_samples(&samples, &regions, &[], 512);
        let total_real: usize = windows.iter().map(|w| w.real_len).sum();

        assert!(
            (total_real as f32 / SAMPLE_RATE as f32 - after_vad).abs() < 0.01,
            "window real lengths ({}) must sum to duration_after_vad ({after_vad})",
            total_real as f32 / SAMPLE_RATE as f32
        );
    }

    #[test]
    fn no_speech_sample_lands_in_two_windows() {
        let samples = vec![0.3f32; 95 * SAMPLE_RATE];
        let regions = vec![SpeechRegion { start: 0, end: 95 * SAMPLE_RATE }];

        let (windows, _) = windows_from_samples(&samples, &regions, &[], 512);

        for pair in windows.windows(2) {
            let a_end = pair[0].offset + pair[0].real_len;
            assert!(
                a_end <= pair[1].offset,
                "windows overlap: {}..{} then {}",
                pair[0].offset,
                a_end,
                pair[1].offset
            );
        }
    }

    #[test]
    fn silent_audio_yields_no_windows() {
        let samples = vec![0.0f32; 10 * SAMPLE_RATE];
        let (windows, after_vad) = windows_from_samples(&samples, &[], &[], 512);
        assert!(windows.is_empty());
        assert_eq!(after_vad, 0.0);
    }
}

//! Pure: per-frame speech probabilities to speech regions.

use super::VadParams;
use crate::types::SpeechRegion;

/// Turn per-frame speech probabilities into speech regions in sample space.
///
/// Two thresholds (hysteresis): a frame at or above `threshold` opens a region,
/// and the region only closes once the probability has stayed below
/// `neg_threshold` for at least `min_silence_ms`.
pub fn regions_from_probs(
    probs: &[f32],
    params: &VadParams,
    total_samples: usize,
) -> Vec<SpeechRegion> {
    let frame = params.frame_samples;
    let min_silence = VadParams::ms_to_samples(params.min_silence_ms);
    let min_speech = VadParams::ms_to_samples(params.min_speech_ms);
    let pad = VadParams::ms_to_samples(params.speech_pad_ms);

    let sample_at = |i: usize| (i * frame).min(total_samples);

    let mut regions: Vec<SpeechRegion> = Vec::new();
    let mut start: Option<usize> = None;
    // First frame index of the current below-neg_threshold run, if any.
    let mut silence_from: Option<usize> = None;

    for (i, &p) in probs.iter().enumerate() {
        if p >= params.threshold {
            if start.is_none() {
                start = Some(i);
            }
            silence_from = None;
        } else if p < params.neg_threshold
            && let Some(s) = start
        {
            let run_start = *silence_from.get_or_insert(i);
            let silence_len = sample_at(i + 1) - sample_at(run_start);
            if silence_len >= min_silence {
                regions.push(SpeechRegion {
                    start: sample_at(s),
                    end: sample_at(run_start),
                });
                start = None;
                silence_from = None;
            }
        }
        // Between neg_threshold and threshold: hold the current state.
    }

    if let Some(s) = start {
        regions.push(SpeechRegion {
            start: sample_at(s),
            end: total_samples,
        });
    }

    regions.retain(|r| r.len() >= min_speech && !r.is_empty());

    if pad > 0 {
        for r in &mut regions {
            r.start = r.start.saturating_sub(pad);
            r.end = (r.end + pad).min(total_samples);
        }
    }

    merge_overlapping(regions)
}

/// Merge regions that touch or overlap. Input must be sorted by `start`.
fn merge_overlapping(regions: Vec<SpeechRegion>) -> Vec<SpeechRegion> {
    let mut out: Vec<SpeechRegion> = Vec::with_capacity(regions.len());
    for r in regions {
        match out.last_mut() {
            Some(prev) if r.start <= prev.end => prev.end = prev.end.max(r.end),
            _ => out.push(r),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Params with no padding and no minimum durations, so tests can check the
    /// hysteresis machine in isolation. 1600 samples per frame = 100 ms.
    fn bare_params() -> VadParams {
        VadParams {
            threshold: 0.5,
            neg_threshold: 0.35,
            min_speech_ms: 0,
            min_silence_ms: 0,
            speech_pad_ms: 0,
            frame_samples: 1600,
        }
    }

    #[test]
    fn all_silence_yields_no_regions() {
        let probs = vec![0.0; 50];
        let r = regions_from_probs(&probs, &bare_params(), 50 * 1600);
        assert!(r.is_empty(), "got {r:?}");
    }

    #[test]
    fn all_speech_yields_one_region_covering_everything() {
        let probs = vec![0.9; 10];
        let r = regions_from_probs(&probs, &bare_params(), 10 * 1600);
        assert_eq!(r, vec![SpeechRegion { start: 0, end: 16_000 }]);
    }

    #[test]
    fn speech_at_the_very_start_is_kept() {
        // speech in frames 0..2, then silence
        let mut probs = vec![0.0; 10];
        probs[0] = 0.9;
        probs[1] = 0.9;
        let r = regions_from_probs(&probs, &bare_params(), 10 * 1600);
        assert_eq!(r, vec![SpeechRegion { start: 0, end: 3_200 }]);
    }

    #[test]
    fn speech_running_to_the_last_frame_is_closed_at_total_samples() {
        let mut probs = vec![0.0; 5];
        probs[3] = 0.9;
        probs[4] = 0.9;
        let total = 5 * 1600;
        let r = regions_from_probs(&probs, &bare_params(), total);
        assert_eq!(r, vec![SpeechRegion { start: 4_800, end: total }]);
    }

    #[test]
    fn dip_between_thresholds_does_not_split_the_region() {
        // 0.9, 0.4 (below threshold but above neg_threshold), 0.9
        let probs = vec![0.9, 0.4, 0.9];
        let r = regions_from_probs(&probs, &bare_params(), 3 * 1600);
        assert_eq!(
            r,
            vec![SpeechRegion { start: 0, end: 4_800 }],
            "a dip inside the hysteresis band must not split speech"
        );
    }

    #[test]
    fn short_silence_below_min_silence_does_not_split_the_region() {
        let params = VadParams {
            min_silence_ms: 300, // 3 frames
            ..bare_params()
        };
        // speech, 2 frames of true silence (200 ms < 300 ms), speech
        let probs = vec![0.9, 0.0, 0.0, 0.9];
        let r = regions_from_probs(&probs, &params, 4 * 1600);
        assert_eq!(r, vec![SpeechRegion { start: 0, end: 6_400 }]);
    }

    #[test]
    fn long_silence_splits_into_two_regions() {
        let params = VadParams {
            min_silence_ms: 200, // 2 frames
            ..bare_params()
        };
        let probs = vec![0.9, 0.0, 0.0, 0.0, 0.9];
        let r = regions_from_probs(&probs, &params, 5 * 1600);
        assert_eq!(
            r,
            vec![
                SpeechRegion { start: 0, end: 1_600 },
                SpeechRegion { start: 6_400, end: 8_000 },
            ]
        );
    }

    #[test]
    fn regions_shorter_than_min_speech_are_discarded() {
        let params = VadParams {
            min_speech_ms: 250, // needs 4000 samples, one frame is 1600
            min_silence_ms: 100,
            ..bare_params()
        };
        // a single 100 ms speech blip
        let probs = vec![0.0, 0.9, 0.0, 0.0, 0.0];
        let r = regions_from_probs(&probs, &params, 5 * 1600);
        assert!(r.is_empty(), "a 100 ms blip must be dropped, got {r:?}");
    }

    #[test]
    fn padding_expands_edges_and_is_clamped_to_the_audio() {
        let params = VadParams {
            speech_pad_ms: 100, // 1600 samples
            ..bare_params()
        };
        // speech only in frame 0
        let probs = vec![0.9, 0.0, 0.0];
        let r = regions_from_probs(&probs, &params, 3 * 1600);
        // start clamps to 0, end grows by 1600
        assert_eq!(r, vec![SpeechRegion { start: 0, end: 3_200 }]);
    }

    #[test]
    fn padding_merges_regions_that_come_to_overlap() {
        let params = VadParams {
            min_silence_ms: 100,
            speech_pad_ms: 200, // 3200 samples each side
            ..bare_params()
        };
        // speech, silence, speech — padding closes the 1600 sample gap
        let probs = vec![0.9, 0.0, 0.9];
        let r = regions_from_probs(&probs, &params, 3 * 1600);
        assert_eq!(
            r,
            vec![SpeechRegion { start: 0, end: 4_800 }],
            "padded regions that overlap must merge"
        );
    }

    #[test]
    fn empty_probs_yields_no_regions() {
        let r = regions_from_probs(&[], &bare_params(), 0);
        assert!(r.is_empty());
    }
}

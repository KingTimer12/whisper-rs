//! Pack speech regions into decoder windows of at most WINDOW_SAMPLES.
#![allow(dead_code)]

use crate::types::{SpeechRegion, Window, WINDOW_SAMPLES};

/// Samples searched for a good split point at the end of an over-long window.
const CUT_SEARCH_SAMPLES: usize = 2 * crate::types::SAMPLE_RATE;

/// Plan window boundaries as (start, end) sample ranges.
///
/// Regions are packed in order while they fit inside WINDOW_SAMPLES from the
/// current offset. A single region longer than the ceiling is sliced, cutting at
/// the lowest speech probability inside the last CUT_SEARCH_SAMPLES of the
/// window — a slightly early cut in a quiet moment beats a hard cut mid-word.
pub fn plan_windows(
    regions: &[SpeechRegion],
    probs: &[f32],
    frame_samples: usize,
) -> Vec<(usize, usize)> {
    let mut out: Vec<(usize, usize)> = Vec::new();
    let mut current: Option<(usize, usize)> = None;

    for region in regions {
        if region.is_empty() {
            continue;
        }

        // Flush the accumulator when this region cannot fit.
        if let Some((start, end)) = current
            && region.end - start > WINDOW_SAMPLES
        {
            out.push((start, end));
            current = None;
        }

        if current.is_none() && region.len() > WINDOW_SAMPLES {
            // Slice the over-long region on its own.
            let mut pos = region.start;
            while region.end - pos > WINDOW_SAMPLES {
                let cut = choose_cut(pos, probs, frame_samples);
                out.push((pos, cut));
                pos = cut;
            }
            current = Some((pos, region.end));
            continue;
        }

        current = match current {
            Some((start, _)) => Some((start, region.end)),
            None => Some((region.start, region.end)),
        };
    }

    if let Some(range) = current {
        out.push(range);
    }

    out
}

/// Pick the end of a window starting at `start`, at or before start+WINDOW_SAMPLES.
fn choose_cut(start: usize, probs: &[f32], frame_samples: usize) -> usize {
    let hard = start + WINDOW_SAMPLES;
    if probs.is_empty() || frame_samples == 0 {
        return hard;
    }

    let search_from = hard - CUT_SEARCH_SAMPLES;
    let first = search_from / frame_samples;
    let last = (hard / frame_samples).min(probs.len());
    if first >= last {
        return hard;
    }

    let quietest = (first..last)
        .min_by(|&a, &b| probs[a].partial_cmp(&probs[b]).unwrap_or(std::cmp::Ordering::Equal));

    match quietest {
        Some(i) => {
            let cut = i * frame_samples;
            // Never go backwards or produce an empty window.
            if cut > start { cut } else { hard }
        }
        None => hard,
    }
}

/// Copy `range` out of `samples` and zero-pad to exactly WINDOW_SAMPLES.
pub fn build_window(samples: &[f32], range: (usize, usize)) -> Window {
    let (start, end) = range;
    let start = start.min(samples.len());
    let end = end.min(samples.len()).max(start);

    let mut buf = Vec::with_capacity(WINDOW_SAMPLES);
    buf.extend_from_slice(&samples[start..end]);
    let real_len = buf.len().min(WINDOW_SAMPLES);
    buf.truncate(WINDOW_SAMPLES);
    buf.resize(WINDOW_SAMPLES, 0.0);

    Window {
        offset: start,
        samples: buf,
        real_len,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const S: usize = crate::types::SAMPLE_RATE; // 1 second

    #[test]
    fn no_regions_yields_no_windows() {
        assert!(plan_windows(&[], &[], 512).is_empty());
    }

    #[test]
    fn several_short_regions_pack_into_one_window() {
        let regions = vec![
            SpeechRegion { start: 0, end: 5 * S },
            SpeechRegion { start: 7 * S, end: 12 * S },
            SpeechRegion { start: 15 * S, end: 20 * S },
        ];
        let w = plan_windows(&regions, &[], 512);
        assert_eq!(w, vec![(0, 20 * S)], "all within 30 s, so one window");
    }

    #[test]
    fn regions_crossing_thirty_seconds_start_a_new_window() {
        let regions = vec![
            SpeechRegion { start: 0, end: 20 * S },
            SpeechRegion { start: 25 * S, end: 40 * S },
        ];
        let w = plan_windows(&regions, &[], 512);
        assert_eq!(
            w,
            vec![(0, 20 * S), (25 * S, 40 * S)],
            "the second region would exceed 30 s from offset 0"
        );
    }

    #[test]
    fn every_window_respects_the_thirty_second_ceiling() {
        let regions = vec![
            SpeechRegion { start: 0, end: 10 * S },
            SpeechRegion { start: 11 * S, end: 29 * S },
            SpeechRegion { start: 30 * S, end: 95 * S },
        ];
        let w = plan_windows(&regions, &[], 512);
        for (start, end) in &w {
            assert!(
                end - start <= WINDOW_SAMPLES,
                "window {start}..{end} is longer than 30 s"
            );
        }
    }

    #[test]
    fn window_offsets_are_strictly_increasing() {
        let regions = vec![
            SpeechRegion { start: 0, end: 95 * S },
            SpeechRegion { start: 100 * S, end: 130 * S },
        ];
        let w = plan_windows(&regions, &[], 512);
        for pair in w.windows(2) {
            assert!(pair[0].0 < pair[1].0, "offsets must increase: {w:?}");
        }
    }

    #[test]
    fn a_ninety_second_region_becomes_three_windows() {
        let regions = vec![SpeechRegion { start: 0, end: 90 * S }];
        let w = plan_windows(&regions, &[], 512);
        assert_eq!(w.len(), 3, "got {w:?}");
        assert_eq!(w[0].0, 0);
        assert_eq!(w.last().unwrap().1, 90 * S, "the tail must not be lost");
    }

    #[test]
    fn no_speech_sample_is_lost() {
        let regions = vec![
            SpeechRegion { start: 3 * S, end: 40 * S },
            SpeechRegion { start: 50 * S, end: 55 * S },
        ];
        let w = plan_windows(&regions, &[], 512);

        for r in &regions {
            for sample in [r.start, (r.start + r.end) / 2, r.end - 1] {
                assert!(
                    w.iter().any(|&(s, e)| sample >= s && sample < e),
                    "sample {sample} of region {r:?} is in no window: {w:?}"
                );
            }
        }
    }

    #[test]
    fn no_speech_sample_appears_in_two_windows() {
        let regions = vec![SpeechRegion { start: 0, end: 95 * S }];
        let w = plan_windows(&regions, &[], 512);
        for pair in w.windows(2) {
            assert!(
                pair[0].1 <= pair[1].0,
                "windows overlap: {:?} and {:?}",
                pair[0],
                pair[1]
            );
        }
    }

    #[test]
    fn long_region_splits_at_the_lowest_probability_in_the_last_two_seconds() {
        let frame = S; // 1 s frames keep the arithmetic obvious
        // 40 s region. Candidate cut window is 28..30 s, i.e. frames 28 and 29.
        let mut probs = vec![0.9f32; 40];
        probs[28] = 0.1; // the quietest candidate
        let regions = vec![SpeechRegion { start: 0, end: 40 * S }];

        let w = plan_windows(&regions, &probs, frame);

        assert_eq!(w[0], (0, 28 * S), "must cut at the quietest frame, got {w:?}");
        assert_eq!(w[1].0, 28 * S, "the next window resumes at the cut");
    }

    #[test]
    fn long_region_falls_back_to_a_hard_cut_without_probabilities() {
        let regions = vec![SpeechRegion { start: 0, end: 40 * S }];
        let w = plan_windows(&regions, &[], 512);
        assert_eq!(w[0], (0, WINDOW_SAMPLES), "no probs means cut at exactly 30 s");
    }

    #[test]
    fn build_window_pads_to_exactly_thirty_seconds() {
        let samples = vec![0.5f32; 10 * S];
        let win = build_window(&samples, (2 * S, 7 * S));

        assert_eq!(win.offset, 2 * S);
        assert_eq!(win.real_len, 5 * S);
        assert_eq!(win.samples.len(), WINDOW_SAMPLES, "must be padded to 30 s");
        assert!(win.samples[..5 * S].iter().all(|&s| s == 0.5));
        assert!(win.samples[5 * S..].iter().all(|&s| s == 0.0), "tail must be silence");
    }

    #[test]
    fn build_window_clamps_a_range_past_the_end_of_the_audio() {
        let samples = vec![0.5f32; 3 * S];
        let win = build_window(&samples, (2 * S, 9 * S));

        assert_eq!(win.real_len, S, "only one second of audio actually exists");
        assert_eq!(win.samples.len(), WINDOW_SAMPLES);
    }
}

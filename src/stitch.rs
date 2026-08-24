//! Shift per-window segments onto the global timeline.
#![allow(dead_code)]

use crate::types::{Seg, Window, SAMPLE_RATE};

/// Shift `segs` (window-relative seconds) onto the global timeline.
///
/// Drops segments that start inside the zero padding, clamps ends to the real
/// window end, drops blank segments, and renumbers ids from `next_id`.
pub fn stitch(window: &Window, segs: Vec<Seg>, next_id: &mut u32) -> Vec<Seg> {
    let offset = window.offset as f32 / SAMPLE_RATE as f32;
    let real = window.real_len as f32 / SAMPLE_RATE as f32;
    let limit = offset + real;

    let mut out = Vec::with_capacity(segs.len());

    for seg in segs {
        if seg.start >= real {
            // Started in the padding: the model invented it.
            continue;
        }
        if seg.text.trim().is_empty() {
            continue;
        }

        let words = seg.words.map(|ws| {
            ws.into_iter()
                .filter(|w| w.start < real)
                .map(|w| crate::types::Word {
                    start: offset + w.start,
                    end: (offset + w.end).min(limit),
                    text: w.text,
                    probability: w.probability,
                })
                .collect()
        });

        out.push(Seg {
            id: *next_id,
            start: offset + seg.start,
            end: (offset + seg.end).min(limit),
            text: seg.text,
            words,
        });
        *next_id += 1;
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Word, WINDOW_SAMPLES};

    fn window(offset_secs: f32, real_secs: f32) -> Window {
        Window {
            offset: (offset_secs * SAMPLE_RATE as f32) as usize,
            samples: vec![0.0; WINDOW_SAMPLES],
            real_len: (real_secs * SAMPLE_RATE as f32) as usize,
        }
    }

    fn seg(start: f32, end: f32, text: &str) -> Seg {
        Seg { id: 0, start, end, text: text.into(), words: None }
    }

    #[test]
    fn timestamps_are_shifted_by_the_window_offset() {
        let w = window(60.0, 30.0);
        let mut id = 0;
        let out = stitch(&w, vec![seg(1.0, 2.5, "hello")], &mut id);

        assert_eq!(out.len(), 1);
        assert!((out[0].start - 61.0).abs() < 1e-4, "got {}", out[0].start);
        assert!((out[0].end - 62.5).abs() < 1e-4, "got {}", out[0].end);
    }

    #[test]
    fn ids_are_sequential_across_windows() {
        let mut id = 0;
        let a = stitch(&window(0.0, 30.0), vec![seg(0.0, 1.0, "a"), seg(1.0, 2.0, "b")], &mut id);
        let b = stitch(&window(30.0, 30.0), vec![seg(0.0, 1.0, "c")], &mut id);

        assert_eq!(a.iter().map(|s| s.id).collect::<Vec<_>>(), vec![0, 1]);
        assert_eq!(b[0].id, 2, "numbering must continue across windows");
        assert_eq!(id, 3, "the counter must be left ready for the next window");
    }

    #[test]
    fn segments_starting_inside_the_padding_are_dropped() {
        // Only 10 s of real audio in this window.
        let w = window(0.0, 10.0);
        let mut id = 0;
        let out = stitch(
            &w,
            vec![seg(2.0, 4.0, "real"), seg(12.0, 14.0, "hallucinated in padding")],
            &mut id,
        );

        assert_eq!(out.len(), 1, "got {out:?}");
        assert_eq!(out[0].text, "real");
    }

    #[test]
    fn segment_end_is_clamped_to_the_real_window_end() {
        let w = window(0.0, 10.0);
        let mut id = 0;
        let out = stitch(&w, vec![seg(9.0, 25.0, "runs into padding")], &mut id);

        assert_eq!(out.len(), 1);
        assert!((out[0].end - 10.0).abs() < 1e-4, "end must clamp to 10 s, got {}", out[0].end);
    }

    #[test]
    fn word_timestamps_are_shifted_and_clamped_too() {
        let w = window(10.0, 10.0);
        let mut id = 0;
        let segs = vec![Seg {
            id: 0,
            start: 1.0,
            end: 12.0,
            text: "two words".into(),
            words: Some(vec![
                Word { start: 1.0, end: 1.5, text: "two".into(), probability: 0.9 },
                Word { start: 9.5, end: 12.0, text: "words".into(), probability: 0.8 },
            ]),
        }];

        let out = stitch(&w, segs, &mut id);
        let words = out[0].words.as_ref().unwrap();

        assert!((words[0].start - 11.0).abs() < 1e-4, "got {}", words[0].start);
        assert!((words[1].end - 20.0).abs() < 1e-4, "word end must clamp, got {}", words[1].end);
    }

    #[test]
    fn no_segments_in_yields_no_segments_out() {
        let mut id = 5;
        let out = stitch(&window(0.0, 30.0), vec![], &mut id);
        assert!(out.is_empty());
        assert_eq!(id, 5, "the counter must not move");
    }

    #[test]
    fn empty_text_segments_are_dropped() {
        let mut id = 0;
        let out = stitch(&window(0.0, 30.0), vec![seg(0.0, 1.0, "   ")], &mut id);
        assert!(out.is_empty(), "whitespace-only segments carry no information");
    }
}

//! Shift per-window segments onto the global timeline.
#![allow(dead_code)]

use crate::types::{Seg, Window, SAMPLE_RATE};

/// Shift `segs` (window-relative seconds) onto the global timeline.
///
/// Drops segments that start inside the zero padding, clamps ends to the real
/// window end, and drops blank segments. Returned segments carry `id: 0`;
/// numbering happens later, via `number`, once the final segment count is
/// known (per-word diarization can split a stitched segment further).
pub fn stitch(window: &Window, segs: Vec<Seg>) -> Vec<Seg> {
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

        let words = seg
            .words
            .map(|ws| {
                ws.into_iter()
                    .filter(|w| w.start < real)
                    .map(|w| crate::types::Word {
                        start: offset + w.start,
                        end: (offset + w.end).min(limit),
                        text: w.text,
                        probability: w.probability,
                        speaker: None,
                    })
                    .collect::<Vec<_>>()
            })
            // If every word started inside the padding, `words` would become
            // `Some(vec![])` here instead of `None`. That's ambiguous with
            // "word timestamps were requested and none survived stitching"
            // versus the actual sentinel callers rely on
            // (`seg.words is None` means "word_timestamps was not
            // requested"), so collapse the all-filtered-out case back to
            // `None`.
            .filter(|ws| !ws.is_empty());

        out.push(Seg {
            id: 0,
            start: offset + seg.start,
            end: (offset + seg.end).min(limit),
            text: seg.text,
            words,
            speaker: None,
        });
    }

    out
}

/// Assign sequential ids to `segs`, continuing from `next_id`.
///
/// Numbering is separate from stitching because per-word diarization can split
/// one stitched segment into several, and the iterator handing segments to the
/// caller is lazy — an id given out early cannot be revised once a later split
/// changes the count. Numbering last keeps ids sequential and gap-free, which
/// is what they promise.
pub fn number(segs: &mut [Seg], next_id: &mut u32) {
    for seg in segs {
        seg.id = *next_id;
        *next_id += 1;
    }
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
        Seg { id: 0, start, end, text: text.into(), words: None, speaker: None }
    }

    #[test]
    fn timestamps_are_shifted_by_the_window_offset() {
        let w = window(60.0, 30.0);
        let out = stitch(&w, vec![seg(1.0, 2.5, "hello")]);

        assert_eq!(out.len(), 1);
        assert!((out[0].start - 61.0).abs() < 1e-4, "got {}", out[0].start);
        assert!((out[0].end - 62.5).abs() < 1e-4, "got {}", out[0].end);
    }

    #[test]
    fn stitch_leaves_ids_at_zero() {
        // Numbering happens after splitting, so stitch must not claim ids.
        let out = stitch(&window(0.0, 30.0), vec![seg(0.0, 0.5, "hi")]);
        assert_eq!(out[0].id, 0);
    }

    #[test]
    fn numbering_is_sequential_across_calls() {
        let mut next = 0;
        let mut first = vec![seg(0.0, 1.0, " a"), seg(1.0, 2.0, " b")];
        number(&mut first, &mut next);
        let mut second = vec![seg(2.0, 3.0, " c")];
        number(&mut second, &mut next);

        assert_eq!(first.iter().map(|s| s.id).collect::<Vec<_>>(), vec![0, 1]);
        assert_eq!(second[0].id, 2, "ids continue across calls, without a gap");
        assert_eq!(next, 3);
    }

    #[test]
    fn numbering_an_empty_slice_does_not_advance_the_counter() {
        let mut next = 7;
        number(&mut [], &mut next);
        assert_eq!(next, 7);
    }

    #[test]
    fn segments_starting_inside_the_padding_are_dropped() {
        // Only 10 s of real audio in this window.
        let w = window(0.0, 10.0);
        let out = stitch(
            &w,
            vec![seg(2.0, 4.0, "real"), seg(12.0, 14.0, "hallucinated in padding")],
        );

        assert_eq!(out.len(), 1, "got {out:?}");
        assert_eq!(out[0].text, "real");
    }

    #[test]
    fn segment_end_is_clamped_to_the_real_window_end() {
        let w = window(0.0, 10.0);
        let out = stitch(&w, vec![seg(9.0, 25.0, "runs into padding")]);

        assert_eq!(out.len(), 1);
        assert!((out[0].end - 10.0).abs() < 1e-4, "end must clamp to 10 s, got {}", out[0].end);
    }

    #[test]
    fn word_timestamps_are_shifted_and_clamped_too() {
        let w = window(10.0, 10.0);
        let segs = vec![Seg {
            id: 0,
            start: 1.0,
            end: 12.0,
            text: "two words".into(),
            words: Some(vec![
                Word { start: 1.0, end: 1.5, text: "two".into(), probability: 0.9, speaker: None },
                Word { start: 9.5, end: 12.0, text: "words".into(), probability: 0.8, speaker: None },
            ]),
            speaker: None,
        }];

        let out = stitch(&w, segs);
        let words = out[0].words.as_ref().unwrap();

        assert!((words[0].start - 11.0).abs() < 1e-4, "got {}", words[0].start);
        assert!((words[1].end - 20.0).abs() < 1e-4, "word end must clamp, got {}", words[1].end);
    }

    #[test]
    fn no_segments_in_yields_no_segments_out() {
        let out = stitch(&window(0.0, 30.0), vec![]);
        assert!(out.is_empty());
    }

    #[test]
    fn words_all_filtered_out_by_padding_become_none_not_an_empty_list() {
        // Real window audio ends at 10 s; the only word starts in the padding.
        let w = window(0.0, 10.0);
        let segs = vec![Seg {
            id: 0,
            start: 1.0,
            end: 9.5,
            text: "real text".into(),
            words: Some(vec![Word {
                start: 12.0,
                end: 13.0,
                text: "hallucinated".into(),
                probability: 0.5,
                speaker: None,
            }]),
            speaker: None,
        }];

        let out = stitch(&w, segs);

        assert_eq!(out.len(), 1);
        assert!(
            out[0].words.is_none(),
            "an all-filtered word list must collapse to None, not Some(vec![]), \
             or callers cannot distinguish it from word_timestamps=False; got {:?}",
            out[0].words
        );
    }

    #[test]
    fn empty_text_segments_are_dropped() {
        let out = stitch(&window(0.0, 30.0), vec![seg(0.0, 1.0, "   ")]);
        assert!(out.is_empty(), "whitespace-only segments carry no information");
    }
}

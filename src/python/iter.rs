use crate::asr::Asr;
use crate::diarize::SpeakerTurn;
use crate::python::segment::Segment;
use crate::python::to_pyerr;
use crate::types::{Seg, Window};
use pyo3::prelude::*;
use std::collections::VecDeque;
use std::sync::Arc;

/// Lazily decodes one window per __next__ call.
#[pyclass]
pub struct SegmentIterator {
    asr: Arc<dyn Asr>,
    windows: VecDeque<Window>,
    pending: VecDeque<Seg>,
    next_id: u32,
    language: String,
    word_timestamps: bool,
    /// Empty when `diarize=False`; `assign` then leaves every speaker `None`
    /// and splits nothing, so v1 behaviour is preserved by the same code
    /// path rather than by a branch.
    turns: Vec<SpeakerTurn>,
}

impl SegmentIterator {
    pub fn new(
        asr: Arc<dyn Asr>,
        windows: Vec<Window>,
        language: String,
        word_timestamps: bool,
        turns: Vec<SpeakerTurn>,
    ) -> Self {
        Self {
            asr,
            windows: windows.into(),
            pending: VecDeque::new(),
            next_id: 0,
            language,
            word_timestamps,
            turns,
        }
    }
}

#[pymethods]
impl SegmentIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>, py: Python<'_>) -> PyResult<Option<Segment>> {
        loop {
            if let Some(seg) = slf.pending.pop_front() {
                return Ok(Some(seg.into()));
            }

            let Some(window) = slf.windows.pop_front() else {
                return Ok(None);
            };

            let asr = Arc::clone(&slf.asr);
            let language = slf.language.clone();
            let words = slf.word_timestamps;

            // Release the GIL for the whole decode: it is the slow part.
            let raw = py.detach(|| asr.transcribe(&window.samples, Some(&language), words));

            let raw = match raw {
                Ok(raw) => raw,
                Err(e) => {
                    // The window was already popped above so it could be
                    // moved into the decode closure; if decoding it failed,
                    // put it back at the front instead of letting it stay
                    // consumed. Otherwise a caller that catches this
                    // `RuntimeError` (Python generators are resumable -- a
                    // `try`/`except` around `next(segments)` is a completely
                    // reasonable thing to do) and keeps iterating would
                    // silently skip 30 s of audio with no indication
                    // anything was lost.
                    slf.windows.push_front(window);
                    return Err(to_pyerr(e));
                }
            };

            let mut next_id = slf.next_id;
            let advanced = advance(&window, raw, &mut next_id, &slf.turns);
            slf.next_id = next_id;
            slf.pending.extend(advanced);
            // Loop again: an empty window must not end the iteration.
        }
    }
}

/// One window's worth of post-decode work: stitch onto the global timeline,
/// assign speakers, then number sequentially from `next_id`.
///
/// Extracted from `__next__` so the id threading is testable without a model:
/// the counter round-trip is the property segment ids depend on, and
/// `__next__` itself cannot be constructed in a unit test because it owns an
/// `Arc<Ct2Asr>`. `turns` is empty on the `diarize=False` path; `assign`
/// leaves every speaker `None` and splits nothing in that case, so that path
/// is the same code path as the diarized one, not a branch.
fn advance(window: &Window, raw: Vec<Seg>, next_id: &mut u32, turns: &[SpeakerTurn]) -> Vec<Seg> {
    let stitched = crate::stitch::stitch(window, raw);
    let mut assigned = crate::diarize::assign::assign(stitched, turns);
    crate::stitch::number(&mut assigned, next_id);
    assigned
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{SAMPLE_RATE, WINDOW_SAMPLES};

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
    fn ids_thread_sequentially_across_windows_via_advance() {
        // This is the production path `__next__` calls: it exercises the
        // exact counter round-trip (read slf.next_id, stitch + number,
        // write the advanced value back) that ids depend on end to end.
        let mut next_id = 0;

        let first = advance(&window(0.0, 30.0), vec![seg(0.0, 1.0, "a"), seg(1.0, 2.0, "b")], &mut next_id, &[]);
        assert_eq!(first.iter().map(|s| s.id).collect::<Vec<_>>(), vec![0, 1]);

        let second = advance(&window(30.0, 30.0), vec![seg(0.0, 1.0, "c")], &mut next_id, &[]);
        assert_eq!(second[0].id, 2, "ids must continue across windows without a gap");

        assert_eq!(next_id, 3, "the counter must be left ready for the next window");
    }
    fn worded(start: f32, end: f32, words: &[(f32, f32, &str)]) -> Seg {
        Seg {
            id: 0,
            start,
            end,
            // Space-joined, and the words below are trimmed: this is the
            // shape ct2rs really produces. A fixture that gives words a
            // leading space encodes an assumption the ASR does not satisfy --
            // exactly what hid a run-together text bug in `assign`.
            text: words.iter().map(|(_, _, w)| *w).collect::<Vec<_>>().join(" "),
            words: Some(
                words
                    .iter()
                    .map(|(s, e, w)| crate::types::Word {
                        start: *s,
                        end: *e,
                        text: (*w).to_string(),
                        probability: 1.0,
                        speaker: None,
                    })
                    .collect(),
            ),
            speaker: None,
        }
    }

    #[test]
    fn turns_reach_the_words_through_advance() {
        // The wiring test that matters: turns handed to `advance` must come
        // back attached to the words. A `turns` field that is stored but never
        // passed on would leave every speaker None and pass any test that only
        // inspects ids.
        let turns = vec![
            SpeakerTurn { start: 0.0, end: 1.0, speaker: 0 },
            SpeakerTurn { start: 1.0, end: 2.0, speaker: 1 },
        ];
        let raw = vec![worded(0.0, 2.0, &[(0.0, 1.0, "a"), (1.0, 2.0, "b")])];

        let mut next_id = 0;
        let out = advance(&window(0.0, 30.0), raw, &mut next_id, &turns);

        // Two speakers over one segment: `assign` splits it at the change.
        assert_eq!(
            out.iter().map(|s| s.speaker).collect::<Vec<_>>(),
            vec![Some(0), Some(1)],
            "each half must carry its own speaker"
        );
        assert_eq!(out.iter().map(|s| s.id).collect::<Vec<_>>(), vec![0, 1]);
        assert_eq!(next_id, 2, "numbering must count the segments after splitting");
    }

    #[test]
    fn no_turns_leaves_speakers_unset_and_splits_nothing() {
        // The diarize=False path, exercised through the same code path rather
        // than a branch: an empty turns slice must be inert.
        let raw = vec![worded(0.0, 2.0, &[(0.0, 1.0, "a"), (1.0, 2.0, "b")])];

        let mut next_id = 0;
        let out = advance(&window(0.0, 30.0), raw, &mut next_id, &[]);

        assert_eq!(out.len(), 1, "nothing to split on");
        assert_eq!(out[0].speaker, None);
        assert_eq!(out[0].text, "a b", "the original ASR text must survive");
        let words = out[0].words.as_ref().expect("words were requested");
        assert!(words.iter().all(|w| w.speaker.is_none()));
    }

}

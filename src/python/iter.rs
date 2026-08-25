use crate::asr::{ct2::Ct2Asr, Asr};
use crate::python::segment::Segment;
use crate::python::to_pyerr;
use crate::types::{Seg, Window};
use pyo3::prelude::*;
use std::collections::VecDeque;
use std::sync::Arc;

/// Lazily decodes one window per __next__ call.
#[pyclass]
pub struct SegmentIterator {
    asr: Arc<Ct2Asr>,
    windows: VecDeque<Window>,
    pending: VecDeque<Seg>,
    next_id: u32,
    language: String,
    word_timestamps: bool,
}

impl SegmentIterator {
    pub fn new(
        asr: Arc<Ct2Asr>,
        windows: Vec<Window>,
        language: String,
        word_timestamps: bool,
    ) -> Self {
        Self {
            asr,
            windows: windows.into(),
            pending: VecDeque::new(),
            next_id: 0,
            language,
            word_timestamps,
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
            let stitched = advance(&window, raw, &mut next_id);
            slf.next_id = next_id;
            slf.pending.extend(stitched);
            // Loop again: an empty window must not end the iteration.
        }
    }
}

/// One window's worth of post-decode work: stitch onto the global timeline,
/// then number sequentially from `next_id`.
///
/// Extracted from `__next__` so the id threading is testable without a model:
/// the counter round-trip is the property segment ids depend on, and
/// `__next__` itself cannot be constructed in a unit test because it owns an
/// `Arc<Ct2Asr>`. (A later `turns: &[SpeakerTurn]` parameter for
/// speaker-assignment slots in here, between stitch and number.)
fn advance(window: &Window, raw: Vec<Seg>, next_id: &mut u32) -> Vec<Seg> {
    let mut stitched = crate::stitch::stitch(window, raw);
    crate::stitch::number(&mut stitched, next_id);
    stitched
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

        let first = advance(&window(0.0, 30.0), vec![seg(0.0, 1.0, "a"), seg(1.0, 2.0, "b")], &mut next_id);
        assert_eq!(first.iter().map(|s| s.id).collect::<Vec<_>>(), vec![0, 1]);

        let second = advance(&window(30.0, 30.0), vec![seg(0.0, 1.0, "c")], &mut next_id);
        assert_eq!(second[0].id, 2, "ids must continue across windows without a gap");

        assert_eq!(next_id, 3, "the counter must be left ready for the next window");
    }
}

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
            let stitched = crate::stitch::stitch(&window, raw, &mut next_id);
            slf.next_id = next_id;
            slf.pending.extend(stitched);
            // Loop again: an empty window must not end the iteration.
        }
    }
}

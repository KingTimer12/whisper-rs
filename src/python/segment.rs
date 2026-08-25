use pyo3::prelude::*;

#[pyclass(frozen, get_all, skip_from_py_object)]
#[derive(Clone)]
pub struct Word {
    pub start: f32,
    pub end: f32,
    pub word: String,
    pub probability: f32,
}

#[pymethods]
impl Word {
    fn __repr__(&self) -> String {
        format!(
            "Word(start={:.2}, end={:.2}, word={:?}, probability={:.2})",
            self.start, self.end, self.word, self.probability
        )
    }
}

#[pyclass(frozen, get_all, skip_from_py_object)]
#[derive(Clone)]
pub struct Segment {
    pub id: u32,
    pub start: f32,
    pub end: f32,
    pub text: String,
    pub words: Option<Vec<Word>>,
}

#[pymethods]
impl Segment {
    fn __repr__(&self) -> String {
        format!(
            "Segment(id={}, start={:.2}, end={:.2}, text={:?})",
            self.id, self.start, self.end, self.text
        )
    }
}

#[pyclass(frozen, get_all, skip_from_py_object)]
#[derive(Clone)]
pub struct TranscriptionInfo {
    pub language: String,
    /// The detector's probability for the detected language. `None` when
    /// `language=` was pinned (nothing was detected) or the audio held no
    /// speech at all.
    pub language_probability: Option<f32>,
    pub duration: f32,
    pub duration_after_vad: f32,
}

#[pymethods]
impl TranscriptionInfo {
    fn __repr__(&self) -> String {
        format!(
            "TranscriptionInfo(language={:?}, duration={:.2}, duration_after_vad={:.2})",
            self.language, self.duration, self.duration_after_vad
        )
    }
}

impl From<crate::types::Seg> for Segment {
    fn from(s: crate::types::Seg) -> Self {
        Self {
            id: s.id,
            start: s.start,
            end: s.end,
            text: s.text,
            words: s.words.map(|ws| {
                ws.into_iter()
                    .map(|w| Word {
                        start: w.start,
                        end: w.end,
                        word: w.text,
                        probability: w.probability,
                    })
                    .collect()
            }),
        }
    }
}

impl From<crate::types::Info> for TranscriptionInfo {
    fn from(i: crate::types::Info) -> Self {
        Self {
            language: i.language,
            language_probability: i.language_probability,
            duration: i.duration,
            duration_after_vad: i.duration_after_vad,
        }
    }
}

mod asr;
mod audio;
mod chunk;
mod error;
mod models;
mod pipeline;
mod python;
mod stitch;
mod types;
mod vad;

use pyo3::prelude::*;

#[pymodule]
fn whisper_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<python::model::WhisperModel>()?;
    m.add_class::<python::iter::SegmentIterator>()?;
    m.add_class::<python::segment::Segment>()?;
    m.add_class::<python::segment::Word>()?;
    m.add_class::<python::segment::TranscriptionInfo>()?;
    Ok(())
}

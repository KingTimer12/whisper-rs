//! End-to-end test with a real model. Ignored by default: it downloads.
//! Run with: cargo test --test e2e -- --ignored --nocapture
//!
//! This test links pyo3 as an embedding client (not the `extension-module`
//! feature) and drives the *installed* `whisper_rs` extension module through
//! the Python interpreter, exactly as a Python caller would. That only works
//! if the wheel/extension has already been installed into the active
//! interpreter (e.g. via `maturin develop`), which is why it stays `#[ignore]`.

use pyo3::prelude::*;
use std::path::PathBuf;

/// Write a 16 kHz mono WAV: one second of silence, a tone, one more second of silence.
fn write_test_wav(path: &PathBuf) {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16_000,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut w = hound::WavWriter::create(path, spec).unwrap();
    for i in 0..(16_000 * 4) {
        let t = i as f32 / 16_000.0;
        let amp = if (1.0..3.0).contains(&t) { 0.4 } else { 0.0 };
        w.write_sample(amp * (t * 440.0 * std::f32::consts::TAU).sin())
            .unwrap();
    }
    w.finalize().unwrap();
}

#[test]
#[ignore = "downloads the tiny model; requires `maturin develop` to have installed whisper_rs first"]
fn tiny_model_transcribes_a_wav_end_to_end() {
    let dir = std::env::temp_dir().join("whisper_rs_e2e");
    std::fs::create_dir_all(&dir).unwrap();
    let audio = dir.join("tone.wav");
    write_test_wav(&audio);

    // This mirrors what the Python layer does, through the public Python class.
    // Rust-side integration is exercised via the pyo3 module, so this test
    // asserts the pipeline runs and produces a coherent timeline rather than
    // asserting specific words for a tone.
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let module = py
            .import("whisper_rs")
            .expect("build the wheel first: maturin develop");
        let model = module
            .getattr("WhisperModel")
            .unwrap()
            .call1(("tiny",))
            .expect("loading the tiny model failed");

        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("vad_filter", true).unwrap();
        let result = model
            .call_method("transcribe", (audio.to_str().unwrap(),), Some(&kwargs))
            .expect("transcribe failed");

        let info = result.get_item(1).unwrap();
        let duration: f32 = info.getattr("duration").unwrap().extract().unwrap();
        assert!((duration - 4.0).abs() < 0.2, "duration was {duration}");

        let segments = result.get_item(0).unwrap();
        let mut last_end = 0.0f32;
        for seg in segments.try_iter().unwrap() {
            let seg = seg.unwrap();
            let start: f32 = seg.getattr("start").unwrap().extract().unwrap();
            let end: f32 = seg.getattr("end").unwrap().extract().unwrap();
            assert!(start >= last_end - 0.01, "segments must not go backwards");
            assert!(end <= duration + 0.5, "segment end {end} is past the audio");
            last_end = end;
        }
    });
}

//! CTranslate2 and onnxruntime in one process.
//!
//! This is the regression test for the static-`protobuf` ODR collision that
//! forced Silero VAD out of the v1 default build. It failed with
//! `signal: 10, SIGBUS: access to undefined memory` before `ort` was moved to
//! `load-dynamic`, and it is the check that must pass on any platform before
//! that platform is claimed to support diarization.
//!
//! Ignored by default: it needs a model and the onnxruntime dylib.

#![cfg(any(feature = "diarization", feature = "nemotron"))]

#[cfg(feature = "diarization")]
use whisper_rs::diarize::Diarizer;
#[cfg(all(feature = "diarization", feature = "nemotron"))]
use whisper_rs::asr::Asr;

#[cfg(feature = "diarization")]
#[test]
#[ignore = "needs CT2_MODEL_DIR and an installed onnxruntime"]
fn ctranslate2_and_onnxruntime_coexist() {
    let dir = std::path::PathBuf::from(
        std::env::var("CT2_MODEL_DIR").expect("set CT2_MODEL_DIR"),
    );

    let whisper =
        ct2rs::Whisper::new(&dir, Default::default()).expect("whisper load");

    // Constructing the diarizer initialises ort and builds an ONNX session:
    // the exact sequence that used to abort the process.
    let diarizer = whisper_rs::diarize_for_test(4).expect("diarizer construction");

    let silence = vec![0.0f32; 16_000];
    let _ = diarizer.diarize(&silence);

    // Use CTranslate2 *after* ort is live, in case the collision only bites
    // once both runtimes have allocated.
    let window = vec![0.0f32; whisper.n_samples()];
    let out = whisper
        .generate(&window, Some("en"), false, &Default::default())
        .expect("whisper still works after ort");

    eprintln!("coexistence OK, whisper returned {out:?}");
}

/// CTranslate2, onnxruntime-via-polyvoice, and onnxruntime-via-parakeet-rs
/// in one process. Only meaningful with both features on: this is the
/// specific combination the v3 design's "Shared `ort` init" section depends
/// on (`onnx::init_ort` must be safely callable from both `PolyvoiceDiarizer::new`
/// and `NemotronAsr::new` in the same process without a second, racing
/// `ort::init_from`).
#[cfg(all(feature = "diarization", feature = "nemotron"))]
#[test]
#[ignore = "needs CT2_MODEL_DIR, NEMOTRON_MODEL_DIR, and an installed onnxruntime"]
fn ctranslate2_diarization_and_nemotron_coexist() {
    let ct2_dir = std::path::PathBuf::from(
        std::env::var("CT2_MODEL_DIR").expect("set CT2_MODEL_DIR"),
    );
    let nemotron_dir = std::path::PathBuf::from(
        std::env::var("NEMOTRON_MODEL_DIR").expect("set NEMOTRON_MODEL_DIR"),
    );

    let whisper = ct2rs::Whisper::new(&ct2_dir, Default::default()).expect("whisper load");
    let diarizer = whisper_rs::diarize_for_test(4).expect("diarizer construction");
    let nemotron = whisper_rs::nemotron_for_test(&nemotron_dir).expect("nemotron construction");

    let silence = vec![0.0f32; 16_000];
    let _ = diarizer.diarize(&silence);
    let _ = nemotron.transcribe(&silence, None, false);

    let window = vec![0.0f32; whisper.n_samples()];
    let out = whisper
        .generate(&window, Some("en"), false, &Default::default())
        .expect("whisper still works after both onnx backends ran");

    eprintln!("three-way coexistence OK, whisper returned {out:?}");
}

//! CTranslate2 and onnxruntime in one process.
//!
//! This is the regression test for the static-`protobuf` ODR collision that
//! forced Silero VAD out of the v1 default build. It failed with
//! `signal: 10, SIGBUS: access to undefined memory` before `ort` was moved to
//! `load-dynamic`, and it is the check that must pass on any platform before
//! that platform is claimed to support diarization.
//!
//! Ignored by default: it needs a model and the onnxruntime dylib.

#![cfg(feature = "diarization")]

use whisper_rs::diarize::Diarizer;

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

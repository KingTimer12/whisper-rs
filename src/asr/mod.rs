//! Speech recognition backends.

pub mod ct2;
pub mod detect;

use crate::error::Result;
use crate::types::Seg;

/// A speech recognition backend. One call decodes one window.
pub trait Asr: Send + Sync {
    /// Transcribe exactly one window of samples.
    fn transcribe(
        &self,
        samples: &[f32],
        language: Option<&str>,
        word_timestamps: bool,
    ) -> Result<Vec<Seg>>;

    /// Detected language code and its probability for `samples`.
    ///
    /// Backends are expected to use a real detector; see
    /// [`detect::LanguageDetector`] for why the language cannot be recovered
    /// from decoder output.
    fn detect_language(&self, samples: &[f32]) -> Result<(String, f32)>;
}

#[cfg(test)]
mod tests {
    /// The Arc<Ct2Asr> design in the Python layer requires this.
    /// If this fails to compile, Ct2Asr must wrap Whisper in a Mutex.
    #[test]
    fn whisper_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ct2rs::Whisper>();
    }
}

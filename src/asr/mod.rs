//! Speech recognition backends.
//!
//! Nothing here is consumed yet: Task 11 wires `Ct2Asr` into the Python
//! layer. Allow dead_code until then so clippy stays clean.
#![allow(dead_code)]

pub mod ct2;

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

    /// Best-effort language code for `samples`.
    fn detect_language(&self, samples: &[f32]) -> Result<String>;
}

/// Pull the language code out of raw Whisper output.
///
/// ct2rs exposes no language-detection API, so the code is recovered from the
/// `<|xx|>` token Whisper emits at the start of its output. Returns None when
/// no such token is present.
pub fn parse_language_token(raw: &str) -> Option<String> {
    let start = raw.find("<|")?;
    let rest = &raw[start + 2..];
    let end = rest.find("|>")?;
    let code = &rest[..end];

    let is_lang =
        (2..=3).contains(&code.len()) && code.chars().all(|c| c.is_ascii_lowercase());

    if is_lang {
        Some(code.to_string())
    } else {
        // A timestamp or task token: keep looking after it.
        parse_language_token(&rest[end + 2..])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The Arc<Ct2Asr> design in the Python layer requires this.
    /// If this fails to compile, Ct2Asr must wrap Whisper in a Mutex.
    #[test]
    fn whisper_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ct2rs::Whisper>();
    }

    #[test]
    fn parses_a_leading_language_token() {
        assert_eq!(parse_language_token("<|pt|><|0.00|> olá"), Some("pt".into()));
    }

    #[test]
    fn parses_a_three_letter_code() {
        assert_eq!(parse_language_token("<|yue|><|0.00|> hi"), Some("yue".into()));
    }

    #[test]
    fn skips_a_leading_timestamp_token() {
        assert_eq!(
            parse_language_token("<|0.00|><|en|> hello"),
            Some("en".into()),
            "a timestamp before the language token must be skipped"
        );
    }

    #[test]
    fn skips_task_tokens() {
        assert_eq!(
            parse_language_token("<|startoftranscript|><|de|><|transcribe|> hallo"),
            Some("de".into())
        );
    }

    #[test]
    fn plain_text_has_no_language_token() {
        assert_eq!(parse_language_token("just some text"), None);
    }

    #[test]
    fn empty_input_has_no_language_token() {
        assert_eq!(parse_language_token(""), None);
    }

    #[test]
    fn an_unterminated_token_is_not_a_language() {
        assert_eq!(parse_language_token("<|pt"), None);
    }
}

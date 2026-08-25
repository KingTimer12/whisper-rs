//! `polyvoice` diarization backend.
//!
//! Entirely behind the `diarization` feature: without it, `ort` is absent from
//! the dependency graph and this crate contains no ONNX runtime at all.

use super::{dylib, Diarizer, SpeakerTurn};
use crate::error::{Error, Result};
use crate::types::SAMPLE_RATE;

use polyvoice::clusterer::{Clusterer, ClustererError};

/// Wraps `polyvoice::kmeans::kmeans_pp` to force an exact cluster count.
///
/// `polyvoice`'s shipped automatic clustering (silhouette-driven k-means,
/// AS-norm-thresholded AHC) was measured (see task-0's validation gate) to
/// under-count badly: 3 speakers found in 5-speaker audio, on fixtures whose
/// embeddings are demonstrably well separated (pairwise cosine off-diagonal
/// 0.0176-0.6070). Forcing k via k-means++ on those same embeddings recovered
/// all 5 speakers exactly. So when the caller knows the speaker count, this
/// clusterer bypasses the weak automatic k-selection entirely rather than
/// trying to tune it.
struct ExactKClusterer {
    k: usize,
}

impl Clusterer for ExactKClusterer {
    fn cluster(&self, embeddings: &[Vec<f32>]) -> std::result::Result<Vec<usize>, ClustererError> {
        if embeddings.is_empty() {
            return Ok(Vec::new());
        }
        // Fewer segments than requested speakers is a legitimate input (e.g.
        // a short clip with one speaker but num_speakers requested as 3), not
        // a bug: `kmeans_pp` itself clamps k to embeddings.len(), so mirror
        // that here rather than erroring or panicking.
        let k = self.k.min(embeddings.len());
        Ok(polyvoice::kmeans::kmeans_pp(embeddings, k, 100))
    }

    fn max_clusters(&self) -> usize {
        self.k
    }
}

pub struct PolyvoiceDiarizer {
    pipeline: polyvoice::pipeline_v2::Pipeline,
}

impl PolyvoiceDiarizer {
    /// Build a diarizer bounded at `max_speakers`, optionally forcing an
    /// exact speaker count via `num_speakers`.
    ///
    /// `max_speakers` is an upper bound for clustering, not a model limit —
    /// which is the whole reason this backend was chosen over Sortformer,
    /// whose `NUM_SPEAKERS = 4` is fixed in the model.
    ///
    /// `num_speakers`, when `Some(k)`, forces exactly `k` clusters via a
    /// custom k-means clusterer instead of `polyvoice`'s automatic k
    /// selection. This exists because the automatic path under-counts (see
    /// `ExactKClusterer`'s doc comment) — it is not redundant with
    /// `max_speakers`, which only bounds the automatic search and does not
    /// fix its under-counting.
    pub fn new(max_speakers: usize, num_speakers: Option<usize>) -> Result<Self> {
        let bounded: u8 = u8::try_from(max_speakers)
            .ok()
            .filter(|n| *n >= 1)
            .ok_or_else(|| {
                Error::Diarize(format!(
                    "max_speakers must be in 1..=255, got {max_speakers}"
                ))
            })?;

        if let Some(k) = num_speakers {
            if k == 0 {
                return Err(Error::Diarize(
                    "num_speakers must be at least 1, got 0".to_string(),
                ));
            }
            if k > max_speakers {
                return Err(Error::Diarize(format!(
                    "num_speakers ({k}) cannot exceed max_speakers ({max_speakers})"
                )));
            }
        }

        dylib::init_ort()?;

        let registry = polyvoice::models::ModelRegistry::default()
            .map_err(|e| Error::Diarize(format!("model registry construction failed: {e}")))?;

        let mut builder = polyvoice::pipeline_v2::Pipeline::builder()
            .max_speakers(bounded)
            .with_models_from(registry);

        if let Some(k) = num_speakers {
            builder = builder.with_clusterer(Box::new(ExactKClusterer { k }));
        }

        let pipeline = builder
            .build()
            .map_err(|e| Error::Diarize(format!("building the pipeline failed: {e}")))?;

        Ok(Self { pipeline })
    }
}

impl Diarizer for PolyvoiceDiarizer {
    fn diarize(&self, samples: &[f32]) -> Result<Vec<SpeakerTurn>> {
        let sr = polyvoice::types::SampleRate::new(SAMPLE_RATE as u32).ok_or_else(|| {
            Error::Diarize(format!(
                "{SAMPLE_RATE} Hz is outside polyvoice's supported range"
            ))
        })?;

        let result = self
            .pipeline
            .run(samples, sr)
            .map_err(|e| Error::Diarize(e.to_string()))?;

        // polyvoice's TimeRange is f64; this crate's timeline is f32 seconds
        // throughout, so it narrows once here rather than forcing
        // mixed-precision comparisons into the per-word overlap arithmetic.
        Ok(result
            .turns
            .into_iter()
            .map(|t| SpeakerTurn {
                start: t.time.start as f32,
                end: t.time.end as f32,
                speaker: t.speaker.0 as usize,
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_speakers_must_fit_polyvoice_u8_range() {
        // polyvoice's builder takes u8, so 256 cannot be honoured. Truncating
        // it to 0 (or to 255) would silently transcribe under a limit the
        // caller did not ask for.
        let err = match PolyvoiceDiarizer::new(256, None) {
            Err(e) => e,
            Ok(_) => panic!("256 exceeds the range"),
        };
        assert!(err.to_string().contains("1..=255"), "got: {err}");
    }

    #[test]
    fn zero_max_speakers_is_rejected() {
        let err = match PolyvoiceDiarizer::new(0, None) {
            Err(e) => e,
            Ok(_) => panic!("0 speakers is meaningless"),
        };
        assert!(err.to_string().contains("1..=255"), "got: {err}");
    }

    #[test]
    fn zero_num_speakers_is_rejected() {
        let err = match PolyvoiceDiarizer::new(8, Some(0)) {
            Err(e) => e,
            Ok(_) => panic!("0 num_speakers is meaningless"),
        };
        assert!(err.to_string().contains("num_speakers"), "got: {err}");
    }

    #[test]
    fn num_speakers_cannot_exceed_max_speakers() {
        let err = match PolyvoiceDiarizer::new(4, Some(5)) {
            Err(e) => e,
            Ok(_) => panic!("num_speakers > max_speakers must error, not clamp"),
        };
        assert!(
            err.to_string().contains("num_speakers") && err.to_string().contains("max_speakers"),
            "got: {err}"
        );
    }

    #[test]
    fn exact_k_clusterer_handles_fewer_embeddings_than_k() {
        // Fewer segments than requested speakers is legitimate input, not a
        // bug: must not panic, and must still satisfy the trait's compact
        // 0..K numbering contract.
        let clusterer = ExactKClusterer { k: 5 };
        let embeddings = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let result = clusterer.cluster(&embeddings).expect("must not error");
        assert_eq!(result.len(), 2);
        let distinct = result.iter().collect::<std::collections::HashSet<_>>().len();
        assert!(result.iter().all(|&label| label < distinct));
    }

    #[test]
    fn exact_k_clusterer_handles_empty_embeddings() {
        let clusterer = ExactKClusterer { k: 5 };
        let result = clusterer.cluster(&[]).expect("must not error");
        assert!(result.is_empty());
    }
}

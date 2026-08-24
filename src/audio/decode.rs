//! symphonia-backed decoding to interleaved f32, downmixed to mono.
//!
//! Written against symphonia 0.6, whose `formats`/`codecs` APIs are a
//! significant reshape from 0.5: audio buffers are now generic
//! (`GenericAudioBufferRef`) rather than typed `AudioBufferRef`, decoders are
//! split by media kind (`AudioDecoder`/`VideoDecoder`/`SubtitleDecoder`), and
//! `FormatReader::next_packet` returns `Ok(None)` at end of stream instead of
//! signalling it via an IO error.

use crate::error::{Error, Result};
use std::fs::File;
use std::path::Path;
use symphonia::core::codecs::audio::{AudioDecoderOptions, CODEC_ID_NULL_AUDIO};
use symphonia::core::codecs::CodecParameters;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatOptions, FormatReader};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;

pub struct DecodedAudio {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

pub fn decode_file(path: &Path) -> Result<DecodedAudio> {
    let file = File::open(path).map_err(|e| Error::AudioRead {
        path: path.to_path_buf(),
        message: e.to_string(),
    })?;

    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let mut format: Box<dyn FormatReader> = symphonia::default::get_probe()
        .probe(&hint, mss, FormatOptions::default(), MetadataOptions::default())
        .map_err(|e| Error::AudioFormat {
            path: path.to_path_buf(),
            message: e.to_string(),
        })?;

    let track = format
        .tracks()
        .iter()
        .find(|t| match &t.codec_params {
            Some(CodecParameters::Audio(params)) => params.codec != CODEC_ID_NULL_AUDIO,
            _ => false,
        })
        .ok_or_else(|| Error::AudioEmpty {
            path: path.to_path_buf(),
        })?;
    let track_id = track.id;

    let audio_params = match &track.codec_params {
        Some(CodecParameters::Audio(params)) => params.clone(),
        _ => {
            return Err(Error::AudioEmpty {
                path: path.to_path_buf(),
            })
        }
    };

    let sample_rate = audio_params.sample_rate.ok_or_else(|| Error::AudioFormat {
        path: path.to_path_buf(),
        message: "stream declares no sample rate".into(),
    })?;

    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(&audio_params, &AudioDecoderOptions::default())
        .map_err(|e| Error::AudioFormat {
            path: path.to_path_buf(),
            message: e.to_string(),
        })?;

    let mut out: Vec<f32> = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(Some(p)) => p,
            // End of stream.
            Ok(None) => break,
            Err(symphonia::core::errors::Error::IoError(_)) => break,
            Err(symphonia::core::errors::Error::ResetRequired) => break,
            Err(e) => {
                return Err(Error::AudioRead {
                    path: path.to_path_buf(),
                    message: e.to_string(),
                })
            }
        };

        if packet.track_id != track_id {
            continue;
        }

        match decoder.decode(&packet) {
            Ok(buf) => {
                let channels = buf.spec().channels().count().max(1);
                let mut interleaved: Vec<f32> = Vec::new();
                buf.copy_to_vec_interleaved(&mut interleaved);
                for frame in interleaved.chunks(channels) {
                    let sum: f32 = frame.iter().sum();
                    out.push(sum / channels as f32);
                }
            }
            // A corrupt packet mid-file should not lose the whole file.
            Err(symphonia::core::errors::Error::DecodeError(_)) => continue,
            Err(e) => {
                return Err(Error::AudioRead {
                    path: path.to_path_buf(),
                    message: e.to_string(),
                })
            }
        }
    }

    if out.is_empty() {
        return Err(Error::AudioEmpty {
            path: path.to_path_buf(),
        });
    }

    Ok(DecodedAudio {
        samples: out,
        sample_rate,
    })
}

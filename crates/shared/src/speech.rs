use std::pin::Pin;
use std::task::{Context, Poll};

use axum::http::StatusCode;
use base64::{Engine, engine::general_purpose::STANDARD as BASE64};
use futures::Stream;
use serde::{Deserialize, Serialize};
use serde_json::json;
use utoipa::ToSchema;

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct AudioSpec {
  pub channels: u16,
  pub sample_rate: u32,
}

impl AudioSpec {
  #[must_use]
  pub const fn new(channels: u16, sample_rate: u32) -> Self {
    Self {
      channels,
      sample_rate,
    }
  }
}

pub struct SpeechAudioStream {
  spec: AudioSpec,
  stream: Pin<Box<dyn Stream<Item = Result<Vec<f32>, crate::AppError>> + Send>>,
}

impl SpeechAudioStream {
  pub fn new<S>(spec: AudioSpec, stream: S) -> Self
  where
    S: Stream<Item = Result<Vec<f32>, crate::AppError>> + Send + 'static,
  {
    Self {
      spec,
      stream: Box::pin(stream),
    }
  }

  #[must_use]
  pub const fn spec(&self) -> AudioSpec {
    self.spec
  }
}

impl Stream for SpeechAudioStream {
  type Item = Result<Vec<f32>, crate::AppError>;

  fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
    self.get_mut().stream.as_mut().poll_next(cx)
  }
}

#[must_use]
pub fn pcm_bytes(samples: &[f32]) -> Vec<u8> {
  samples
    .iter()
    .flat_map(|sample| sample.to_le_bytes())
    .collect()
}

#[must_use]
pub fn wav_header(audio_spec: AudioSpec) -> Vec<u8> {
  let bytes_per_sample = 4_u32;
  let block_align = audio_spec.channels as u32 * bytes_per_sample;
  let byte_rate = audio_spec.sample_rate * block_align;
  let mut header = Vec::with_capacity(44);
  header.extend_from_slice(b"RIFF");
  header.extend_from_slice(&u32::MAX.to_le_bytes());
  header.extend_from_slice(b"WAVEfmt ");
  header.extend_from_slice(&16_u32.to_le_bytes());
  header.extend_from_slice(&3_u16.to_le_bytes());
  header.extend_from_slice(&audio_spec.channels.to_le_bytes());
  header.extend_from_slice(&audio_spec.sample_rate.to_le_bytes());
  header.extend_from_slice(&byte_rate.to_le_bytes());
  header.extend_from_slice(&(block_align as u16).to_le_bytes());
  header.extend_from_slice(&32_u16.to_le_bytes());
  header.extend_from_slice(b"data");
  header.extend_from_slice(&u32::MAX.to_le_bytes());
  header
}

#[derive(Debug, Clone, Copy, Deserialize, Eq, PartialEq, Serialize, ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum StreamFormat {
  Audio,
  Sse,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct SpeechAudioDeltaEvent {
  #[serde(rename = "type")]
  pub event_type: String,
  pub audio: String,
}

impl SpeechAudioDeltaEvent {
  #[must_use]
  pub fn new(bytes: &[u8]) -> Self {
    Self {
      event_type: String::from("speech.audio.delta"),
      audio: BASE64.encode(bytes),
    }
  }
}

#[derive(Debug, Default, Serialize, ToSchema)]
pub struct SpeechUsage {
  pub input_tokens: u32,
  pub output_tokens: u32,
  pub total_tokens: u32,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct SpeechAudioDoneEvent {
  #[serde(rename = "type")]
  pub event_type: String,
  pub usage: SpeechUsage,
}

impl SpeechAudioDoneEvent {
  #[must_use]
  pub fn new() -> Self {
    Self {
      event_type: String::from("speech.audio.done"),
      usage: SpeechUsage::default(),
    }
  }
}

#[derive(Debug, Serialize, ToSchema)]
#[serde(untagged)]
pub enum SpeechAudioStreamEvent {
  Delta(SpeechAudioDeltaEvent),
  Done(SpeechAudioDoneEvent),
}

/// Request body
#[derive(Debug, Deserialize, ToSchema)]
pub struct SpeechOptions {
  /// The text to generate audio for.
  pub input: String,
  /// One of the available TTS models: `chatterbox-multilingual`.
  pub model: String,
  /// The voice to use when generating the audio.
  pub voice: String, // TODO: instructions
  // TODO: response_format
  // TODO: speed
  /// The format to stream the audio in. Defaults to `audio` when omitted.
  #[schema(default = "audio")]
  pub stream_format: Option<StreamFormat>,
}

impl SpeechOptions {
  pub const MAX_INPUT_CHARS: usize = 4096;

  pub fn validate(&self) -> Result<(), crate::AppError> {
    if self.input.trim().is_empty() {
      return Err(crate::AppError::new(
        String::from("Input must not be empty"),
        String::from("invalid_request_error"),
        Some(StatusCode::BAD_REQUEST),
        Some(json!("input")),
        None,
      ));
    }
    if self.input.chars().count() > Self::MAX_INPUT_CHARS {
      return Err(crate::AppError::new(
        format!("Input must not exceed {} characters", Self::MAX_INPUT_CHARS),
        String::from("invalid_request_error"),
        Some(StatusCode::BAD_REQUEST),
        Some(json!("input")),
        None,
      ));
    }
    Ok(())
  }
}

#[derive(Debug, ToSchema)]
#[schema(value_type = String, format = Binary)]
#[allow(dead_code)]
/// OpenAPI schema marker for the binary audio response.
pub struct SpeechAudio(Vec<u8>);

#[cfg(test)]
mod tests {
  use serde_json::json;

  use super::{SpeechAudioDeltaEvent, SpeechAudioDoneEvent, SpeechOptions, StreamFormat};

  #[test]
  fn stream_format_accepts_audio_and_sse() {
    let default_options: SpeechOptions = serde_json::from_value(json!({
      "input": "hello",
      "model": "kokoro",
      "voice": "af_heart"
    }))
    .unwrap();
    assert_eq!(default_options.stream_format, None);

    let audio_options: SpeechOptions = serde_json::from_value(json!({
      "input": "hello",
      "model": "kokoro",
      "voice": "af_heart",
      "stream_format": "audio"
    }))
    .unwrap();
    assert_eq!(audio_options.stream_format, Some(StreamFormat::Audio));

    let sse_options: SpeechOptions = serde_json::from_value(json!({
      "input": "hello",
      "model": "kokoro",
      "voice": "af_heart",
      "stream_format": "sse"
    }))
    .unwrap();
    assert_eq!(sse_options.stream_format, Some(StreamFormat::Sse));
  }

  #[test]
  fn speech_events_match_openai_shapes() {
    assert_eq!(
      serde_json::to_value(SpeechAudioDeltaEvent::new(&[0, 1, 2, 255])).unwrap(),
      json!({
        "type": "speech.audio.delta",
        "audio": "AAEC/w=="
      })
    );
    assert_eq!(
      serde_json::to_value(SpeechAudioDoneEvent::new()).unwrap(),
      json!({
        "type": "speech.audio.done",
        "usage": {
          "input_tokens": 0,
          "output_tokens": 0,
          "total_tokens": 0
        }
      })
    );
  }

  #[test]
  fn speech_options_reject_empty_and_overlong_input() {
    let mut options: SpeechOptions = serde_json::from_value(json!({
      "input": " ",
      "model": "kokoro",
      "voice": "af_heart"
    }))
    .unwrap();
    assert!(options.validate().is_err());

    options.input = "a".repeat(SpeechOptions::MAX_INPUT_CHARS + 1);
    assert!(options.validate().is_err());
  }
}

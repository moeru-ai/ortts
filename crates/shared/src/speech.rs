use std::{
  io::Cursor,
  pin::Pin,
  task::{Context, Poll},
};

use axum::{
  http::StatusCode,
  response::{IntoResponse, Response},
};
use axum_extra::TypedHeader;
use base64::{Engine, engine::general_purpose::STANDARD as BASE64};
use futures::{Stream, StreamExt};
use headers::{ContentLength, ContentType, Mime};
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

pub struct SpeechStream {
  spec: AudioSpec,
  stream: Pin<Box<dyn Stream<Item = Result<Vec<f32>, crate::AppError>> + Send>>,
}

impl SpeechStream {
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

impl Stream for SpeechStream {
  type Item = Result<Vec<f32>, crate::AppError>;

  fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
    self.get_mut().stream.as_mut().poll_next(cx)
  }
}

pub async fn collect_speech_stream(
  mut speech_stream: SpeechStream,
) -> Result<Vec<u8>, crate::AppError> {
  let mut samples = Vec::new();
  while let Some(chunk) = speech_stream.next().await {
    samples.extend(chunk?);
  }

  encode_wav(&samples, speech_stream.spec)
}

fn encode_wav(samples: &[f32], audio_spec: AudioSpec) -> Result<Vec<u8>, crate::AppError> {
  let spec = hound::WavSpec {
    channels: audio_spec.channels,
    sample_rate: audio_spec.sample_rate,
    bits_per_sample: 32,
    sample_format: hound::SampleFormat::Float,
  };
  let mut buffer = Cursor::new(Vec::new());
  let mut writer = hound::WavWriter::new(&mut buffer, spec)?;
  for sample in samples {
    writer.write_sample(*sample)?;
  }
  writer.finalize()?;

  Ok(buffer.into_inner())
}

#[must_use]
pub fn pcm_bytes(samples: &[f32]) -> Vec<u8> {
  samples
    .iter()
    .flat_map(|sample| sample.to_le_bytes())
    .collect()
}

#[must_use]
pub fn wav_stream_header(audio_spec: AudioSpec) -> Vec<u8> {
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
  /// Omit this field for a complete response, or select `audio` or `sse` for streaming.
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
pub struct SpeechResult(Vec<u8>);

impl SpeechResult {
  #[must_use]
  pub const fn new(bytes: Vec<u8>) -> Self {
    Self(bytes)
  }
}

impl IntoResponse for SpeechResult {
  fn into_response(self) -> Response {
    // TODO: custom mime type
    let mime = "audio/wav".parse::<Mime>().unwrap();
    let content_type = TypedHeader(ContentType::from(mime));
    let content_length = TypedHeader(ContentLength(self.0.len() as u64));

    (StatusCode::OK, content_type, content_length, self.0).into_response()
  }
}

#[cfg(test)]
mod tests {
  use futures::stream;
  use serde_json::json;

  use super::{
    AudioSpec, SpeechAudioDeltaEvent, SpeechAudioDoneEvent, SpeechOptions, SpeechStream,
    StreamFormat, collect_speech_stream,
  };

  #[tokio::test]
  async fn collecting_audio_chunks_produces_a_wav_file() {
    let stream = SpeechStream::new(
      AudioSpec::new(1, 24_000),
      stream::iter([Ok(vec![0.0_f32, 0.25]), Ok(vec![-0.5_f32, 1.0])]),
    );

    let wav = collect_speech_stream(stream).await.unwrap();
    let reader = hound::WavReader::new(std::io::Cursor::new(wav)).unwrap();

    assert_eq!(reader.spec().channels, 1);
    assert_eq!(reader.spec().sample_rate, 24_000);
    assert_eq!(
      reader
        .into_samples::<f32>()
        .collect::<Result<Vec<_>, _>>()
        .unwrap(),
      vec![0.0, 0.25, -0.5, 1.0]
    );
  }

  #[test]
  fn stream_format_preserves_omitted_audio_and_sse() {
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

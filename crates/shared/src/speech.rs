use axum::{
  http::StatusCode,
  response::{IntoResponse, Response},
};
use axum_extra::TypedHeader;
use headers::{ContentLength, ContentType, Mime};
use serde::Deserialize;
use utoipa::ToSchema;

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
                     // TODO: stream_format
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

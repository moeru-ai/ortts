use axum::{
  Json, debug_handler,
  http::StatusCode,
  response::{IntoResponse, Response},
};
use ortts_shared::AppError;
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
pub struct SpeechResult(String);

impl IntoResponse for SpeechResult {
  fn into_response(self) -> Response {
    (StatusCode::OK, self.0).into_response()
  }
}

/// Create speech
///
/// Generates audio from the input text.
#[utoipa::path(
  get,
  path = "/v1/audio/speech",
  responses(
    (status = 200, body = SpeechResult)
  )
)]
#[debug_handler]
pub async fn speech(Json(_options): Json<SpeechOptions>) -> Result<SpeechResult, AppError> {
  Ok(SpeechResult(String::from("TODO")))
}

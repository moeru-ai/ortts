use axum::{
  Json, debug_handler,
  http::StatusCode,
  response::{IntoResponse, Response},
};
use axum_extra::TypedHeader;
use headers::{ContentLength, ContentType, Mime};
use ortts_shared::AppError;
use serde::Deserialize;
use utoipa::ToSchema;

use crate::utils::inference;

/// Request body
#[derive(Debug, Deserialize, ToSchema)]
pub struct ChatterboxMultilingualSpeechOptions {
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
pub struct ChatterboxMultilingualSpeechResult(Vec<u8>);

impl IntoResponse for ChatterboxMultilingualSpeechResult {
  fn into_response(self) -> Response {
    let mime = "audio/wav".parse::<Mime>().unwrap();
    let content_type = TypedHeader(ContentType::from(mime));
    let content_length = TypedHeader(ContentLength(self.0.len() as u64));

    (StatusCode::OK, content_type, content_length, self.0).into_response()
  }
}

/// Create speech (Chatterbox Multilingual)
///
/// Generates audio from the input text.
#[utoipa::path(
  get,
  path = "/v0/chatterbox-multilingual/audio/speech",
  responses(
    (status = 200, body = ChatterboxMultilingualSpeechResult)
  )
)]
#[debug_handler]
pub async fn speech(
  Json(options): Json<ChatterboxMultilingualSpeechOptions>,
) -> Result<ChatterboxMultilingualSpeechResult, AppError> {
  let bytes = inference(options.input).await?;

  Ok(ChatterboxMultilingualSpeechResult(bytes))
}

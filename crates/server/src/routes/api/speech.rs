use axum::{Json, debug_handler};
use ortts_shared::AppError;

use ortts_shared::{SpeechOptions, SpeechResult};

/// Create speech
///
/// Generates audio from the input text.
#[utoipa::path(
  post,
  path = "/v1/audio/speech",
  responses(
    (status = 200, body = SpeechResult)
  )
)]
#[debug_handler]
pub async fn speech(Json(_options): Json<SpeechOptions>) -> Result<SpeechResult, AppError> {
  Ok(SpeechResult::new(Vec::new()))
}

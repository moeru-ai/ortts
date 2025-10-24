use axum::{Json, debug_handler};
use ortts_shared::{AppError, SpeechOptions, SpeechResult};

use crate::utils::inference;

/// Create speech (Chatterbox Multilingual)
///
/// Generates audio from the input text.
#[utoipa::path(
  post,
  path = "/v0/chatterbox-multilingual/audio/speech",
  responses(
    (status = 200, body = SpeechResult)
  )
)]
#[debug_handler]
pub async fn speech(Json(options): Json<SpeechOptions>) -> Result<SpeechResult, AppError> {
  Ok(SpeechResult::new(inference(options).await?))
}

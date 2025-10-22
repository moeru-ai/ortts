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
pub async fn speech(Json(options): Json<SpeechOptions>) -> Result<SpeechResult, AppError> {
  match options.model {
    m if m.starts_with("chatterbox-multilingual") => Ok(SpeechResult::new(
      ortts_model_chatterbox_multilingual::inference(options.input).await?,
    )),
    _ => todo!(),
  }
}

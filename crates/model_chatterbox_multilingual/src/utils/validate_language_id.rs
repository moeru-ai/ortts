use anyhow::anyhow;
use ortts_shared::AppError;

/// https://github.com/resemble-ai/chatterbox/blob/bf169fe5f518760cb0b6c6a6eba3f885e10fa86f/src/chatterbox/mtl_tts.py#L24-L48
const SUPPORTED_LANGUAGE_IDS: [&str; 23] = [
  "ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi", "it", "ja", "ko", "ms", "nl", "no",
  "pl", "pt", "ru", "sv", "sw", "tr", "zh",
];

pub fn validate_language_id(model: &str) -> Result<String, AppError> {
  model
    .rsplit_once('/')
    .map(|(_, language_id)| language_id)
    .map_or_else(
      // use English by default
      || Ok(String::from("en")),
      |language_id| {
        let lowercase_id = language_id.to_lowercase();
        let is_supported = SUPPORTED_LANGUAGE_IDS
          .iter()
          .any(|&supported_id| supported_id == lowercase_id.as_str());

        if is_supported {
          Ok(lowercase_id)
        } else {
          Err(AppError::anyhow(&anyhow!(
            "Unsupported language_id '{}'. Supported languages: {}",
            lowercase_id,
            SUPPORTED_LANGUAGE_IDS.join(", ")
          )))
        }
      },
    )
}

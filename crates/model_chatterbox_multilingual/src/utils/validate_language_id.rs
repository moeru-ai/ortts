use anyhow::anyhow;
use ortts_shared::AppError;

/// https://github.com/resemble-ai/chatterbox/blob/bf169fe5f518760cb0b6c6a6eba3f885e10fa86f/src/chatterbox/mtl_tts.py#L24-L48
const SUPPORTED_LANGUAGE_IDS: [&str; 23] = [
  "ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi", "it", "ja", "ko", "ms", "nl", "no",
  "pl", "pt", "ru", "sv", "sw", "tr", "zh",
];

pub fn validate_language_id(model: String) -> Result<String, AppError> {
  match model.rsplit_once('/').map(|(_, language_id)| language_id) {
    Some(language_id) => {
      let lowercase_id = language_id.to_lowercase();
      let is_supported = SUPPORTED_LANGUAGE_IDS
        .iter()
        .any(|&supported_id| supported_id == lowercase_id.as_str());

      match is_supported {
        true => Ok(lowercase_id),
        false => Err(AppError::anyhow(&anyhow!(
          "Unsupported language_id '{}'. Supported languages: {}",
          lowercase_id,
          SUPPORTED_LANGUAGE_IDS.join(", ")
        ))),
      }
    }
    // use English by default
    None => Ok(String::from("en")),
  }
}

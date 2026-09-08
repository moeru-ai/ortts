use anyhow::anyhow;
use ortts_shared::AppError;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Language {
  Auto,
  Chinese,
  English,
  Japanese,
}

impl Language {
  pub fn from_model_name(model: &str) -> Result<Self, AppError> {
    let model = model.to_ascii_lowercase();
    let suffix = model
      .strip_prefix("qwen3-tts-base")
      .ok_or_else(|| anyhow!("invalid Qwen3-TTS Base model name: {model}"))?;

    match suffix {
      "" | ":auto" => Ok(Self::Auto),
      ":zh" | ":chinese" => Ok(Self::Chinese),
      ":en" | ":english" => Ok(Self::English),
      ":ja" | ":jp" | ":japanese" => Ok(Self::Japanese),
      _ => Err(anyhow!(
        "unsupported Qwen3-TTS Base model name '{model}'; use qwen3-tts-base or qwen3-tts-base:zh, :en, or :ja"
      )
      .into()),
    }
  }

  pub const fn config_key(self) -> Option<&'static str> {
    match self {
      Self::Auto => None,
      Self::Chinese => Some("chinese"),
      Self::English => Some("english"),
      Self::Japanese => Some("japanese"),
    }
  }
}

#[cfg(test)]
mod tests {
  use super::Language;

  #[test]
  fn parses_supported_model_suffixes() {
    assert_eq!(
      Language::from_model_name("qwen3-tts-base:zh").unwrap(),
      Language::Chinese
    );
    assert_eq!(
      Language::from_model_name("QWEN3-TTS-BASE:JA").unwrap(),
      Language::Japanese
    );
    assert_eq!(
      Language::from_model_name("qwen3-tts-base").unwrap(),
      Language::Auto
    );
    assert_eq!(
      Language::from_model_name("qwen3-tts-base:auto").unwrap(),
      Language::Auto
    );
  }

  #[test]
  fn rejects_unknown_language_suffix() {
    assert!(Language::from_model_name("qwen3-tts-base:fr").is_err());
    assert!(Language::from_model_name("qwen3-tts-base-zh").is_err());
    assert!(Language::from_model_name("qwen3-tts-base::zh").is_err());
  }
}

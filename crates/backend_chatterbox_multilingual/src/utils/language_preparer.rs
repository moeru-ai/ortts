use ortts_shared::AppError;

use crate::utils::{ChineseCangjieConverter, hiragana_normalize};

pub struct LanguagePreparer {
  cangjie_converter: ChineseCangjieConverter,
}

impl LanguagePreparer {
  pub async fn new() -> Result<Self, AppError> {
    Ok(Self {
      cangjie_converter: ChineseCangjieConverter::new().await?,
    })
  }

  pub fn prepare(&self, text: String, language_id: &str) -> String {
    let text = match language_id {
      "zh" => self.cangjie_converter.convert(&text),
      "ja" => hiragana_normalize(&text),
      // TODO: add_hebrew_diacritics
      "he" => todo!(),
      // TODO: korean_normalize
      "ko" => todo!(),
      _ => text,
    };

    format!("[{language_id}]{text}")
  }
}

use ortts_shared::AppError;

use super::chinese_cangjie_converter::ChineseCangjieConverter;

pub struct LanguagePreparer {
  cangjie_converter: ChineseCangjieConverter,
}

impl LanguagePreparer {
  pub async fn new() -> Result<Self, AppError> {
    Ok(Self {
      cangjie_converter: ChineseCangjieConverter::new().await?,
    })
  }

  pub async fn prepare(
    &self,
    text: String,
    language_id: Option<String>,
  ) -> Result<String, AppError> {
    if let Some(language_id) = language_id {
      let text = match language_id.as_str() {
        "zh" => self.cangjie_converter.convert(&text),
        // TODO: hiragana_normalize
        "ja" => todo!(),
        // TODO: add_hebrew_diacritics
        "he" => todo!(),
        // TODO: korean_normalize
        "ko" => todo!(),
        _ => text,
      };

      Ok(format!("[{}]{}", language_id, text))
    } else {
      Ok(text)
    }
  }
}

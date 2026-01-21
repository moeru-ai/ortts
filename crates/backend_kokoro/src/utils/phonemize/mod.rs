// Rust Rewrite of https://github.com/hexgrad/kokoro/blob/dfb907a02bba8152ca444717ca5d78747ccb4bec/kokoro.js/src/phonemize.js

mod escape_reg_exp;
pub use escape_reg_exp::escape_reg_exp;

mod flip_money;
pub use flip_money::flip_money;

mod normalize_text;
pub use normalize_text::normalize_text;

mod point_num;
use ortts_shared::AppError;
pub use point_num::point_num;

mod split;
pub use split::split;

mod split_num;
pub use split_num::split_num;

use regex::Regex;
use std::sync::OnceLock;

static PUNCTUATION_PATTERN: OnceLock<Regex> = OnceLock::new();

fn get_punctuation_pattern() -> &'static Regex {
  PUNCTUATION_PATTERN.get_or_init(|| {
    const PUNCTUATION: &str = ";:,.!?¡¿—…\"«»“”(){}[]";
    let escaped_punctuation = escape_reg_exp(PUNCTUATION);
    let pattern_str = format!(r"(\s*[{escaped_punctuation}]+\s*)+");
    Regex::new(&pattern_str).unwrap()
  })
}

/// Phonemize text using the eSpeak-NG phonemizer
pub async fn phonemize(mut text: String, normalization: bool) -> Result<String, AppError> {
  if normalization {
    text = normalize_text(&text);
  }

  let pattern = get_punctuation_pattern();
  let sections = split(&text, pattern);

  let ps: Vec<_> = sections
    .into_iter()
    .map(|(is_match, text)| async move {
      if is_match {
        text
      } else {
        espeak_rs::text_to_phonemes(&text, "en", None, true, false)
          .unwrap_or_default()
          .join("")
      }
    })
    .collect();

  let ps: Vec<String> = futures::future::join_all(ps).await;
  let processed = ps.join("");

  // TODO: 4. Post-process phonemes

  // TODO: 5. Additional post-processing for American English

  Ok(processed.trim().to_string())
}

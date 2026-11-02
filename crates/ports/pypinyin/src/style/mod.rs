//! Port of `pypinyin/style/__init__.py`: the style registry.
//!
//! Python registers style functions in a dict at import time
//! (`register`, `auto_discover`). The port dispatches with a `match` over
//! the ported [`Style`] variants instead; custom registration is not ported.

pub mod _constants;
pub mod _tone_convert;
pub mod _tone_rule;
pub mod _utils;
pub mod finals;
pub mod initials;
pub mod others;
pub mod tone;

use crate::constants::Style;

/// Port of `pypinyin.style.convert`: convert one toned pinyin to `style`.
///
/// `default` is what Python returns for a style without a registered
/// implementation. Every ported style has one, so it is never returned. `han`
/// is the original character, which no ported style uses.
pub fn convert(pinyin: &str, style: Style, strict: bool, default: &str, han: &str) -> String {
  let _ = (default, han);
  match style {
    Style::NORMAL => others::converter.to_normal(pinyin),
    Style::TONE => tone::converter.to_tone(pinyin),
    Style::TONE2 => tone::converter.to_tone2(pinyin),
    Style::TONE3 => tone::converter.to_tone3(pinyin),
    Style::INITIALS => initials::convert(pinyin, strict),
    Style::FIRST_LETTER => others::converter.to_first_letter(pinyin),
    Style::FINALS => finals::converter.to_finals(pinyin, strict),
    Style::FINALS_TONE => finals::converter.to_finals_tone(pinyin, strict),
    Style::FINALS_TONE2 => finals::converter.to_finals_tone2(pinyin, strict),
    Style::FINALS_TONE3 => finals::converter.to_finals_tone3(pinyin, strict),
  }
}

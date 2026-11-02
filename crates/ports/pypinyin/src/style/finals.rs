//! Port of `pypinyin/style/finals.py`: `Style.FINALS`, `Style.FINALS_TONE`,
//! `Style.FINALS_TONE2` and `Style.FINALS_TONE3`.

use crate::style::_tone_convert::{to_finals, to_finals_tone, to_finals_tone2, to_finals_tone3};

/// Port of `pypinyin.style.finals.FinalsConverter`.
#[derive(Clone, Copy, Debug, Default)]
pub struct FinalsConverter;

impl FinalsConverter {
  /// Port of `FinalsConverter.to_finals`: 无声调韵母.
  pub fn to_finals(&self, pinyin: &str, strict: bool) -> String {
    to_finals(pinyin, strict, false)
  }

  /// Port of `FinalsConverter.to_finals_tone`: 声调在韵母头上.
  pub fn to_finals_tone(&self, pinyin: &str, strict: bool) -> String {
    to_finals_tone(pinyin, strict)
  }

  /// Port of `FinalsConverter.to_finals_tone2`: 数字声调.
  pub fn to_finals_tone2(&self, pinyin: &str, strict: bool) -> String {
    to_finals_tone2(pinyin, strict, false, false)
  }

  /// Port of `FinalsConverter.to_finals_tone3`: 数字声调.
  pub fn to_finals_tone3(&self, pinyin: &str, strict: bool) -> String {
    to_finals_tone3(pinyin, strict, false, false)
  }
}

/// Port of `pypinyin.style.finals.converter`.
#[allow(non_upper_case_globals)]
pub const converter: FinalsConverter = FinalsConverter;

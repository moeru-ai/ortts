//! Port of `pypinyin/style/tone.py`: `Style.TONE`, `Style.TONE2` and
//! `Style.TONE3`.

use crate::style::_constants::re_tone3_sub;
use crate::style::_utils::replace_symbol_to_number;

/// Port of `pypinyin.style.tone.ToneConverter`.
#[derive(Clone, Copy, Debug, Default)]
pub struct ToneConverter;

impl ToneConverter {
  /// Port of `ToneConverter.to_tone`.
  pub fn to_tone(&self, pinyin: &str) -> String {
    pinyin.to_string()
  }

  /// Port of `ToneConverter.to_tone2`: 用数字表示声调.
  pub fn to_tone2(&self, pinyin: &str) -> String {
    replace_symbol_to_number(pinyin)
  }

  /// Port of `ToneConverter.to_tone3`: 将声调数字移动到最后.
  pub fn to_tone3(&self, pinyin: &str) -> String {
    let pinyin = self.to_tone2(pinyin);
    re_tone3_sub(&pinyin)
  }
}

/// Port of `pypinyin.style.tone.converter`.
#[allow(non_upper_case_globals)]
pub const converter: ToneConverter = ToneConverter;

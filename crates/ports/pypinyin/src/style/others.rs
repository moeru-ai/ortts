//! Port of `pypinyin/style/others.py`: `Style.NORMAL` and
//! `Style.FIRST_LETTER`.

use crate::style::_utils::replace_symbol_to_no_symbol;

/// Port of `pypinyin.style.others.OthersConverter`.
#[derive(Clone, Copy, Debug, Default)]
pub struct OthersConverter;

impl OthersConverter {
  /// Port of `OthersConverter.to_normal`.
  pub fn to_normal(&self, pinyin: &str) -> String {
    replace_symbol_to_no_symbol(pinyin)
  }

  /// Port of `OthersConverter.to_first_letter`.
  ///
  /// Panics on an empty pinyin, like the `IndexError` of `pinyin[0]`.
  pub fn to_first_letter(&self, pinyin: &str) -> String {
    let pinyin = self.to_normal(pinyin);
    pinyin
      .chars()
      .next()
      .expect("string index out of range")
      .to_string()
  }
}

/// Port of `pypinyin.style.others.converter`.
#[allow(non_upper_case_globals)]
pub const converter: OthersConverter = OthersConverter;

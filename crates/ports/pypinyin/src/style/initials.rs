//! Port of `pypinyin/style/initials.py`: `Style.INITIALS`.

use crate::style::_utils::get_initials;

/// Port of `pypinyin.style.initials.convert`.
pub fn convert(pinyin: &str, strict: bool) -> String {
  get_initials(pinyin, strict)
}

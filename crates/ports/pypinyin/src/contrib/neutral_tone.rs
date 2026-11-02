//! Port of `pypinyin/contrib/neutral_tone.py`.

use crate::_re_digit::is_digit;
use crate::constants::Style;
use crate::contrib::_tone_rule::right_mark_index;

/// Port of `pypinyin.contrib.neutral_tone.NeutralToneWith5Mixin` as used by
/// `pypinyin.converter._neutralToneWith5Converter`
/// (`NeutralToneWith5Mixin` over `DefaultConverter`): mark the neutral tone
/// with `5` in the number tone styles.
#[derive(Clone, Copy, Debug, Default)]
pub struct NeutralToneWith5Mixin;

impl NeutralToneWith5Mixin {
  /// Port of `NeutralToneWith5Mixin.NUMBER_TONE`.
  pub const NUMBER_TONE: &[Style] = &[
    Style::TONE2,
    Style::TONE3,
    Style::FINALS_TONE2,
    Style::FINALS_TONE3,
  ];
  /// Port of `NeutralToneWith5Mixin.NUMBER_AT_END`.
  pub const NUMBER_AT_END: &[Style] = &[Style::TONE3, Style::FINALS_TONE3];

  /// Port of `NeutralToneWith5Mixin.post_convert_style`. The `super()` call
  /// resolves to `DefaultConverter.post_convert_style`, which returns
  /// `None`.
  ///
  /// Panics when a number tone style without a tone has no vowel to mark,
  /// like the `TypeError` of `None + 1` in Python.
  pub fn post_convert_style(
    &self,
    _han: &str,
    _orig_pinyin: &str,
    converted_pinyin: &str,
    style: Style,
    _strict: bool,
  ) -> Option<String> {
    let pre_data: Option<String> = None;

    if !Self::NUMBER_TONE.contains(&style) {
      return pre_data;
    }

    if converted_pinyin.is_empty() {
      // 空字符串
      return Some(converted_pinyin.to_string());
    }
    // 有声调，跳过
    if converted_pinyin.chars().any(is_digit) {
      return Some(converted_pinyin.to_string());
    }

    if Self::NUMBER_AT_END.contains(&style) {
      return Some(format!("{converted_pinyin}5"));
    }

    // 找到应该在哪个字母上标声调
    let mark_index = right_mark_index(converted_pinyin)
      .expect("unsupported operand type(s) for +: 'NoneType' and 'int'");
    let chars: Vec<char> = converted_pinyin.chars().collect();
    let split = (mark_index + 1).min(chars.len());
    let before: String = chars[..split].iter().collect();
    let after: String = chars[split..].iter().collect();

    Some(format!("{before}5{after}"))
  }
}

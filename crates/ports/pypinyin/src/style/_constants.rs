//! Port of `pypinyin/style/_constants.py`.
//!
//! The regular expressions of the Python module are written out by hand
//! where they are used: `RE_PHONETIC_SYMBOL` in
//! [`replace_symbol_to_number`](super::_utils::replace_symbol_to_number),
//! `RE_TONE2` in
//! [`tone2_to_tone`](super::_tone_convert::tone2_to_tone), `RE_TONE3` in
//! [`re_tone3_sub`] and `RE_NUMBER` in [`crate::_re_digit`].

use crate::standard::split_py_dollar;

/// Port of `pypinyin.style._constants._INITIALS` (声母表).
pub const _INITIALS: &[&str] = &[
  "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h", "j", "q", "x", "zh", "ch", "sh", "r", "z",
  "c", "s",
];

/// Port of `pypinyin.style._constants._INITIALS_NOT_STRICT` (`_INITIALS + ['y', 'w']`).
pub const _INITIALS_NOT_STRICT: &[&str] = &[
  "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h", "j", "q", "x", "zh", "ch", "sh", "r", "z",
  "c", "s", "y", "w",
];

/// Port of `pypinyin.style._constants._FINALS` (韵母表).
pub const _FINALS: &[&str] = &[
  "i", "u", "ü", "a", "ia", "ua", "o", "uo", "e", "ie", "üe", "ai", "uai", "ei", "uei", "ao",
  "iao", "ou", "iou", "an", "ian", "uan", "üan", "en", "in", "uen", "ün", "ang", "iang", "uang",
  "eng", "ing", "ueng", "ong", "iong", "er", "ê",
];

/// Character class `[a-zêü]` used by `RE_TONE3`.
fn is_tone3_letter(c: char) -> bool {
  c.is_ascii_lowercase() || c == 'ê' || c == 'ü'
}

/// Port of `RE_TONE3.sub(r'\1\3\2', pinyin)` with
/// `RE_TONE3 = ^([a-zêü]+)([1-5])([a-zêü]*)$`: move the tone number to the end.
pub fn re_tone3_sub(pinyin: &str) -> String {
  let (body, tail) = split_py_dollar(pinyin);
  let chars: Vec<char> = body.chars().collect();
  let Some(i) = chars.iter().position(|c| !is_tone3_letter(*c)) else {
    return pinyin.to_string();
  };
  if i == 0
    || !('1'..='5').contains(&chars[i])
    || !chars[i + 1..].iter().all(|c| is_tone3_letter(*c))
  {
    return pinyin.to_string();
  }
  let mut out: String = chars[..i].iter().collect();
  out.extend(&chars[i + 1..]);
  out.push(chars[i]);
  out.push_str(tail);
  out
}

/// Character class `[aeoiuvnmêü]` used by `RE_TONE2`.
pub(crate) fn is_tone2_vowel(c: char) -> bool {
  matches!(c, 'a' | 'e' | 'o' | 'i' | 'u' | 'v' | 'n' | 'm' | 'ê' | 'ü')
}

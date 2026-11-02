//! Port of `GPT_SoVITS/text/zh_normalization/constants.py` (GPT-SoVITS @ 04a44eb).
//!
//! Python's translation tables (`{ord: ord}` dicts used with `str.translate`)
//! are lists of `(from, to)` char pairs here; [`translate`] applies one.

use std::sync::LazyLock;

use super::pyre::PyRegex;

/// Python `string.ascii_letters`.
const ASCII_LETTERS: &str = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
/// Python `string.digits`.
const DIGITS: &str = "0123456789";
/// Python `string.punctuation`.
const PUNCTUATION: &str = r##"!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"##;

fn full_width_table(chars: &str) -> Vec<(char, char)> {
  chars
    .chars()
    .map(|c| {
      (
        char::from_u32(c as u32 + 65248).expect("full-width char"),
        c,
      )
    })
    .collect()
}

fn invert(table: &[(char, char)]) -> Vec<(char, char)> {
  table.iter().map(|(k, v)| (*v, *k)).collect()
}

/// Port of `F2H_ASCII_LETTERS` (英文字符全角 -> 半角映射表, num: 52).
pub static F2H_ASCII_LETTERS: LazyLock<Vec<(char, char)>> =
  LazyLock::new(|| full_width_table(ASCII_LETTERS));
/// Port of `H2F_ASCII_LETTERS` (英文字符半角 -> 全角映射表).
pub static H2F_ASCII_LETTERS: LazyLock<Vec<(char, char)>> =
  LazyLock::new(|| invert(&F2H_ASCII_LETTERS));
/// Port of `F2H_DIGITS` (数字字符全角 -> 半角映射表, num: 10).
pub static F2H_DIGITS: LazyLock<Vec<(char, char)>> = LazyLock::new(|| full_width_table(DIGITS));
/// Port of `H2F_DIGITS` (数字字符半角 -> 全角映射表).
pub static H2F_DIGITS: LazyLock<Vec<(char, char)>> = LazyLock::new(|| invert(&F2H_DIGITS));
/// Port of `F2H_PUNCTUATIONS` (标点符号全角 -> 半角映射表, num: 32).
pub static F2H_PUNCTUATIONS: LazyLock<Vec<(char, char)>> =
  LazyLock::new(|| full_width_table(PUNCTUATION));
/// Port of `H2F_PUNCTUATIONS` (标点符号半角 -> 全角映射表).
pub static H2F_PUNCTUATIONS: LazyLock<Vec<(char, char)>> =
  LazyLock::new(|| invert(&F2H_PUNCTUATIONS));
/// Port of `F2H_SPACE` (空格, num: 1).
///
/// Unlike the other tables, upstream keys this dict by `str` (`{"\u3000": " "}`),
/// not by code point. `str.translate` looks characters up by code point, so this
/// table never matches anything; see [`translate_str_keyed`].
pub const F2H_SPACE: [(&str, &str); 1] = [("\u{3000}", " ")];
/// Port of `H2F_SPACE` (also keyed by `str`).
pub const H2F_SPACE: [(&str, &str); 1] = [(" ", "\u{3000}")];

/// Python `str.translate(table)` for a char-to-char table (`{ord: ord}`).
pub fn translate(text: &str, table: &[(char, char)]) -> String {
  text
    .chars()
    .map(|c| table.iter().find(|(k, _)| *k == c).map_or(c, |(_, v)| *v))
    .collect()
}

/// Python `str.translate(table)` for a table whose keys are `str` instead of
/// code points ([`F2H_SPACE`], [`H2F_SPACE`]). `str.translate` only looks up
/// `ord(c)`, which never equals a `str` key, so the text is returned unchanged.
pub fn translate_str_keyed(text: &str, _table: &[(&str, &str)]) -> String {
  text.to_string()
}

/// Port of `RE_NSW`: 非"有拼音的汉字"的字符串，可用于NSW提取.
///
/// This is the `SUPPORT_UCS4` branch (`pypinyin.constants.SUPPORT_UCS4` is true
/// on every Python 3 build).
pub static RE_NSW: LazyLock<PyRegex> = LazyLock::new(|| {
  PyRegex::compile(concat!(
    r"(?:[^",
    r"\u3007",                // 〇
    r"\u3400-\u4dbf",         // CJK扩展A:[3400-4DBF]
    r"\u4e00-\u9fff",         // CJK基本:[4E00-9FFF]
    r"\uf900-\ufaff",         // CJK兼容:[F900-FAFF]
    r"\U00020000-\U0002A6DF", // CJK扩展B:[20000-2A6DF]
    r"\U0002A703-\U0002B73F", // CJK扩展C:[2A700-2B73F]
    r"\U0002B740-\U0002B81D", // CJK扩展D:[2B740-2B81D]
    r"\U0002F80A-\U0002FA1F", // CJK兼容扩展:[2F800-2FA1F]
    r"])+"
  ))
});

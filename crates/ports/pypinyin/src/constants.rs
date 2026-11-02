//! Port of `pypinyin/constants.py`.

use std::collections::HashMap;
use std::sync::LazyLock;

pub use crate::compat::SUPPORT_UCS4;

/// Port of `pypinyin.pinyin_dict.pinyin_dict`, the single character pinyin
/// dictionary (`pinyin_dict.json`, keys are code points).
pub static PINYIN_DICT: LazyLock<HashMap<u32, String>> = LazyLock::new(|| {
  let raw: HashMap<String, String> =
    serde_json::from_str(include_str!("../data/pinyin_dict.json")).expect("valid pinyin_dict.json");
  raw
    .into_iter()
    .map(|(k, v)| {
      (
        k.parse::<u32>().expect("integer key in pinyin_dict.json"),
        v,
      )
    })
    .collect()
});

/// Port of `pypinyin.phrases_dict.phrases_dict`, the phrase pinyin dictionary
/// (`phrases_dict.json`).
///
/// `PYPINYIN_NO_PHRASES` is not supported: the dictionary is always loaded.
pub static PHRASES_DICT: LazyLock<HashMap<String, Vec<Vec<String>>>> = LazyLock::new(|| {
  serde_json::from_str(include_str!("../data/phrases_dict.json")).expect("valid phrases_dict.json")
});

/// Port of the character class inside `pypinyin.constants.RE_HANS`
/// (the `SUPPORT_UCS4` branch).
// The upstream class lists overlapping ranges; they are kept as written.
#[allow(unreachable_patterns)]
pub fn is_hans_char(c: char) -> bool {
  matches!(
      c as u32,
      0x3007
          | 0xE815..=0xE864
          | 0xFA18
          | 0x3400..=0x4DBF
          | 0x4E00..=0x9FFF
          | 0xF900..=0xFAFF
          | 0x20000..=0x2A6DF
          | 0x2A703..=0x2B73F
          | 0x2B740..=0x2B81D
          | 0x2B825..=0x2BF6E
          | 0x2C029..=0x2CE93
          | 0x2D016
          | 0x2D11B..=0x2EBD9
          | 0x2F80A..=0x2FA1F
          | 0x30000..=0x3134A
          | 0x300F7..=0x31288
          | 0x30EDD
          | 0x30EDE
          | 0x31350..=0x32389
  )
}

/// Port of the compiled regex `pypinyin.constants.RE_HANS`
/// (`^(?:[...])+$`).
#[derive(Clone, Copy, Debug)]
pub struct ReHans;

impl ReHans {
  /// Port of `RE_HANS.match(s)` (truthiness of the match object).
  ///
  /// Python's `$` also matches before a trailing `"\n"`, so `"中\n"`
  /// matches too.
  pub fn is_match(&self, s: &str) -> bool {
    let body = s.strip_suffix('\n').unwrap_or(s);
    !body.is_empty() && body.chars().all(is_hans_char)
  }
}

/// Port of `pypinyin.constants.RE_HANS`: matches strings made only of
/// characters that have pinyin.
pub const RE_HANS: ReHans = ReHans;

/// Port of `pypinyin.constants.Style` (an `IntEnum`).
///
/// Only the styles whose implementations are ported are listed (see the
/// crate README). The discriminants are the Python values.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
#[allow(non_camel_case_types)]
pub enum Style {
  /// `Style.NORMAL`: no tone, e.g. `zhong guo`.
  NORMAL = 0,
  /// `Style.TONE`: tone marks on the vowel, e.g. `zhōng guó`.
  TONE = 1,
  /// `Style.TONE2`: tone number after the marked vowel, e.g. `zho1ng guo2`.
  TONE2 = 2,
  /// `Style.TONE3`: tone number at the end, e.g. `zhong1 guo2`.
  TONE3 = 8,
  /// `Style.INITIALS`: initials only, e.g. `zh g`.
  INITIALS = 3,
  /// `Style.FIRST_LETTER`: first letter only, e.g. `z g`.
  FIRST_LETTER = 4,
  /// `Style.FINALS`: finals without tone, e.g. `ong uo`.
  FINALS = 5,
  /// `Style.FINALS_TONE`: finals with tone marks, e.g. `ōng uó`.
  FINALS_TONE = 6,
  /// `Style.FINALS_TONE2`: finals with tone number after the vowel, e.g. `o1ng uo2`.
  FINALS_TONE2 = 7,
  /// `Style.FINALS_TONE3`: finals with tone number at the end, e.g. `ong1 uo2`.
  FINALS_TONE3 = 9,
}

impl Style {
  /// Look up a style by its Python name (`Style['TONE3']`).
  pub fn from_name(name: &str) -> Option<Self> {
    Some(match name {
      "NORMAL" => Self::NORMAL,
      "TONE" => Self::TONE,
      "TONE2" => Self::TONE2,
      "TONE3" => Self::TONE3,
      "INITIALS" => Self::INITIALS,
      "FIRST_LETTER" => Self::FIRST_LETTER,
      "FINALS" => Self::FINALS,
      "FINALS_TONE" => Self::FINALS_TONE,
      "FINALS_TONE2" => Self::FINALS_TONE2,
      "FINALS_TONE3" => Self::FINALS_TONE3,
      _ => return None,
    })
  }
}

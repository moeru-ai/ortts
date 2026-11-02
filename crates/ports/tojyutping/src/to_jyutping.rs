//! Port of `ToJyutping/ToJyutping.py` (the jyutping methods of
//! `JyutpingConverter`).

use crate::trie::Trie;
use crate::utils::format_romanization_text;

/// Port of `ToJyutping.ToJyutping.JyutpingConverter`.
#[derive(Debug, Clone, Copy, Default)]
pub struct JyutpingConverter {
  t: Trie,
}

impl JyutpingConverter {
  /// Port of `JyutpingConverter.__init__`.
  pub const fn new(t: Trie) -> Self {
    Self { t }
  }

  /// Port of `JyutpingConverter.get_jyutping_list`: one item per input
  /// character, with its jyutping (space-separated when one character has
  /// several syllables) or `None`.
  pub fn get_jyutping_list(&self, s: &str) -> Vec<(String, Option<String>)> {
    self
      .t
      .get(s)
      .into_iter()
      .map(|(k, v)| (k, v.map(|v| v.jyutping())))
      .collect()
  }

  /// Port of `JyutpingConverter.get_jyutping`.
  pub fn get_jyutping(&self, s: &str) -> String {
    self
      .get_jyutping_list(s)
      .into_iter()
      .map(|(k, v)| match v {
        Some(v) if !v.is_empty() => format!("{k}({v})"),
        _ => k,
      })
      .collect()
  }

  /// Port of `JyutpingConverter.get_jyutping_text`.
  pub fn get_jyutping_text(&self, s: &str) -> String {
    format_romanization_text(s, |m| self.get_jyutping_list(m))
  }
}

/// Port of the module-level `ToJyutping.ToJyutping.ToJyutping` instance.
#[allow(non_upper_case_globals)]
pub static ToJyutping: JyutpingConverter = JyutpingConverter::new(Trie);

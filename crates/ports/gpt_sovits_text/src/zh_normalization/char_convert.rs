//! Port of `GPT_SoVITS/text/zh_normalization/char_convert.py` (GPT-SoVITS @ 04a44eb).
//!
//! Traditional and simplified Chinese conversion, a simplified character may
//! correspond to multiple traditional characters.
//!
//! The two literal strings live in `data/zh_normalization/` (extracted verbatim
//! by `parity/zh_normalization/convert_data.py`).

use std::collections::HashMap;
use std::sync::LazyLock;

/// Port of `simplified_charcters` (upstream spelling).
pub const SIMPLIFIED_CHARCTERS: &str =
  include_str!("../../data/zh_normalization/simplified_charcters.txt");

/// Port of `traditional_characters`.
pub const TRADITIONAL_CHARACTERS: &str =
  include_str!("../../data/zh_normalization/traditional_characters.txt");

/// Port of `(s2t_dict, t2s_dict)`, built like the module-level loop: index `i`
/// of `simplified_charcters` maps to index `i` of `traditional_characters`, and
/// later entries overwrite earlier ones.
static DICTS: LazyLock<(HashMap<char, char>, HashMap<char, char>)> = LazyLock::new(|| {
  let traditional: Vec<char> = TRADITIONAL_CHARACTERS.chars().collect();
  let mut s2t_dict = HashMap::new();
  let mut t2s_dict = HashMap::new();
  for (i, item) in SIMPLIFIED_CHARCTERS.chars().enumerate() {
    s2t_dict.insert(item, traditional[i]);
    t2s_dict.insert(traditional[i], item);
  }
  (s2t_dict, t2s_dict)
});

/// Port of `s2t_dict`.
pub fn s2t_dict() -> &'static HashMap<char, char> {
  &DICTS.0
}

/// Port of `t2s_dict`.
pub fn t2s_dict() -> &'static HashMap<char, char> {
  &DICTS.1
}

/// Port of `tranditional_to_simplified` (upstream spelling).
pub fn tranditional_to_simplified(text: &str) -> String {
  let t2s = t2s_dict();
  text
    .chars()
    .map(|item| *t2s.get(&item).unwrap_or(&item))
    .collect()
}

/// Port of `simplified_to_traditional`.
pub fn simplified_to_traditional(text: &str) -> String {
  let s2t = s2t_dict();
  text
    .chars()
    .map(|item| *s2t.get(&item).unwrap_or(&item))
    .collect()
}

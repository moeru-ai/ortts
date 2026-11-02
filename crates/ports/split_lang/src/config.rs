//! Port of `split_lang/config.py`.

/// Port of `DEFAULT_LANG_MAP` (in dict insertion order).
pub const DEFAULT_LANG_MAP: &[(&str, &str)] = &[
  ("zh", "zh"),
  ("yue", "zh"),
  ("wuu", "zh"),
  ("zh-cn", "zh"),
  ("zh-tw", "x"),
  ("ko", "ko"),
  ("ja", "ja"),
  ("de", "de"),
  ("fr", "fr"),
  ("en", "en"),
  ("hr", "en"),
];

/// Port of `ZH_JA_LANG_MAP`.
pub const ZH_JA_LANG_MAP: &[(&str, &str)] = &[
  ("zh", "zh"),
  ("yue", "zh"),
  ("wuu", "zh"),
  ("zh-cn", "zh"),
  ("zh-tw", "x"),
  ("ja", "ja"),
];

/// Port of `NO_ZH_JA_LANG_MAP`.
pub const NO_ZH_JA_LANG_MAP: &[(&str, &str)] =
  &[("de", "de"), ("fr", "fr"), ("en", "en"), ("hr", "en")];

/// Port of `DEFAULT_LANG`.
pub const DEFAULT_LANG: &str = "x";

/// `lang_map.get(key, default)` for a lang map given as `(key, value)` pairs.
pub fn lang_map_get<K: AsRef<str>, V: AsRef<str>>(
  lang_map: &[(K, V)],
  key: &str,
  default: &str,
) -> String {
  lang_map
    .iter()
    .find(|(k, _)| k.as_ref() == key)
    .map_or(default, |(_, v)| v.as_ref())
    .to_string()
}

//! Port of `pypinyin/phonetic_symbol.py`.

/// Port of `pypinyin.phonetic_symbol.phonetic_symbol`: toned characters and
/// their tone-number form, in the Python dict order.
///
/// `m̄`, `m̀`, `ê̄` and `ê̌` are two code points long.
#[allow(non_upper_case_globals)]
pub const phonetic_symbol: &[(&str, &str)] = &[
  ("ā", "a1"),
  ("á", "a2"),
  ("ǎ", "a3"),
  ("à", "a4"),
  ("ē", "e1"),
  ("é", "e2"),
  ("ě", "e3"),
  ("è", "e4"),
  ("ō", "o1"),
  ("ó", "o2"),
  ("ǒ", "o3"),
  ("ò", "o4"),
  ("ī", "i1"),
  ("í", "i2"),
  ("ǐ", "i3"),
  ("ì", "i4"),
  ("ū", "u1"),
  ("ú", "u2"),
  ("ǔ", "u3"),
  ("ù", "u4"),
  // üe
  ("ü", "v"),
  ("ǖ", "v1"),
  ("ǘ", "v2"),
  ("ǚ", "v3"),
  ("ǜ", "v4"),
  ("ń", "n2"),
  ("ň", "n3"),
  ("ǹ", "n4"),
  ("m\u{304}", "m1"),
  ("ḿ", "m2"),
  ("m\u{300}", "m4"),
  ("ê\u{304}", "ê1"),
  ("ế", "ê2"),
  ("ê\u{30c}", "ê3"),
  ("ề", "ê4"),
];

/// Port of `pypinyin.phonetic_symbol.phonetic_symbol_reverse`
/// (`dict((v, k) for k, v in phonetic_symbol.items())`).
pub fn phonetic_symbol_reverse(key: &str) -> Option<&'static str> {
  phonetic_symbol
    .iter()
    .find(|(_, v)| *v == key)
    .map(|(k, _)| *k)
}

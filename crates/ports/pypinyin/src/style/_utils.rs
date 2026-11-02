//! Port of `pypinyin/style/_utils.py`.

use crate::_re_digit::is_digit;
use crate::phonetic_symbol::phonetic_symbol;
use crate::standard::convert_finals;
use crate::style::_constants::{_FINALS, _INITIALS, _INITIALS_NOT_STRICT};

/// Port of `pypinyin.style._utils.get_initials`: the initial of one pinyin.
pub fn get_initials(pinyin: &str, strict: bool) -> String {
  let _initials = if strict {
    _INITIALS
  } else {
    _INITIALS_NOT_STRICT
  };

  for i in _initials {
    if pinyin.starts_with(i) {
      return (*i).to_string();
    }
  }
  String::new()
}

/// Port of `pypinyin.style._utils.get_finals`: the final of one pinyin
/// without tone marks.
pub fn get_finals(pinyin: &str, strict: bool) -> String {
  let pinyin = if strict {
    convert_finals(pinyin)
  } else {
    pinyin.to_string()
  };

  let initials = get_initials(&pinyin, strict);

  // 按声母分割，剩下的就是韵母
  let finals = &pinyin[initials.len()..];

  // 处理既没有声母也没有韵母的情况
  if strict && !_FINALS.contains(&finals) {
    // 处理 y, w 导致误判的问题，比如 yo
    let initials = get_initials(&pinyin, false);
    let finals = &pinyin[initials.len()..];
    if _FINALS.contains(&finals) {
      return finals.to_string();
    }
    return String::new();
  }

  // ń, ḿ
  if finals.is_empty() && !strict {
    return pinyin;
  }

  finals.to_string()
}

/// Port of `pypinyin.style._utils.replace_symbol_to_number`: replace tone
/// marks with tone numbers.
pub fn replace_symbol_to_number(pinyin: &str) -> String {
  // RE_PHONETIC_SYMBOL.sub: the single character keys.
  let mut value = String::with_capacity(pinyin.len() + 4);
  let mut buf = [0u8; 4];
  for c in pinyin.chars() {
    let key: &str = c.encode_utf8(&mut buf);
    match phonetic_symbol.iter().find(|(k, _)| *k == key) {
      Some((_, to)) => value.push_str(to),
      None => value.push(c),
    }
  }
  // PHONETIC_SYMBOL_DICT_KEY_LENGTH_NOT_ONE
  for (symbol, to) in phonetic_symbol {
    if symbol.chars().count() > 1 {
      value = value.replace(symbol, to);
    }
  }
  value
}

/// Port of `RE_NUMBER.sub('', value)` / `_re_number.sub('', value)` (`\d`).
pub fn re_number_sub_empty(value: &str) -> String {
  value.chars().filter(|c| !is_digit(*c)).collect()
}

/// Port of `pypinyin.style._utils.replace_symbol_to_no_symbol`: drop tone marks.
pub fn replace_symbol_to_no_symbol(pinyin: &str) -> String {
  let value = replace_symbol_to_number(pinyin);
  re_number_sub_empty(&value)
}

/// Port of `pypinyin.style._utils.has_finals`.
pub fn has_finals(pinyin: &str) -> bool {
  // 鼻音: 'm̄', 'ḿ', 'm̀', 'ń', 'ň', 'ǹ ' 没有韵母
  for symbol in ["m\u{304}", "ḿ", "m\u{300}", "ń", "ň", "ǹ"] {
    if pinyin.contains(symbol) {
      return false;
    }
  }
  true
}

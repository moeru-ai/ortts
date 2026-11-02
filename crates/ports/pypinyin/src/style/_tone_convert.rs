//! Port of `pypinyin/style/_tone_convert.py`: convert one pinyin between
//! the TONE / TONE2 / TONE3 / NORMAL / INITIALS / FINALS* styles.
//!
//! Keyword arguments keep their Python names and become positional
//! parameters in the Python order. The legacy `neutral_tone_with_5` keyword
//! is not ported.

use crate::_re_digit::is_digit;
use crate::phonetic_symbol::phonetic_symbol_reverse;
use crate::style::_constants::{is_tone2_vowel, re_tone3_sub};
use crate::style::_tone_rule::right_mark_index;
use crate::style::_utils::{
  get_finals, get_initials, re_number_sub_empty, replace_symbol_to_no_symbol,
  replace_symbol_to_number,
};
use crate::style::tone::converter;

/// Port of `pypinyin.style._tone_convert.to_normal`.
pub fn to_normal(pinyin: &str, v_to_u: bool) -> String {
  let s = tone_to_tone2(pinyin, true, false);
  let s = tone2_to_normal(&s, false);
  _fix_v_u(pinyin, &s, v_to_u)
}

/// Port of `pypinyin.style._tone_convert.to_tone`.
pub fn to_tone(pinyin: &str) -> String {
  let pinyin = pinyin.replace('v', "ü");
  if !pinyin.chars().any(is_digit) {
    return pinyin;
  }

  let s = tone_to_tone2(&pinyin, false, false);
  tone2_to_tone(&s)
}

/// Port of `pypinyin.style._tone_convert.to_tone2`.
pub fn to_tone2(pinyin: &str, v_to_u: bool, neutral_tone_with_five: bool) -> String {
  let pinyin = pinyin.replace('5', "");
  let s = tone_to_tone3(&pinyin, true, neutral_tone_with_five);
  let s = tone3_to_tone2(&s, false);
  _fix_v_u(&pinyin, &s, v_to_u)
}

/// Port of `pypinyin.style._tone_convert.to_tone3`.
pub fn to_tone3(pinyin: &str, v_to_u: bool, neutral_tone_with_five: bool) -> String {
  let pinyin = pinyin.replace('5', "");
  let s = tone_to_tone2(&pinyin, true, neutral_tone_with_five);
  let s = tone2_to_tone3(&s, false);
  _fix_v_u(&pinyin, &s, v_to_u)
}

/// Port of `pypinyin.style._tone_convert.to_initials`.
pub fn to_initials(pinyin: &str, strict: bool) -> String {
  get_initials(pinyin, strict)
}

/// Port of `pypinyin.style._tone_convert.to_finals`.
pub fn to_finals(pinyin: &str, strict: bool, v_to_u: bool) -> String {
  let new_pinyin = replace_symbol_to_no_symbol(pinyin).replace('v', "ü");
  let finals = get_finals(&new_pinyin, strict);
  _fix_v_u(&finals, &finals, v_to_u)
}

/// Port of `pypinyin.style._tone_convert.to_finals_tone`.
pub fn to_finals_tone(pinyin: &str, strict: bool) -> String {
  let finals = to_finals_tone2(pinyin, strict, false, false);
  tone2_to_tone(&finals)
}

/// Port of `pypinyin.style._tone_convert.to_finals_tone2`.
pub fn to_finals_tone2(
  pinyin: &str,
  strict: bool,
  v_to_u: bool,
  neutral_tone_with_five: bool,
) -> String {
  let pinyin = pinyin.replace('5', "");
  let finals = to_finals_tone3(&pinyin, strict, v_to_u, neutral_tone_with_five);
  tone3_to_tone2(&finals, v_to_u)
}

/// Port of `pypinyin.style._tone_convert.to_finals_tone3`.
pub fn to_finals_tone3(
  pinyin: &str,
  strict: bool,
  v_to_u: bool,
  neutral_tone_with_five: bool,
) -> String {
  let pinyin = pinyin.replace('5', "");
  let mut finals = to_finals(&pinyin, strict, v_to_u);
  if finals.is_empty() {
    return finals;
  }

  let number = match replace_symbol_to_number(&pinyin)
    .chars()
    .find(|c| is_digit(*c))
  {
    Some(n) => n,
    None => {
      if neutral_tone_with_five {
        '5'
      } else {
        return finals;
      }
    }
  };
  finals.push(number);
  finals
}

/// Port of `pypinyin.style._tone_convert.tone_to_normal`.
pub fn tone_to_normal(tone: &str, v_to_u: bool) -> String {
  let s = tone_to_tone2(tone, v_to_u, false);
  let s = re_number_sub_empty(&s);
  _v_to_u(&s, v_to_u)
}

/// Port of `pypinyin.style._tone_convert.tone_to_tone2`.
pub fn tone_to_tone2(tone: &str, v_to_u: bool, neutral_tone_with_five: bool) -> String {
  let tone3 = tone_to_tone3(tone, v_to_u, neutral_tone_with_five);
  let s = tone3_to_tone2(&tone3, false);
  _v_to_u(&s, v_to_u)
}

/// Port of `pypinyin.style._tone_convert.tone_to_tone3`.
pub fn tone_to_tone3(tone: &str, v_to_u: bool, neutral_tone_with_five: bool) -> String {
  let tone3 = converter.to_tone3(tone);
  let s = _improve_tone3(&tone3, neutral_tone_with_five);
  _v_to_u(&s, v_to_u)
}

/// Port of `pypinyin.style._tone_convert.tone2_to_normal`.
pub fn tone2_to_normal(tone2: &str, v_to_u: bool) -> String {
  let s = re_number_sub_empty(tone2);
  let s = _v_to_u(&s, v_to_u);
  _fix_v_u(tone2, &s, v_to_u)
}

/// Port of `pypinyin.style._tone_convert.tone2_to_tone`.
pub fn tone2_to_tone(tone2: &str) -> String {
  // regex = RE_TONE2 without '$': ([aeoiuvnmêü])([1-5])
  let string = tone2.replace('ü', "v").replace(['5', '0'], "");
  let chars: Vec<char> = string.chars().collect();
  let mut out = String::with_capacity(string.len() + 4);
  let mut i = 0;
  while i < chars.len() {
    if is_tone2_vowel(chars[i]) && i + 1 < chars.len() && ('1'..='5').contains(&chars[i + 1]) {
      let s: String = [chars[i], chars[i + 1]].iter().collect();
      match phonetic_symbol_reverse(&s) {
        Some(d) => out.push_str(d),
        None => out.push_str(&s),
      }
      i += 2;
    } else {
      out.push(chars[i]);
      i += 1;
    }
  }
  out.replace('v', "ü")
}

/// Port of `pypinyin.style._tone_convert.tone2_to_tone3`.
pub fn tone2_to_tone3(tone2: &str, v_to_u: bool) -> String {
  let tone3 = re_tone3_sub(tone2);
  _fix_v_u(tone2, &tone3, v_to_u)
}

/// Port of `pypinyin.style._tone_convert.tone3_to_normal`.
pub fn tone3_to_normal(tone3: &str, v_to_u: bool) -> String {
  let s = re_number_sub_empty(tone3);
  let s = _v_to_u(&s, v_to_u);
  _fix_v_u(tone3, &s, v_to_u)
}

/// Port of `pypinyin.style._tone_convert.tone3_to_tone`.
pub fn tone3_to_tone(tone3: &str) -> String {
  let tone2 = tone3_to_tone2(tone3, true);
  tone2_to_tone(&tone2)
}

/// Port of `pypinyin.style._tone_convert.tone3_to_tone2`.
pub fn tone3_to_tone2(tone3: &str, v_to_u: bool) -> String {
  let no_number_tone3: Vec<char> = tone3_to_normal(tone3, false).chars().collect();
  let mark_index: isize = match right_mark_index(&no_number_tone3.iter().collect::<String>()) {
    Some(i) => i as isize,
    None => no_number_tone3.len() as isize - 1,
  };
  let split = ((mark_index + 1) as usize).min(no_number_tone3.len());
  let before: String = no_number_tone3[..split].iter().collect();
  let after: String = no_number_tone3[split..].iter().collect();

  let Some(number) = _get_number_from_pinyin(tone3) else {
    return tone3.to_string();
  };

  let s = format!("{before}{number}{after}");
  _fix_v_u(tone3, &s, v_to_u)
}

/// Port of `pypinyin.style._tone_convert._improve_tone3`.
pub fn _improve_tone3(tone3: &str, neutral_tone_with_five: bool) -> String {
  let number = _get_number_from_pinyin(tone3);
  if number.is_none() && neutral_tone_with_five && !tone3.is_empty() {
    return format!("{tone3}5");
  }
  tone3.to_string()
}

/// Port of `pypinyin.style._tone_convert._get_number_from_pinyin`: the first
/// `\d` character.
pub fn _get_number_from_pinyin(pinyin: &str) -> Option<char> {
  pinyin.chars().find(|c| is_digit(*c))
}

/// Port of `pypinyin.style._tone_convert._v_to_u`.
pub fn _v_to_u(pinyin: &str, replace: bool) -> String {
  if !replace {
    return pinyin.to_string();
  }
  pinyin.replace('v', "ü")
}

/// Port of `pypinyin.style._tone_convert._fix_v_u`.
pub fn _fix_v_u(_origin_py: &str, new_py: &str, v_to_u: bool) -> String {
  if !v_to_u {
    return new_py.replace('ü', "v");
  }

  _v_to_u(new_py, true)
}

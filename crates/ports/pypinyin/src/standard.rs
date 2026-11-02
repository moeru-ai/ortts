//! Port of `pypinyin/standard.py`: restore the original finals of the
//! 《汉语拼音方案》 spelling rules.
//!
//! The Python module uses regular expressions. They are written out by hand
//! here. Python's `$` also matches before a trailing `"\n"`, which
//! `split_py_dollar` reproduces.

use crate::style::_constants::_FINALS;

/// Split `s` the way a Python `...$` pattern sees it: the part that the
/// pattern must match and a trailing `"\n"` that `$` may stand before.
pub(crate) fn split_py_dollar(s: &str) -> (&str, &str) {
  match s.strip_suffix('\n') {
    Some(body) => (body, "\n"),
    None => (s, ""),
  }
}

/// Port of `pypinyin.standard.UV_MAP` (u -> ü).
pub const UV_MAP: &[(char, char)] = &[('u', 'ü'), ('ū', 'ǖ'), ('ú', 'ǘ'), ('ǔ', 'ǚ'), ('ù', 'ǜ')];

/// Port of `pypinyin.standard.I_TONES`.
pub const I_TONES: &[char] = &['i', 'ī', 'í', 'ǐ', 'ì'];

/// Port of `pypinyin.standard.IU_MAP` (iu -> iou).
pub const IU_MAP: &[(&str, &str)] = &[
  ("iu", "iou"),
  ("iū", "ioū"),
  ("iú", "ioú"),
  ("iǔ", "ioǔ"),
  ("iù", "ioù"),
];

/// Port of `pypinyin.standard.UI_MAP` (ui -> uei).
pub const UI_MAP: &[(&str, &str)] = &[
  ("ui", "uei"),
  ("uī", "ueī"),
  ("uí", "ueí"),
  ("uǐ", "ueǐ"),
  ("uì", "ueì"),
];

/// Port of `pypinyin.standard.UN_MAP` (un -> uen).
pub const UN_MAP: &[(&str, &str)] = &[
  ("un", "uen"),
  ("ūn", "ūen"),
  ("ún", "úen"),
  ("ǔn", "ǔen"),
  ("ùn", "ùen"),
];

fn uv_map(c: char) -> Option<char> {
  UV_MAP.iter().find(|(k, _)| *k == c).map(|(_, v)| *v)
}

fn is_u_tone(c: char) -> bool {
  uv_map(c).is_some()
}

fn is_ascii_lower(c: char) -> bool {
  c.is_ascii_lowercase()
}

/// Port of `pypinyin.standard.convert_zero_consonant`: restore the finals of
/// syllables written with `y` / `w`.
pub fn convert_zero_consonant(pinyin: &str) -> String {
  let raw_pinyin = pinyin;
  let mut pinyin: String = pinyin.to_string();
  // y: yu -> v, yi -> i, y -> i
  if raw_pinyin.starts_with('y') {
    let no_y_py: String = pinyin.chars().skip(1).collect();
    let first_char = no_y_py.chars().next();

    if let Some(fc) = first_char.filter(|c| is_u_tone(*c)) {
      // yu -> ü: yue -> üe
      let rest: String = pinyin.chars().skip(2).collect();
      pinyin = format!("{}{}", uv_map(fc).unwrap(), rest);
    } else if first_char.is_some_and(|c| I_TONES.contains(&c)) {
      // yi -> i: yi -> i
      pinyin = no_y_py;
    } else {
      // y -> i: ya -> ia
      pinyin = format!("i{no_y_py}");
    }
  }

  // w: wu -> u, w -> u
  if raw_pinyin.starts_with('w') {
    let no_w_py: String = pinyin.chars().skip(1).collect();
    let first_char = no_w_py.chars().next();

    if first_char.is_some_and(is_u_tone) {
      // wu -> u: wu -> u
      pinyin = pinyin.chars().skip(1).collect();
    } else {
      // w -> u: wa -> ua
      let rest: String = pinyin.chars().skip(1).collect();
      pinyin = format!("u{rest}");
    }
  }

  // 确保不会出现韵母表中不存在的韵母
  if !_FINALS.contains(&pinyin.as_str()) {
    return raw_pinyin.to_string();
  }

  pinyin
}

/// Port of `pypinyin.standard.convert_uv` (`UV_RE.sub`, with
/// `UV_RE = ^(j|q|x)(u|ū|ú|ǔ|ù)(.*)$`).
pub fn convert_uv(pinyin: &str) -> String {
  let (body, tail) = split_py_dollar(pinyin);
  let mut chars = body.chars();
  let (Some(c1), Some(c2)) = (chars.next(), chars.next()) else {
    return pinyin.to_string();
  };
  let rest = chars.as_str();
  // `.` does not match "\n".
  if !matches!(c1, 'j' | 'q' | 'x') || !is_u_tone(c2) || rest.contains('\n') {
    return pinyin.to_string();
  }
  format!("{c1}{}{rest}{tail}", uv_map(c2).unwrap())
}

/// Shared shape of `IU_RE`, `UI_RE` and `UN_RE`:
/// `{anchor}([a-z]+)({tones})$` where every tone is two characters long.
///
/// The tone has to be the last two characters of the matched part, and the
/// character before it has to be in `[a-z]`. With `anchored` (`^`) the whole
/// prefix has to be `[a-z]+`.
fn sub_final_tone(pinyin: &str, map: &[(&str, &str)], anchored: bool) -> String {
  let (body, tail) = split_py_dollar(pinyin);
  let chars: Vec<char> = body.chars().collect();
  if chars.len() < 3 {
    return pinyin.to_string();
  }
  let split = chars.len() - 2;
  let tone: String = chars[split..].iter().collect();
  let Some((_, replacement)) = map.iter().find(|(k, _)| **k == tone) else {
    return pinyin.to_string();
  };
  let prefix = &chars[..split];
  let ok = if anchored {
    prefix.iter().all(|c| is_ascii_lower(*c))
  } else {
    is_ascii_lower(prefix[prefix.len() - 1])
  };
  if !ok {
    return pinyin.to_string();
  }
  let prefix: String = prefix.iter().collect();
  format!("{prefix}{replacement}{tail}")
}

/// Port of `pypinyin.standard.convert_iou` (`IU_RE.sub`, with
/// `IU_RE = ^([a-z]+)(iu|iū|iú|iǔ|iù)$`).
pub fn convert_iou(pinyin: &str) -> String {
  sub_final_tone(pinyin, IU_MAP, true)
}

/// Port of `pypinyin.standard.convert_uei` (`UI_RE.sub`, with
/// `UI_RE = ([a-z]+)(ui|uī|uí|uǐ|uì)$`).
pub fn convert_uei(pinyin: &str) -> String {
  sub_final_tone(pinyin, UI_MAP, false)
}

/// Port of `pypinyin.standard.convert_uen` (`UN_RE.sub`, with
/// `UN_RE = ([a-z]+)(un|ūn|ún|ǔn|ùn)$`).
pub fn convert_uen(pinyin: &str) -> String {
  sub_final_tone(pinyin, UN_MAP, false)
}

/// Port of `pypinyin.standard.convert_finals`: restore the original finals.
pub fn convert_finals(pinyin: &str) -> String {
  let pinyin = convert_zero_consonant(pinyin);
  let pinyin = convert_uv(&pinyin);
  let pinyin = convert_iou(&pinyin);
  let pinyin = convert_uei(&pinyin);
  convert_uen(&pinyin)
}

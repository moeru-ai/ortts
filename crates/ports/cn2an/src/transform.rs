//! Port of `cn2an/transform.py` for `method="an2cn"`.
//!
//! The regular expressions of the an2cn path are implemented as hand-written
//! matchers over `char`s. Each matcher documents the Python pattern it
//! replaces. `\d` follows Python 3.11 `re` (Unicode decimal digits), see
//! `data/python_decimal_ranges.txt`.

use crate::an2cn::An2Cn;
use crate::py_re::{char_at, digits_end, is_digit, sub, try_sub};
use crate::{Error, Result};

/// Port of `Transform.measure_words`, split on `|`.
const MEASURE_WORDS: [&str; 34] = [
  "斤", "克", "千克", "公斤", "吨", "米", "厘米", "毫米", "公里", "升", "毫升", "元", "角", "分",
  "个", "只", "条", "张", "块", "瓶", "杯", "份", "本", "辆", "台", "匹", "头", "位", "亩", "小时",
  "分钟", "秒", "天", "半",
];

/// A hand-written matcher: the end of the match that starts at `pos`.
type Matcher = fn(&[char], usize) -> Option<usize>;

/// `(?=(measure_words))` at `pos`.
fn lookahead_measure_word(chars: &[char], pos: usize) -> bool {
  MEASURE_WORDS
    .iter()
    .any(|word| (pos..).zip(word.chars()).all(|(p, c)| char_at(chars, p, c)))
}

/// `\d+(\.\d+)?` at `pos`, greedy. Backtracking into it never helps the
/// patterns below because what follows it is never a digit or `.`.
fn match_decimal(chars: &[char], pos: usize) -> Option<usize> {
  let end = digits_end(chars, pos);
  if end == pos {
    return None;
  }
  if char_at(chars, end, '.') {
    let frac_end = digits_end(chars, end + 1);
    if frac_end > end + 1 {
      return Some(frac_end);
    }
  }
  Some(end)
}

/// `-?(\d+\.)?\d+` at `pos`.
fn match_number(chars: &[char], pos: usize) -> Option<usize> {
  let start = if char_at(chars, pos, '-') {
    pos + 1
  } else {
    pos
  };
  // Without the `-`, the pattern would have to start at `-`, which is not a
  // digit, so there is nothing to backtrack to.
  match_decimal(chars, start)
}

/// `\d+(\.\d+)?-\d+(\.\d+)?(?=(measure_words))` at `pos`.
fn match_range(chars: &[char], pos: usize) -> Option<usize> {
  let first = match_decimal(chars, pos)?;
  if !char_at(chars, first, '-') {
    return None;
  }
  let second = match_decimal(chars, first + 1)?;
  lookahead_measure_word(chars, second).then_some(second)
}

/// `\d{min,max}<unit>` at `pos`.
fn match_date_part(
  chars: &[char],
  pos: usize,
  min: usize,
  max: usize,
  unit: char,
) -> Option<usize> {
  let end = digits_end(chars, pos);
  let len = end - pos;
  // Fewer digits would leave a digit, not `unit`, after the group.
  (min <= len && len <= max && char_at(chars, end, unit)).then_some(end + 1)
}

/// `(\d{2,4}年)?(\d{1,2}月)?(\d{1,2}日)?` at `pos` (always matches, maybe
/// empty).
fn match_date(chars: &[char], pos: usize) -> Option<usize> {
  let mut end = pos;
  if let Some(e) = match_date_part(chars, end, 2, 4, '年') {
    end = e;
  }
  if let Some(e) = match_date_part(chars, end, 1, 2, '月') {
    end = e;
  }
  if let Some(e) = match_date_part(chars, end, 1, 2, '日') {
    end = e;
  }
  Some(end)
}

/// `\d+/\d+` at `pos`.
fn match_fraction(chars: &[char], pos: usize) -> Option<usize> {
  let first = digits_end(chars, pos);
  if first == pos || !char_at(chars, first, '/') {
    return None;
  }
  let second = digits_end(chars, first + 1);
  (second > first + 1).then_some(second)
}

/// `-?(\d+\.)?\d+%` at `pos`.
fn match_percent(chars: &[char], pos: usize) -> Option<usize> {
  let start = if char_at(chars, pos, '-') {
    pos + 1
  } else {
    pos
  };
  let first = digits_end(chars, start);
  if first == start {
    return None;
  }
  if char_at(chars, first, '%') {
    return Some(first + 1);
  }
  if char_at(chars, first, '.') {
    let second = digits_end(chars, first + 1);
    if second > first + 1 && char_at(chars, second, '%') {
      return Some(second + 1);
    }
  }
  None
}

/// `\d+℃` at `pos`.
fn match_celsius(chars: &[char], pos: usize) -> Option<usize> {
  let end = digits_end(chars, pos);
  (end > pos && char_at(chars, end, '℃')).then_some(end + 1)
}

/// `(?<!\d)-?(\d+\.)?\d+` (`Transform.direct_an_pattern`) at `pos`.
fn match_direct_an(chars: &[char], pos: usize) -> Option<usize> {
  if pos > 0 && is_digit(chars[pos - 1]) {
    return None;
  }
  match_number(chars, pos)
}

/// `\d+` at `pos`.
fn match_digits(chars: &[char], pos: usize) -> Option<usize> {
  let end = digits_end(chars, pos);
  (end > pos).then_some(end)
}

/// `\d+(?=年)` at `pos`.
fn match_digits_before_year(chars: &[char], pos: usize) -> Option<usize> {
  let end = digits_end(chars, pos);
  (end > pos && char_at(chars, end, '年')).then_some(end)
}

/// Port of `cn2an.transform.Transform`, an2cn direction only.
#[derive(Debug, Clone, Default)]
pub struct Transform {
  /// Port of `Transform.an2cn` (`An2Cn().an2cn`).
  an2cn: An2Cn,
}

impl Transform {
  /// Port of `Transform.__init__`.
  pub fn new() -> Self {
    Self {
      an2cn: An2Cn::new(),
    }
  }

  /// Port of `Transform.transform`.
  ///
  /// `method="an2cn"` is ported. `method="cn2an"` returns
  /// [`Error::NotPorted`]; any other method returns [`Error::ValueError`]
  /// as in Python.
  pub fn transform(&self, inputs: &str, method: &str, direct: bool) -> Result<String> {
    match method {
      "an2cn" => Ok(self.transform_an2cn(inputs, direct)),
      "cn2an" => Err(Error::NotPorted("transform(method=\"cn2an\")".to_owned())),
      _ => Err(Error::ValueError(format!(
        "error method: {method}, only support 'cn2an' and 'an2cn'!"
      ))),
    }
  }

  /// The `method == "an2cn"` branch of `Transform.transform`.
  fn transform_an2cn(&self, inputs: &str, direct: bool) -> String {
    let chars: Vec<char> = inputs.chars().collect();
    if direct {
      return sub(&chars, match_direct_an, |m| self.sub_util(m, "direct"));
    }

    let passes: [(Matcher, &str); 5] = [
      (match_range, "range"),
      // date
      (match_date, "date"),
      // fraction
      (match_fraction, "fraction"),
      // percent
      (match_percent, "percent"),
      // celsius
      (match_celsius, "celsius"),
    ];
    let mut chars = chars;
    for (matcher, sub_mode) in passes {
      let out = sub(&chars, matcher, |m| self.sub_util(m, sub_mode));
      chars = out.chars().collect();
    }
    // number
    sub(&chars, match_number, |m| self.sub_util(m, "number"))
  }

  /// Port of `Transform.__sub_util` for `method="an2cn"`.
  ///
  /// Python catches every exception, warns, and returns the input
  /// unchanged; empty input returns `None`, which `re.sub` treats as an
  /// empty string.
  fn sub_util(&self, inputs: &[char], sub_mode: &str) -> String {
    if inputs.is_empty() {
      return String::new();
    }
    self
      .try_sub_util(inputs, sub_mode)
      .unwrap_or_else(|_| inputs.iter().collect())
  }

  fn try_sub_util(&self, inputs: &[char], sub_mode: &str) -> Result<String> {
    let an2cn = |s: &[char], mode: &str| -> Result<String> {
      self.an2cn.an2cn(&s.iter().collect::<String>(), mode)
    };
    match sub_mode {
      "date" => {
        let inputs: Vec<char> = try_sub(inputs, match_digits_before_year, |m| an2cn(m, "direct"))?
          .chars()
          .collect();
        try_sub(&inputs, match_digits, |m| an2cn(m, "low"))
      }
      "fraction" => {
        let frac_result = try_sub(inputs, match_digits, |m| an2cn(m, "low"))?;
        let parts: Vec<&str> = frac_result.split('/').collect();
        match parts.as_slice() {
          [numerator, denominator] => Ok(format!("{denominator}分之{numerator}")),
          _ => Err(Error::ValueError("too many values to unpack".to_owned())),
        }
      }
      "celsius" => Ok(an2cn(&inputs[..inputs.len() - 1], "low")? + "摄氏度"),
      "percent" => Ok("百分之".to_owned() + &an2cn(&inputs[..inputs.len() - 1], "low")?),
      "range" => {
        let parts: Vec<&[char]> = inputs.split(|&c| c == '-').collect();
        match parts.as_slice() {
          [start, end] => Ok(format!("{}到{}", an2cn(start, "low")?, an2cn(end, "low")?)),
          _ => Err(Error::ValueError("too many values to unpack".to_owned())),
        }
      }
      "number" => an2cn(inputs, "low"),
      "direct" => an2cn(inputs, "direct"),
      _ => Err(Error::ValueError(format!("error sub_mode: {sub_mode} !"))),
    }
  }
}

//! Port of `GPT_SoVITS/text/zh_normalization/chronology.py` (GPT-SoVITS @ 04a44eb).

use std::sync::LazyLock;

use super::Error;
use super::num::{digits, num2str, verbalize_cardinal, verbalize_digit};
use super::pyre::{PyMatch, PyRegex};

/// Port of `_time_num2str`: a special case for verbalizing number in time.
pub fn _time_num2str(num_string: &str) -> Result<String, Error> {
  let mut result = num2str(num_string.trim_start_matches('0'))?;
  if num_string.starts_with('0') {
    result = format!("{}{}", digits("0")?, result);
  }
  Ok(result)
}

/// Port of `RE_TIME` (时刻表达式).
pub static RE_TIME: LazyLock<PyRegex> = LazyLock::new(|| {
  PyRegex::compile(concat!(
    r"([0-1]?[0-9]|2[0-3])",
    r":([0-5][0-9])",
    r"(:([0-5][0-9]))?"
  ))
});

/// Port of `RE_TIME_RANGE` (时间范围，如8:30-12:30).
pub static RE_TIME_RANGE: LazyLock<PyRegex> = LazyLock::new(|| {
  PyRegex::compile(concat!(
    r"([0-1]?[0-9]|2[0-3])",
    r":([0-5][0-9])",
    r"(:([0-5][0-9]))?",
    r"(~|-)",
    r"([0-1]?[0-9]|2[0-3])",
    r":([0-5][0-9])",
    r"(:([0-5][0-9]))?"
  ))
});

/// Python `int(s)` for the ASCII digit strings matched by `[0-5][0-9]`.
fn int(s: &str) -> i64 {
  s.parse().expect("ASCII digits")
}

/// Port of `replace_time`.
pub fn replace_time(m: &PyMatch) -> Result<String, Error> {
  let is_range = m.groups_len() > 5;

  let hour = m.group(1).unwrap_or_default();
  let minute = m.group(2).unwrap_or_default();
  let second = m.group(4);

  let mut result = format!("{}点", num2str(hour)?);
  if !minute.trim_start_matches('0').is_empty() {
    if int(minute) == 30 {
      result.push('半');
    } else {
      result.push_str(&format!("{}分", _time_num2str(minute)?));
    }
  }
  if let Some(second) = second.filter(|s| !s.trim_start_matches('0').is_empty()) {
    result.push_str(&format!("{}秒", _time_num2str(second)?));
  }

  if is_range {
    let hour_2 = m.group(6).unwrap_or_default();
    let minute_2 = m.group(7).unwrap_or_default();
    let second_2 = m.group(9);

    result.push('至');
    result.push_str(&format!("{}点", num2str(hour_2)?));
    if !minute_2.trim_start_matches('0').is_empty() {
      // Upstream checks the first minute here, not `minute_2`.
      if int(minute) == 30 {
        result.push('半');
      } else {
        result.push_str(&format!("{}分", _time_num2str(minute_2)?));
      }
    }
    if let Some(second_2) = second_2.filter(|s| !s.trim_start_matches('0').is_empty()) {
      result.push_str(&format!("{}秒", _time_num2str(second_2)?));
    }
  }

  Ok(result)
}

/// Port of `RE_DATE`.
pub static RE_DATE: LazyLock<PyRegex> = LazyLock::new(|| {
  PyRegex::compile(concat!(
    r"(\d{4}|\d{2})年",
    r"((0?[1-9]|1[0-2])月)?",
    r"(((0?[1-9])|((1|2)[0-9])|30|31)([日号]))?"
  ))
});

/// Port of `replace_date`.
pub fn replace_date(m: &PyMatch) -> Result<String, Error> {
  let year = m.group(1);
  let month = m.group(3);
  let day = m.group(5);
  let mut result = String::new();
  if let Some(year) = year.filter(|s| !s.is_empty()) {
    result.push_str(&format!("{}年", verbalize_digit(year, false)?));
  }
  if let Some(month) = month.filter(|s| !s.is_empty()) {
    result.push_str(&format!("{}月", verbalize_cardinal(month)?));
  }
  if let Some(day) = day.filter(|s| !s.is_empty()) {
    result.push_str(&format!(
      "{}{}",
      verbalize_cardinal(day)?,
      m.group(9).unwrap_or_default()
    ));
  }
  Ok(result)
}

/// Port of `RE_DATE2` (用 / 或者 - 分隔的 YY/MM/DD 或者 YY-MM-DD 日期).
pub static RE_DATE2: LazyLock<PyRegex> =
  LazyLock::new(|| PyRegex::compile(r"(\d{4})([- /.])(0[1-9]|1[012])\2(0[1-9]|[12][0-9]|3[01])"));

/// Port of `replace_date2`.
pub fn replace_date2(m: &PyMatch) -> Result<String, Error> {
  let year = m.group(1);
  let month = m.group(3);
  let day = m.group(4);
  let mut result = String::new();
  if let Some(year) = year.filter(|s| !s.is_empty()) {
    result.push_str(&format!("{}年", verbalize_digit(year, false)?));
  }
  if let Some(month) = month.filter(|s| !s.is_empty()) {
    result.push_str(&format!("{}月", verbalize_cardinal(month)?));
  }
  if let Some(day) = day.filter(|s| !s.is_empty()) {
    result.push_str(&format!("{}日", verbalize_cardinal(day)?));
  }
  Ok(result)
}

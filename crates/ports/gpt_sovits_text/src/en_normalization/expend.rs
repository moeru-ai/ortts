//! Port of `GPT_SoVITS/text/en_normalization/expend.py` (GPT-SoVITS @ 04a44eb).
//!
//! The regex sources are the Python sources, compiled with
//! [`PyRegex`], which keeps Python's meaning of `\b`,
//! `\d` and `\s`. Python `int` values are kept as decimal strings without
//! leading zeros (`str(int(s))`, see [`py_int`]), so arbitrarily long numbers
//! behave as in Python.

use std::sync::LazyLock;

use inflect::{NumberOrWord, NumberToWordsOptions, engine};
use unicode_normalization::UnicodeNormalization;

use super::Error;
use super::pyre::{PyMatch, PyRegex, py_category_is_mn, py_int, py_isspace};

/// Port of `measurement_map`: unit suffix to `[singular, plural]`.
pub const MEASUREMENT_MAP: &[(&str, [&str; 2])] = &[
  ("m", ["meter", "meters"]),
  ("km", ["kilometer", "kilometers"]),
  ("km/h", ["kilometer per hour", "kilometers per hour"]),
  ("ft", ["feet", "feet"]),
  ("L", ["liter", "liters"]),
  ("tbsp", ["tablespoon", "tablespoons"]),
  ("tsp", ["teaspoon", "teaspoons"]),
  ("h", ["hour", "hours"]),
  ("min", ["minute", "minutes"]),
  ("s", ["second", "seconds"]),
  ("°C", ["degree celsius", "degrees celsius"]),
  ("°F", ["degree fahrenheit", "degrees fahrenheit"]),
];

/// `measurement_map[sign]`.
fn measurement_map(sign: &str) -> [&'static str; 2] {
  MEASUREMENT_MAP
    .iter()
    .find(|(k, _)| *k == sign)
    .map(|(_, v)| *v)
    .unwrap_or_else(|| panic!("KeyError: {sign:?}"))
}

/// Port of `_inflect = inflect.engine()`. The engine keeps no state between
/// calls (`_number_args` is reset by every `number_to_words`), so a fresh
/// engine per call is equivalent.
fn _inflect() -> engine {
  engine::new()
}

/// A Python `int` (as `str(int)`) as an inflect argument. `str(n)` is what
/// inflect works on, so a number beyond `i128` is passed as its string.
fn int_arg(digits: &str) -> NumberOrWord<'_> {
  match digits.parse::<i128>() {
    Ok(n) => NumberOrWord::Number(n),
    Err(_) => NumberOrWord::Word(digits),
  }
}

/// `int(s)` as a small number, for comparisons; `None` if it does not fit.
fn small(digits: &str) -> Option<u64> {
  digits.parse::<u64>().ok()
}

/// `_inflect.number_to_words(num)` with the default keyword arguments.
fn number_to_words(num: NumberOrWord<'_>) -> Result<String, Error> {
  Ok(_inflect().number_to_words(num, &NumberToWordsOptions::default())?)
}

macro_rules! py_re {
  ($name:ident, $src:expr, $doc:expr) => {
    #[doc = $doc]
    pub static $name: LazyLock<PyRegex> = LazyLock::new(|| PyRegex::compile($src));
  };
}

py_re!(
  _ORDINAL_NUMBER_RE,
  r"\b([0-9]+)\. ",
  "Port of `_ordinal_number_re`."
);
py_re!(
  _COMMA_NUMBER_RE,
  r"([0-9][0-9\,]+[0-9])",
  "Port of `_comma_number_re`."
);
py_re!(
  _TIME_RE,
  r"\b([01]?[0-9]|2[0-3]):([0-5][0-9])\b",
  "Port of `_time_re`."
);
py_re!(
  _MEASUREMENT_RE,
  r"\b([0-9]+(\.[0-9]+)?(m|km|km/h|ft|L|tbsp|tsp|h|min|s|°C|°F))\b",
  "Port of `_measurement_re`."
);
py_re!(
  _POUNDS_RE_START,
  r"£([0-9\.\,]*[0-9]+)",
  "Port of `_pounds_re_start`."
);
py_re!(
  _POUNDS_RE_END,
  r"([0-9\.\,]*[0-9]+)£",
  "Port of `_pounds_re_end`."
);
py_re!(
  _DOLLARS_RE_START,
  r"\$([0-9\.\,]*[0-9]+)",
  "Port of `_dollars_re_start`."
);
py_re!(
  _DOLLARS_RE_END,
  r"([(0-9\.\,]*[0-9]+)\$",
  "Port of `_dollars_re_end`."
);
py_re!(
  _DECIMAL_NUMBER_RE,
  r"([0-9]+\.\s*[0-9]+)",
  "Port of `_decimal_number_re`."
);
py_re!(_FRACTION_RE, r"([0-9]+/[0-9]+)", "Port of `_fraction_re`.");
py_re!(
  _ORDINAL_RE,
  r"[0-9]+(st|nd|rd|th)",
  "Port of `_ordinal_re`."
);
py_re!(_NUMBER_RE, r"[0-9]+", "Port of `_number_re`.");
py_re!(
  RE_ASMD,
  r"((-?)((\d+)(\.\d+)?[⁰¹²³⁴⁵⁶⁷⁸⁹ˣʸⁿ]*)|(\.\d+[⁰¹²³⁴⁵⁶⁷⁸⁹ˣʸⁿ]*)|([A-Za-z][⁰¹²³⁴⁵⁶⁷⁸⁹ˣʸⁿ]*))\s+([\+\-\×÷=])\s+((-?)((\d+)(\.\d+)?[⁰¹²³⁴⁵⁶⁷⁸⁹ˣʸⁿ]*)|(\.\d+[⁰¹²³⁴⁵⁶⁷⁸⁹ˣʸⁿ]*)|([A-Za-z][⁰¹²³⁴⁵⁶⁷⁸⁹ˣʸⁿ]*))",
  "Port of `RE_ASMD` (addition, subtraction, multiplication, division, equals)."
);
py_re!(
  RE_INTEGER,
  concat!(r"(?:^|\s+)(-)", r"(\d+)"),
  "Port of `RE_INTEGER`."
);
py_re!(
  _PERCENT_RE,
  "%",
  "The pattern of `re.sub(\"%\", ...)` in `normalize`."
);
py_re!(
  _NON_ALPHA_RE,
  r"[^ A-Za-z'.,?!\-]",
  "The pattern of `re.sub(\"[^ A-Za-z'.,?!\\-]\", ...)` in `normalize`."
);
py_re!(
  _IE_RE,
  r"(?i)i\.e\.",
  "The pattern of `re.sub(r\"(?i)i\\.e\\.\", ...)` in `normalize`. It only sees ASCII, so \
   Python and Rust case folding agree."
);
py_re!(
  _EG_RE,
  r"(?i)e\.g\.",
  "The pattern of `re.sub(r\"(?i)e\\.g\\.\", ...)` in `normalize`."
);

/// Port of `_convert_ordinal`: `"1. "` becomes `"1st, "`.
pub fn _convert_ordinal(m: &PyMatch<'_>) -> Result<String, Error> {
  let ordinal = _inflect().ordinal(m.group_str(1))?;
  Ok(ordinal + ", ")
}

/// Port of `_remove_commas`.
pub fn _remove_commas(m: &PyMatch<'_>) -> Result<String, Error> {
  Ok(m.group_str(1).replace(',', ""))
}

/// Port of `_expand_time`: 24 hour time to 12 hour words.
pub fn _expand_time(m: &PyMatch<'_>) -> Result<String, Error> {
  let mut hours: i64 = py_int(m.group_str(1))?.parse().expect("two digits");
  let minutes: i64 = py_int(m.group_str(2))?.parse().expect("two digits");
  let period = if hours < 12 { "a.m." } else { "p.m." };
  if hours > 12 {
    hours -= 12;
  }

  let hour_word = number_to_words(hours.into())?;
  let minute_word = if minutes != 0 {
    number_to_words(minutes.into())?
  } else {
    String::new()
  };

  if minutes == 0 {
    Ok(format!("{hour_word} o'clock {period}"))
  } else {
    Ok(format!("{hour_word} {minute_word} {period}"))
  }
}

/// Port of `_expand_measurement`: `"1m"` becomes `"1 meter"`, `"2m"` becomes
/// `"2 meters"`.
pub fn _expand_measurement(m: &PyMatch<'_>) -> Result<String, Error> {
  let sign = m.group_str(3);
  let mut ptr = 1;
  let num = py_int(&m.group_str(1).replace(sign, "").replace('.', ""))?;
  let decimal_part = m.group(2);
  if decimal_part.is_none() && num == "1" {
    ptr = 0;
  }
  Ok(
    m.group_str(1)
      .replace(sign, &format!(" {}", measurement_map(sign)[ptr])),
  )
}

/// `s.ljust(2, "0")`.
fn ljust2(s: &str) -> String {
  let mut s = s.to_string();
  while s.chars().count() < 2 {
    s.push('0');
  }
  s
}

/// The shared body of `_expand_pounds` and `_expand_dollars`, which differ
/// only in their unit words.
fn expand_currency(
  m: &PyMatch<'_>,
  units: &str,
  [unit, units_pl]: [&str; 2],
  [sub, sub_pl]: [&str; 2],
) -> Result<String, Error> {
  let matched = m.group_str(1);
  let parts: Vec<&str> = matched.split('.').collect();
  if parts.len() > 2 {
    return Ok(format!("{matched} {units}")); // Unexpected format
  }
  let main = if parts[0].is_empty() {
    "0".to_string()
  } else {
    py_int(parts[0])?
  };
  let minor = if parts.len() > 1 && !parts[1].is_empty() {
    py_int(&ljust2(parts[1]))?
  } else {
    "0".to_string()
  };
  let main_unit = if main == "1" { unit } else { units_pl };
  let minor_unit = if minor == "1" { sub } else { sub_pl };
  if main != "0" && minor != "0" {
    Ok(format!("{main} {main_unit} and {minor} {minor_unit}"))
  } else if main != "0" {
    Ok(format!("{main} {main_unit}"))
  } else if minor != "0" {
    Ok(format!("{minor} {minor_unit}"))
  } else {
    Ok(format!("zero {units_pl}"))
  }
}

/// Port of `_expand_pounds`.
pub fn _expand_pounds(m: &PyMatch<'_>) -> Result<String, Error> {
  expand_currency(m, "pounds", ["pound", "pounds"], ["penny", "pence"])
}

/// Port of `_expand_dollars`.
pub fn _expand_dollars(m: &PyMatch<'_>) -> Result<String, Error> {
  expand_currency(m, "dollars", ["dollar", "dollars"], ["cent", "cents"])
}

/// Port of `_expand_decimal_number`: `"13.234"` becomes
/// `"13 point 2 3 4"` (the digits are read later by `_expand_number`).
pub fn _expand_decimal_number(m: &PyMatch<'_>) -> Result<String, Error> {
  let matched = m.group_str(1);
  let parts: Vec<&str> = matched.split('.').collect();
  let mut words: Vec<String> = Vec::new();
  for char in parts[1].chars() {
    if char == '.' {
      words.push("point".to_string());
    } else {
      words.push(char.to_string());
    }
  }
  Ok(format!("{} point {}", parts[0], words.join(" ")))
}

/// Port of `_expend_fraction`: `"3/4"` becomes `"three fourths"`, `"1/2"`
/// becomes `"one half"`.
pub fn _expend_fraction(m: &PyMatch<'_>) -> Result<String, Error> {
  let matched = m.group0();
  let (num_s, den_s) = matched.split_once('/').expect("fraction has a slash");
  let numerator = py_int(num_s)?;
  let denominator = py_int(den_s)?;

  let numerator_part = number_to_words(int_arg(&numerator))?;
  let denominator_part;
  if denominator == "2" {
    if numerator == "1" {
      denominator_part = "half".to_string();
    } else {
      denominator_part = "halves".to_string();
    }
  } else if denominator == "1" {
    return Ok(numerator_part);
  } else {
    let mut part = _inflect().ordinal(number_to_words(int_arg(&denominator))?.as_str())?;
    // numerator > 1
    if numerator != "0" && numerator != "1" {
      part.push('s');
    }
    denominator_part = part;
  }

  Ok(format!("{numerator_part} {denominator_part}"))
}

/// Port of `_expand_ordinal`: `"21st"` becomes `"twenty-first"`.
pub fn _expand_ordinal(m: &PyMatch<'_>) -> Result<String, Error> {
  number_to_words(NumberOrWord::Word(m.group0()))
}

/// Port of `_expand_number`. Numbers between 1000 and 3000 are read as years.
pub fn _expand_number(m: &PyMatch<'_>) -> Result<String, Error> {
  let num_s = py_int(m.group0())?;
  let no_and = NumberToWordsOptions {
    andword: "",
    ..Default::default()
  };
  match small(&num_s) {
    Some(num) if num > 1000 && num < 3000 => {
      if num == 2000 {
        Ok("two thousand".to_string())
      } else if num > 2000 && num < 2010 {
        Ok(format!(
          "two thousand {}",
          number_to_words((num % 100).into())?
        ))
      } else if num % 100 == 0 {
        Ok(format!("{} hundred", number_to_words((num / 100).into())?))
      } else {
        let opts = NumberToWordsOptions {
          andword: "",
          zero: "oh",
          group: 2,
          ..Default::default()
        };
        Ok(_inflect().number_to_words(num, &opts)?.replace(", ", " "))
      }
    }
    _ => Ok(_inflect().number_to_words(int_arg(&num_s), &no_and)?),
  }
}

/// Port of `asmd_map`.
pub const ASMD_MAP: &[(&str, &str)] = &[
  ("+", " plus "),
  ("-", " minus "),
  ("×", " times "),
  ("÷", " divided by "),
  ("=", " Equals "),
];

/// Port of `replace_asmd`.
pub fn replace_asmd(m: &PyMatch<'_>) -> Result<String, Error> {
  let op = m.group_str(8);
  let word = ASMD_MAP
    .iter()
    .find(|(k, _)| *k == op)
    .map(|(_, v)| *v)
    .expect("operator in asmd_map");
  Ok(format!("{}{}{}", m.group_str(1), word, m.group_str(9)))
}

/// Port of `replace_negative_num`. The whitespace before the sign is part of
/// the match and is dropped, as in Python.
pub fn replace_negative_num(m: &PyMatch<'_>) -> Result<String, Error> {
  let sign = m.group(1);
  let number = m.group_str(2);
  let sign = if sign.is_some_and(|s| !s.is_empty()) {
    "negative "
  } else {
    ""
  };
  Ok(format!("{sign}{number}"))
}

/// Port of `re.sub(r"(?<!^)(?<![\s])([A-Z])", r" \1", text)`: a space before
/// every capital letter that is not at the start and not after whitespace.
/// Both lookbehinds test the original string, so a scan over the chars is
/// equivalent.
fn split_capitals(text: &str) -> String {
  let mut out = String::with_capacity(text.len() * 2);
  let mut prev: Option<char> = None;
  for c in text.chars() {
    if c.is_ascii_uppercase() && prev.is_some_and(|p| !py_isspace(p)) {
      out.push(' ');
    }
    out.push(c);
    prev = Some(c);
  }
  out
}

/// Port of `normalize`: expands numbers, currency, times, units and math
/// operators into words, strips accents and every character other than
/// `[ A-Za-z'.,?!-]`, and splits runs of capital letters.
///
/// Returns the Python exception when `normalize` raises, for example
/// `ValueError` for `"$,5"` (`int(",5")`).
pub fn normalize(text: &str) -> Result<String, Error> {
  let mut text = _ORDINAL_NUMBER_RE.sub(text, _convert_ordinal)?;

  // 处理数学运算
  while RE_ASMD.search(&text) {
    text = RE_ASMD.sub(&text, replace_asmd)?;
  }
  text = RE_INTEGER.sub(&text, replace_negative_num)?;

  text = _COMMA_NUMBER_RE.sub(&text, _remove_commas)?;
  text = _TIME_RE.sub(&text, _expand_time)?;
  text = _MEASUREMENT_RE.sub(&text, _expand_measurement)?;
  text = _POUNDS_RE_START.sub(&text, _expand_pounds)?;
  text = _POUNDS_RE_END.sub(&text, _expand_pounds)?;
  text = _DOLLARS_RE_START.sub(&text, _expand_dollars)?;
  text = _DOLLARS_RE_END.sub(&text, _expand_dollars)?;
  text = _DECIMAL_NUMBER_RE.sub(&text, _expand_decimal_number)?;
  text = _FRACTION_RE.sub(&text, _expend_fraction)?;
  text = _ORDINAL_RE.sub(&text, _expand_ordinal)?;
  text = _NUMBER_RE.sub(&text, _expand_number)?;

  // Strip accents
  text = text.nfd().filter(|&c| !py_category_is_mn(c)).collect();

  text = _PERCENT_RE.sub(&text, |_| Ok(" percent".to_string()))?;
  text = _NON_ALPHA_RE.sub(&text, |_| Ok(String::new()))?;
  text = _IE_RE.sub(&text, |_| Ok("that is".to_string()))?;
  text = _EG_RE.sub(&text, |_| Ok("for example".to_string()))?;
  // 增加纯大写单词拆分
  Ok(split_capitals(&text))
}

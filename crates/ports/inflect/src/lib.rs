//! Subset port of the Python inflect 7.6.0 package (`number_to_words`,
//! `ordinal`) used by GPT-SoVITS.
//!
//! Ports `inflect/__init__.py`. Only `engine.number_to_words` and
//! `engine.ordinal`, and the parts of the module that they call, are ported.
//! See the crate README for the scope.
//!
#![doc = include_str!("../README.md")]

mod pystr;

use std::fmt;

/// Port of `inflect.nth`: the ordinal suffix for `n % 100` or `n % 10`.
fn nth(n: u32) -> Option<&'static str> {
  Some(match n {
    0 => "th",
    1 => "st",
    2 => "nd",
    3 => "rd",
    4..=9 => "th",
    11..=13 => "th",
    _ => return None,
  })
}

/// Port of `inflect.nth_suff`: the set of values of `nth`.
const NTH_SUFF: [&str; 4] = ["th", "st", "nd", "rd"];

/// Port of `inflect.ordinal`: word endings and their ordinal forms, in the
/// order of the Python dict (which is the alternation order of
/// `ordinal_suff`).
const ORDINAL: [(&str, &str); 8] = [
  ("ty", "tieth"),
  ("one", "first"),
  ("two", "second"),
  ("three", "third"),
  ("five", "fifth"),
  ("eight", "eighth"),
  ("nine", "ninth"),
  ("twelve", "twelfth"),
];

/// Port of `inflect.unit`.
const UNIT: [&str; 10] = [
  "", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
];

/// Port of `inflect.teen`.
const TEEN: [&str; 10] = [
  "ten",
  "eleven",
  "twelve",
  "thirteen",
  "fourteen",
  "fifteen",
  "sixteen",
  "seventeen",
  "eighteen",
  "nineteen",
];

/// Port of `inflect.ten`.
const TEN: [&str; 10] = [
  "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
];

/// Port of `inflect.mill`.
const MILL: [&str; 12] = [
  " ",
  " thousand",
  " million",
  " billion",
  " trillion",
  " quadrillion",
  " quintillion",
  " sextillion",
  " septillion",
  " octillion",
  " nonillion",
  " decillion",
];

/// Errors raised by the ported code. Variants are named after the Python
/// exception classes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
  /// Port of `inflect.BadNumValueError` (raised by `ordinal`).
  BadNumValueError,
  /// Port of `inflect.BadChunkingOptionError` (`group` not in `0..=3`).
  BadChunkingOptionError,
  /// Port of `inflect.NumOutOfRangeError` (a number above the decillions).
  NumOutOfRangeError,
  /// Python `IndexError`, which `number_to_words` raises for some inputs,
  /// such as a string that is only whitespace.
  IndexError,
  /// `typeguard.TypeCheckError`: the `@typechecked` decorator rejects an
  /// empty string, which is not an `inflect.Word`.
  TypeCheckError,
}

impl Error {
  /// The Python exception class name.
  pub fn name(self) -> &'static str {
    match self {
      Error::BadNumValueError => "BadNumValueError",
      Error::BadChunkingOptionError => "BadChunkingOptionError",
      Error::NumOutOfRangeError => "NumOutOfRangeError",
      Error::IndexError => "IndexError",
      Error::TypeCheckError => "TypeCheckError",
    }
  }
}

impl fmt::Display for Error {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.write_str(self.name())
  }
}

impl std::error::Error for Error {}

/// Port of the `Number | Word` argument type of `number_to_words` and
/// `ordinal`.
///
/// Python `int` values are `Number`. Python `float` values are not ported.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumberOrWord<'a> {
  /// A Python `int`.
  Number(i128),
  /// A Python `str` (`inflect.Word` requires at least one character).
  Word(&'a str),
}

macro_rules! number_from {
  ($($t:ty),*) => {
    $(impl From<$t> for NumberOrWord<'_> {
      fn from(value: $t) -> Self {
        NumberOrWord::Number(value as i128)
      }
    })*
  };
}
number_from!(i8, i16, i32, i64, i128, u8, u16, u32, u64, isize, usize);

impl<'a> From<&'a str> for NumberOrWord<'a> {
  fn from(value: &'a str) -> Self {
    NumberOrWord::Word(value)
  }
}

impl<'a> From<&'a String> for NumberOrWord<'a> {
  fn from(value: &'a String) -> Self {
    NumberOrWord::Word(value)
  }
}

impl NumberOrWord<'_> {
  /// `str(num)`, after the `@typechecked` check that a `Word` is not empty.
  fn to_py_str(self) -> Result<String, Error> {
    match self {
      NumberOrWord::Number(n) => Ok(n.to_string()),
      NumberOrWord::Word("") => Err(Error::TypeCheckError),
      NumberOrWord::Word(s) => Ok(s.to_string()),
    }
  }
}

/// Keyword arguments of `engine.number_to_words`.
///
/// `wantlist` (always `False`) and `threshold` (always `None`) are not
/// ported.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NumberToWordsOptions<'a> {
  /// `group`: 0 (default), or 1, 2 or 3 to read the digits in groups.
  pub group: i64,
  /// `comma`: the separator between thousands groups (default `","`). An
  /// empty string stands for every falsy Python value.
  pub comma: &'a str,
  /// `andword`: the word between hundreds and tens (default `"and"`).
  pub andword: &'a str,
  /// `zero`: the word for 0 (default `"zero"`).
  pub zero: &'a str,
  /// `one`: the word for 1 (default `"one"`).
  pub one: &'a str,
  /// `decimal`: the word for the decimal point (default `Some("point")`).
  /// `None` is Python `None`. `Some("")` is the empty string, which is falsy
  /// like `None` but behaves differently in `_render`.
  pub decimal: Option<&'a str>,
}

impl Default for NumberToWordsOptions<'_> {
  fn default() -> Self {
    Self {
      group: 0,
      comma: ",",
      andword: "and",
      zero: "zero",
      one: "one",
      decimal: Some("point"),
    }
  }
}

/// Port of `engine._number_args`.
#[derive(Debug, Clone, Default)]
struct NumberArgs {
  andword: String,
  zero: String,
  one: String,
}

/// Port of `inflect.engine` (only the numerical inflections).
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Default)]
pub struct engine {
  /// Port of `engine._number_args`.
  number_args: NumberArgs,
  /// Port of `engine.mill_count`.
  mill_count: usize,
}

/// Port of `inflect._grouped_from_right`.
fn grouped_from_right(digits: &[char], size: usize) -> Vec<&[char]> {
  // Python: head = (len(digits) - 1) % size + 1, with a floor modulo.
  let head = ((digits.len() as i64 - 1).rem_euclid(size as i64) + 1) as usize;
  let head = head.min(digits.len());
  let mut out = vec![&digits[..head]];
  let mut i = head;
  while i < digits.len() {
    out.push(&digits[i..(i + size).min(digits.len())]);
    i += size;
  }
  out
}

/// `PATTERN.sub(repl, s)` for a pattern of `n` consecutive `\d` groups
/// (`DIGIT_GROUP`, `TWO_DIGITS`, `THREE_DIGITS`).
fn sub_digit_runs(
  s: &[char],
  n: usize,
  mut repl: impl FnMut(&[u32]) -> Result<String, Error>,
) -> Result<String, Error> {
  let mut out = String::new();
  let mut i = 0;
  while i < s.len() {
    if i + n <= s.len() && s[i..i + n].iter().all(|&c| pystr::is_digit(c)) {
      let values: Vec<u32> = s[i..i + n]
        .iter()
        .filter_map(|&c| pystr::decimal(c))
        .collect();
      out.push_str(&repl(&values)?);
      i += n;
    } else {
      out.push(s[i]);
      i += 1;
    }
  }
  Ok(out)
}

/// `PATTERN.sub(repl, s, 1)` for `THREE_DIGITS_WORD`, `TWO_DIGITS_WORD` and
/// `ONE_DIGIT_WORD`: `n` digits followed by `\D*\Z`. Returns `None` when the
/// pattern does not match.
fn sub_tail_digits(
  s: &str,
  n: usize,
  repl: impl FnOnce(&[u32]) -> Result<String, Error>,
) -> Result<Option<String>, Error> {
  let chars: Vec<char> = s.chars().collect();
  // Only the last n digits of the string can be followed by `\D*\Z`.
  let Some(last) = chars.iter().rposition(|&c| pystr::is_digit(c)) else {
    return Ok(None);
  };
  if last + 1 < n {
    return Ok(None);
  }
  let start = last + 1 - n;
  if !chars[start..=last].iter().all(|&c| pystr::is_digit(c)) {
    return Ok(None);
  }
  let values: Vec<u32> = chars[start..=last]
    .iter()
    .filter_map(|&c| pystr::decimal(c))
    .collect();
  let mut out: String = chars[..start].iter().collect();
  out.push_str(&repl(&values)?);
  out.extend(&chars[last + 1..]);
  Ok(Some(out))
}

/// Port of `inflect.WHITESPACES_COMMA.sub(",", s)` (`\s+,`).
fn whitespaces_comma_sub(s: &str) -> String {
  let chars: Vec<char> = s.chars().collect();
  let mut out = String::new();
  let mut i = 0;
  while i < chars.len() {
    if pystr::is_space(chars[i]) {
      let mut j = i;
      while j < chars.len() && pystr::is_space(chars[j]) {
        j += 1;
      }
      if j < chars.len() && chars[j] == ',' {
        out.push(',');
        i = j + 1;
      } else {
        out.extend(&chars[i..j]);
        i = j;
      }
    } else {
      out.push(chars[i]);
      i += 1;
    }
  }
  out
}

/// Port of `inflect.WHITESPACES.sub(" ", s)` (`\s+`).
fn whitespaces_sub(s: &str) -> String {
  let mut out = String::new();
  let mut in_space = false;
  for ch in s.chars() {
    if pystr::is_space(ch) {
      if !in_space {
        out.push(' ');
      }
      in_space = true;
    } else {
      out.push(ch);
      in_space = false;
    }
  }
  out
}

/// Port of `inflect.COMMA_WORD.sub(f" {andword} \\1", s)`
/// (`, (\S+)\s+\Z`).
///
/// `andword` is used literally. Python reads it as part of a `re`
/// replacement template, so an `andword` with a backslash differs (not
/// ported).
fn comma_word_sub(s: &str, andword: &str) -> String {
  let chars: Vec<char> = s.chars().collect();
  for i in 0..chars.len() {
    if chars[i] != ',' || chars.get(i + 1) != Some(&' ') {
      continue;
    }
    let word_start = i + 2;
    let mut word_end = word_start;
    while word_end < chars.len() && !pystr::is_space(chars[word_end]) {
      word_end += 1;
    }
    if word_end == word_start || word_end == chars.len() {
      continue;
    }
    if chars[word_end..].iter().all(|&c| pystr::is_space(c)) {
      let mut out: String = chars[..i].iter().collect();
      out.push(' ');
      out.push_str(andword);
      out.push(' ');
      out.extend(&chars[word_start..word_end]);
      return out;
    }
  }
  s.to_string()
}

/// Port of `inflect.DIGIT.match(s)` (`\d` at the start of `s`).
fn digit_match(s: &str) -> bool {
  s.chars().next().is_some_and(pystr::is_digit)
}

impl engine {
  /// Port of `inflect.engine()`.
  pub fn new() -> Self {
    Self::default()
  }

  /// Port of `engine.ordinal`.
  ///
  /// Returns the ordinal of `num`: `1` becomes `"1st"` and `"one"` becomes
  /// `"first"`.
  pub fn ordinal<'a>(&self, num: impl Into<NumberOrWord<'a>>) -> Result<String, Error> {
    let num = num.into();
    let str_num = num.to_py_str()?;
    if digit_match(&str_num) || (str_num.starts_with('-') && digit_match(&str_num[1..])) {
      let n = match num {
        // `isinstance(num, (float, int)) and int(num) == num`
        NumberOrWord::Number(n) => (n.unsigned_abs() % 100) as u32,
        NumberOrWord::Word(_) => {
          if str_num.contains('.') {
            // numbers after decimal, so only need last one for ordinal
            pystr::int_abs_mod_100(pystr::last_chars(&str_num, 1))
              // ends with '.', so need to use whole string
              .or_else(|| pystr::int_abs_mod_100(pystr::drop_last_chars(&str_num, 1)))
              .ok_or(Error::BadNumValueError)?
          } else {
            pystr::int_abs_mod_100(&str_num).ok_or(Error::BadNumValueError)?
          }
        }
      };
      let post = nth(n % 100)
        .or_else(|| nth(n % 10))
        .expect("n % 10 is in nth");
      Ok(format!("{str_num}{post}"))
    } else {
      Ok(Self::_sub_ord(&str_num))
    }
  }

  /// Port of `engine.millfn`.
  fn millfn(&self, ind: usize) -> Result<&'static str, Error> {
    MILL.get(ind).copied().ok_or(Error::NumOutOfRangeError)
  }

  /// Port of `engine.unitfn`.
  fn unitfn(&self, units: u32, mindex: usize) -> Result<String, Error> {
    Ok(format!("{}{}", UNIT[units as usize], self.millfn(mindex)?))
  }

  /// Port of `engine.tenfn`.
  fn tenfn(&self, tens: u32, units: u32, mindex: usize) -> Result<String, Error> {
    if tens != 1 {
      let tens_part = TEN[tens as usize];
      let hyphen = if tens != 0 && units != 0 { "-" } else { "" };
      let unit_part = UNIT[units as usize];
      let mill_part = self.millfn(mindex)?;
      return Ok(format!("{tens_part}{hyphen}{unit_part}{mill_part}"));
    }
    Ok(format!("{}{}", TEEN[units as usize], self.millfn(mindex)?))
  }

  /// Port of `engine.hundfn`.
  fn hundfn(&self, hundreds: u32, tens: u32, units: u32, mindex: usize) -> Result<String, Error> {
    if hundreds != 0 {
      let andword = if tens != 0 || units != 0 {
        format!(" {} ", self.number_args.andword)
      } else {
        String::new()
      };
      // use unit not unitfn as simpler
      return Ok(format!(
        "{} hundred{andword}{}{}, ",
        UNIT[hundreds as usize],
        self.tenfn(tens, units, 0)?,
        self.millfn(mindex)?
      ));
    }
    if tens != 0 || units != 0 {
      return Ok(format!(
        "{}{}, ",
        self.tenfn(tens, units, 0)?,
        self.millfn(mindex)?
      ));
    }
    Ok(String::new())
  }

  /// Port of `engine.group1sub`.
  fn group1sub(&self, units: u32) -> String {
    if units == 1 {
      format!(" {}, ", self.number_args.one)
    } else if units != 0 {
      format!("{}, ", UNIT[units as usize])
    } else {
      format!(" {}, ", self.number_args.zero)
    }
  }

  /// Port of `engine.group1bsub`.
  fn group1bsub(&self, units: u32) -> String {
    if units != 0 {
      format!("{}, ", UNIT[units as usize])
    } else {
      format!(" {}, ", self.number_args.zero)
    }
  }

  /// Port of `engine.group2sub`.
  fn group2sub(&self, tens: u32, units: u32) -> Result<String, Error> {
    let zero = &self.number_args.zero;
    if tens != 0 {
      return Ok(format!("{}, ", self.tenfn(tens, units, 0)?));
    }
    if units != 0 {
      return Ok(format!(" {zero} {}, ", UNIT[units as usize]));
    }
    Ok(format!(" {zero} {zero}, "))
  }

  /// Port of `engine.group3sub`.
  fn group3sub(&self, hundreds: u32, tens: u32, units: u32) -> Result<String, Error> {
    let zero = &self.number_args.zero;
    let hunword = if hundreds == 1 {
      format!(" {}", self.number_args.one)
    } else if hundreds != 0 {
      UNIT[hundreds as usize].to_string()
    } else {
      format!(" {zero}")
    };
    let tenword = if tens != 0 {
      self.tenfn(tens, units, 0)?
    } else if units != 0 {
      format!(" {zero} {}", UNIT[units as usize])
    } else {
      format!(" {zero} {zero}")
    };
    Ok(format!("{hunword} {tenword}, "))
  }

  /// Port of `engine.hundsub`.
  fn hundsub(&mut self, digits: &[u32]) -> Result<String, Error> {
    let ret = self.hundfn(digits[0], digits[1], digits[2], self.mill_count)?;
    self.mill_count += 1;
    Ok(ret)
  }

  /// Port of `engine.tensub`.
  fn tensub(&self, digits: &[u32]) -> Result<String, Error> {
    Ok(format!(
      "{}, ",
      self.tenfn(digits[0], digits[1], self.mill_count)?
    ))
  }

  /// Port of `engine.unitsub`.
  fn unitsub(&self, digits: &[u32]) -> Result<String, Error> {
    Ok(format!("{}, ", self.unitfn(digits[0], self.mill_count)?))
  }

  /// Port of `engine.enword`. `num` contains only decimal digits (the
  /// caller removes everything else) and is not empty.
  fn enword(&mut self, num: &str, group: i64) -> Result<String, Error> {
    let chars: Vec<char> = num.chars().collect();
    if group == 1 {
      return sub_digit_runs(&chars, 1, |d| Ok(self.group1sub(d[0])));
    }
    if group == 2 || group == 3 {
      let mut out = String::new();
      for part in grouped_from_right(&chars, group as usize) {
        out.push_str(&self._render_group(part, group)?);
      }
      return Ok(out);
    }
    // `int(num) == 0` and `int(num) == 1`
    let mut significant = chars
      .iter()
      .filter_map(|&c| pystr::decimal(c))
      .skip_while(|&d| d == 0);
    match (significant.next(), significant.next()) {
      (None, _) => return Ok(self.number_args.zero.clone()),
      (Some(1), None) => return Ok(self.number_args.one.clone()),
      _ => {}
    }
    let mut num = pystr::lstrip(num).trim_start_matches('0').to_string();
    self.mill_count = 0;
    // surely there's a better way to do the next bit
    while let Some(next) = sub_tail_digits(&num, 3, |d| self.hundsub(d))? {
      num = next;
    }
    if let Some(next) = sub_tail_digits(&num, 2, |d| self.tensub(d))? {
      num = next;
    }
    if let Some(next) = sub_tail_digits(&num, 1, |d| self.unitsub(d))? {
      num = next;
    }
    Ok(num)
  }

  /// Port of `engine._render_group`.
  fn _render_group(&self, part: &[char], group: i64) -> Result<String, Error> {
    match part.len() {
      1 if group == 2 => sub_digit_runs(part, 1, |d| Ok(self.group1bsub(d[0]))),
      1 => sub_digit_runs(part, 1, |d| Ok(self.group1sub(d[0]))),
      2 => sub_digit_runs(part, 2, |d| self.group2sub(d[0], d[1])),
      3 => sub_digit_runs(part, 3, |d| self.group3sub(d[0], d[1], d[2])),
      // `{1: ..., 2: ..., 3: ...}[len(part)]` raises KeyError; `enword`
      // never passes an empty part.
      _ => unreachable!("_render_group: part of length {}", part.len()),
    }
  }

  /// Port of `engine._sub_ord`.
  fn _sub_ord(val: &str) -> String {
    let new = match ORDINAL.iter().find(|(suffix, _)| val.ends_with(suffix)) {
      Some((suffix, ord)) => format!("{}{ord}", &val[..val.len() - suffix.len()]),
      None => val.to_string(),
    };
    if new == val { new + "th" } else { new }
  }

  /// Port of `engine._chunk_num`. Returns the chunks and whether a trailing
  /// blank chunk was removed.
  fn _chunk_num(num: &str, decimal: Option<&str>, group: i64) -> (Vec<String>, bool) {
    let chunks: Vec<String> = if decimal.is_some_and(|d| !d.is_empty()) {
      if group != 0 {
        num.split('.').map(str::to_string).collect()
      } else {
        num.splitn(2, '.').map(str::to_string).collect()
      }
    } else {
      vec![num.to_string()]
    };
    Self::_remove_last_blank(chunks)
  }

  /// Port of `engine._remove_last_blank`.
  fn _remove_last_blank(mut chunks: Vec<String>) -> (Vec<String>, bool) {
    let removed = chunks.last().is_some_and(String::is_empty);
    if removed {
      chunks.pop();
    }
    (chunks, removed)
  }

  /// Port of `engine._get_sign`.
  fn _get_sign(num: &str) -> Result<&'static str, Error> {
    match pystr::lstrip(num).chars().next() {
      Some('+') => Ok("plus"),
      Some('-') => Ok("minus"),
      Some(_) => Ok(""),
      None => Err(Error::IndexError),
    }
  }

  /// Port of `engine.number_to_words` with `wantlist=False` and
  /// `threshold=None`.
  ///
  /// Returns `num` in words, such as `"one hundred and one"` for `101`.
  pub fn number_to_words<'a>(
    &mut self,
    num: impl Into<NumberOrWord<'a>>,
    options: &NumberToWordsOptions<'_>,
  ) -> Result<String, Error> {
    let NumberToWordsOptions {
      group,
      comma,
      andword,
      zero,
      one,
      decimal,
    } = *options;
    let num = num.into().to_py_str()?;
    self.number_args = NumberArgs {
      andword: andword.to_string(),
      zero: zero.to_string(),
      one: one.to_string(),
    };

    if !(0..=3).contains(&group) {
      return Err(Error::BadChunkingOptionError);
    }

    let sign = Self::_get_sign(&num)?;

    let mut num = num;
    if NTH_SUFF.contains(&num.as_str()) {
      num = zero.to_string();
    }

    let myord = NTH_SUFF.contains(&pystr::last_chars(&num, 2));
    if myord {
      num = pystr::drop_last_chars(&num, 2).to_string();
    }

    let (mut chunks, finalpoint) = Self::_chunk_num(&num, decimal, group);

    let loopstart = chunks.first().ok_or(Error::IndexError)?.is_empty();
    // Python `first: bool | None`: `Some(b)` is a bool, `None` is `None`.
    let mut first: Option<bool> = Some(!loopstart);

    let start = usize::from(loopstart);
    for chunk in chunks.iter_mut().skip(start) {
      // remove all non numeric \D
      let mut c: String = chunk.chars().filter(|&ch| pystr::is_digit(ch)).collect();
      if c.is_empty() {
        c = "0".to_string();
      }

      let is_first = first == Some(true);
      c = if group == 0 && !is_first {
        self.enword(&c, 1)?
      } else {
        self.enword(&c, group)?
      };

      if c.ends_with(", ") {
        c.truncate(c.len() - 2);
      }
      c = whitespaces_comma_sub(&c);

      if group == 0 && is_first {
        c = comma_word_sub(&c, andword);
        // A punctuation andword (e.g. ",") re-introduces a space before
        // itself; drop it again as above.
        c = whitespaces_comma_sub(&c);
      }
      c = whitespaces_sub(&c);
      c = pystr::strip(&c).to_string();
      if is_first {
        first = None;
      }
      *chunk = c;
    }

    let mut numchunks: Vec<String> = Vec::new();
    // `first != 0`: only Python `False` equals 0.
    if first != Some(false) {
      numchunks = chunks[0].split(", ").map(str::to_string).collect();
    }

    if myord && let Some(last) = numchunks.last_mut() {
      *last = Self::_sub_ord(last);
    }

    // `decimal` is truthy whenever there is more than one chunk or a final
    // point, so it is a string here.
    let decimal_str = decimal.unwrap_or_default();
    for chunk in chunks.iter().skip(1) {
      numchunks.push(decimal_str.to_string());
      numchunks.extend(chunk.split(", ").map(str::to_string));
    }

    if finalpoint {
      numchunks.push(decimal_str.to_string());
    }

    let signout = if sign.is_empty() {
      String::new()
    } else {
      format!("{sign} ")
    };
    let resolved_comma = comma;
    let valout = if group != 0 {
      numchunks.join(&format!("{resolved_comma} "))
    } else {
      Self::_render(numchunks, decimal, resolved_comma)?
    };
    Ok(signout + &valout)
  }

  /// Port of `engine._render`, joined.
  fn _render(chunks: Vec<String>, decimal: Option<&str>, comma: &str) -> Result<String, Error> {
    let mut chunks = chunks.into_iter();
    let first_item = chunks.next().ok_or(Error::IndexError)?;
    let mut first = match decimal {
      None => true,
      Some(d) => !first_item.ends_with(d),
    };
    let mut out = first_item;
    for nc in chunks {
      if Some(nc.as_str()) == decimal {
        first = false;
      } else if first {
        out.push_str(comma);
      }
      out.push(' ');
      out.push_str(&nc);
    }
    Ok(out)
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn docstring_examples() {
    let p = engine::new();
    assert_eq!(p.ordinal(1).unwrap(), "1st");
    assert_eq!(p.ordinal("one").unwrap(), "first");
    let mut p = engine::new();
    let opts = NumberToWordsOptions::default();
    assert_eq!(
      p.number_to_words(1234, &opts).unwrap(),
      "one thousand, two hundred and thirty-four"
    );
  }

  #[test]
  fn grouped() {
    let digits: Vec<char> = "1234567".chars().collect();
    let groups: Vec<String> = grouped_from_right(&digits, 3)
      .iter()
      .map(|g| g.iter().collect())
      .collect();
    assert_eq!(groups, ["1", "234", "567"]);
  }
}

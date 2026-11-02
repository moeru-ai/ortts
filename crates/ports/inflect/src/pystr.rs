//! Python `str`, `re` and `int()` primitives that the port needs.
//!
//! Python's `re` module matches `\d` against Unicode decimal digits
//! (`str.isdecimal`) and `\s` against Unicode whitespace (`str.isspace`).
//! Rust's `char` methods and the `regex` crate use different sets, so the
//! port uses these predicates instead.

use std::sync::OnceLock;

/// Ranges of Unicode decimal digits from `data/decimal_digits.tsv`, generated
/// from the pinned Python by `parity/gen_fixtures.py` (`pixi run data`).
const DECIMAL_DIGITS_TSV: &str = include_str!("../data/decimal_digits.tsv");

/// `(start, end, value_of_start)` code point ranges, sorted by `start`.
fn decimal_ranges() -> &'static [(u32, u32, u32)] {
  static RANGES: OnceLock<Vec<(u32, u32, u32)>> = OnceLock::new();
  RANGES.get_or_init(|| {
    DECIMAL_DIGITS_TSV
      .lines()
      .filter(|line| !line.is_empty() && !line.starts_with('#'))
      .map(|line| {
        let mut cols = line.split('\t');
        let mut next = || cols.next().expect("decimal_digits.tsv: missing column");
        let start = u32::from_str_radix(next(), 16).expect("decimal_digits.tsv: bad start");
        let end = u32::from_str_radix(next(), 16).expect("decimal_digits.tsv: bad end");
        let value = next().parse().expect("decimal_digits.tsv: bad value");
        (start, end, value)
      })
      .collect()
  })
}

/// Value of `ch` as a Python decimal digit (`unicodedata.decimal`), or
/// `None` when `ch` is not matched by `re`'s `\d`.
pub fn decimal(ch: char) -> Option<u32> {
  let cp = ch as u32;
  let ranges = decimal_ranges();
  let idx = ranges.partition_point(|&(start, _, _)| start <= cp);
  let &(start, end, value) = ranges.get(idx.checked_sub(1)?)?;
  (cp <= end).then(|| value + (cp - start))
}

/// `re` `\d`.
pub fn is_digit(ch: char) -> bool {
  decimal(ch).is_some()
}

/// `re` `\s` and `str.isspace`.
pub fn is_space(ch: char) -> bool {
  matches!(
    ch,
    '\t'..='\r'
      | '\x1c'..='\x1f'
      | ' '
      | '\u{85}'
      | '\u{a0}'
      | '\u{1680}'
      | '\u{2000}'..='\u{200a}'
      | '\u{2028}'
      | '\u{2029}'
      | '\u{202f}'
      | '\u{205f}'
      | '\u{3000}'
  )
}

/// `str.lstrip()`.
pub fn lstrip(s: &str) -> &str {
  s.trim_start_matches(is_space)
}

/// `str.strip()`.
pub fn strip(s: &str) -> &str {
  s.trim_matches(is_space)
}

/// `s[-n:]` on code points.
pub fn last_chars(s: &str, n: usize) -> &str {
  match s.char_indices().rev().nth(n.saturating_sub(1)) {
    Some((idx, _)) if n > 0 => &s[idx..],
    _ if n == 0 => "",
    _ => s,
  }
}

/// `s[:-n]` on code points.
pub fn drop_last_chars(s: &str, n: usize) -> &str {
  &s[..s.len() - last_chars(s, n).len()]
}

/// Whitespace that `int()` skips: CPython maps non-ASCII `str.isspace`
/// characters to an ASCII space, keeps ASCII characters as they are, and then
/// skips C `isspace` characters, so `\x1c`..`\x1f` are not skipped.
fn is_int_space(ch: char) -> bool {
  if ch.is_ascii() {
    matches!(ch, '\t'..='\r' | ' ')
  } else {
    is_space(ch)
  }
}

/// `int(s) % 100` for a Python `str` (`abs` taken), or `None` when `int(s)`
/// raises `ValueError`.
///
/// CPython maps decimal digits to ASCII digits first, then parses an
/// optional sign and digits that may be separated by single underscores,
/// with whitespace allowed around the number.
pub fn int_abs_mod_100(s: &str) -> Option<u32> {
  let chars: Vec<char> = s.chars().collect();
  let mut i = 0;
  while i < chars.len() && is_int_space(chars[i]) {
    i += 1;
  }
  if i < chars.len() && (chars[i] == '+' || chars[i] == '-') {
    i += 1;
  }
  let mut value = 0u32;
  let mut digits = 0usize;
  let mut prev_underscore = false;
  while i < chars.len() {
    if let Some(d) = decimal(chars[i]) {
      value = (value * 10 + d) % 100;
      digits += 1;
      prev_underscore = false;
    } else if chars[i] == '_' && digits > 0 && !prev_underscore {
      prev_underscore = true;
    } else {
      break;
    }
    i += 1;
  }
  if digits == 0 || prev_underscore {
    return None;
  }
  while i < chars.len() && is_int_space(chars[i]) {
    i += 1;
  }
  (i == chars.len()).then_some(value)
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn slicing() {
    assert_eq!(last_chars("abc", 2), "bc");
    assert_eq!(last_chars("a", 2), "a");
    assert_eq!(last_chars("", 2), "");
    assert_eq!(last_chars("é١x", 2), "١x");
    assert_eq!(drop_last_chars("1st", 2), "1");
    assert_eq!(drop_last_chars("x", 1), "");
  }

  #[test]
  fn digits() {
    assert_eq!(decimal('7'), Some(7));
    assert_eq!(decimal('\u{0663}'), Some(3));
    assert_eq!(decimal('a'), None);
    assert_eq!(decimal('\u{00b2}'), None);
  }
}

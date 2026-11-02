//! Helpers that reproduce the Python `re` behaviour the ported patterns rely
//! on. Not a port of a Python item.

use std::sync::OnceLock;

/// Code point ranges that Python 3.11 `re` matches with `\d` in a str
/// pattern, exported by `parity/export_data.py`.
const DECIMAL_RANGES: &str = include_str!("../data/python_decimal_ranges.txt");

fn decimal_ranges() -> &'static [(u32, u32)] {
  static RANGES: OnceLock<Vec<(u32, u32)>> = OnceLock::new();
  RANGES.get_or_init(|| {
    DECIMAL_RANGES
      .lines()
      .filter(|line| !line.is_empty() && !line.starts_with('#'))
      .map(|line| {
        let (a, b) = line.split_once('-').expect("range line");
        (
          u32::from_str_radix(a, 16).expect("range start"),
          u32::from_str_radix(b, 16).expect("range end"),
        )
      })
      .collect()
  })
}

/// Python `re` `\d` (str pattern, no `re.ASCII`).
pub(crate) fn is_digit(c: char) -> bool {
  if c.is_ascii() {
    return c.is_ascii_digit();
  }
  let cp = c as u32;
  let ranges = decimal_ranges();
  let idx = ranges.partition_point(|&(_, end)| end < cp);
  idx < ranges.len() && ranges[idx].0 <= cp
}

/// End of the `\d*` run that starts at `pos`.
pub(crate) fn digits_end(chars: &[char], pos: usize) -> usize {
  let mut end = pos;
  while end < chars.len() && is_digit(chars[end]) {
    end += 1;
  }
  end
}

/// Whether `chars[pos]` exists and equals `c`.
pub(crate) fn char_at(chars: &[char], pos: usize, c: char) -> bool {
  chars.get(pos) == Some(&c)
}

/// `re.sub(pattern, repl, string)` where `matcher(chars, pos)` returns the end
/// of the match that the pattern finds when it starts at `pos`.
///
/// The ported patterns are deterministic (backtracking never changes the
/// first match found at a position), so trying each start position in order
/// gives Python's matches. Empty matches produce no output, as the ported
/// callbacks return `None` (an empty string for `re.sub`) for empty input.
pub(crate) fn try_sub<M, R, E>(chars: &[char], matcher: M, mut repl: R) -> Result<String, E>
where
  M: Fn(&[char], usize) -> Option<usize>,
  R: FnMut(&[char]) -> Result<String, E>,
{
  let mut out = String::new();
  let mut pos = 0;
  while pos < chars.len() {
    match matcher(chars, pos) {
      Some(end) if end > pos => {
        out.push_str(&repl(&chars[pos..end])?);
        pos = end;
      }
      _ => {
        out.push(chars[pos]);
        pos += 1;
      }
    }
  }
  Ok(out)
}

/// Infallible form of [`try_sub`].
pub(crate) fn sub<M, R>(chars: &[char], matcher: M, mut repl: R) -> String
where
  M: Fn(&[char], usize) -> Option<usize>,
  R: FnMut(&[char]) -> String,
{
  try_sub::<_, _, std::convert::Infallible>(chars, matcher, |m| Ok(repl(m)))
    .unwrap_or_else(|never| match never {})
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn digits() {
    assert!(is_digit('0'));
    assert!(is_digit('\u{FF13}'));
    assert!(is_digit('\u{0663}'));
    assert!(!is_digit('a'));
    assert!(!is_digit('٫'));
    // Kawi digits were added in Unicode 15, after Python 3.11's Unicode 14.
    assert!(!is_digit('\u{11F50}'));
  }
}

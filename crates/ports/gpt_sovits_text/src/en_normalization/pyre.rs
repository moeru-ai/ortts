//! Helpers that reproduce the parts of Python's `re` and `int()` that
//! `expend.py` relies on. This module has no Python counterpart.
//!
//! - [`PyRegex`] compiles a Python pattern source unchanged. [`translate`]
//!   rewrites the constructs whose meaning differs between Python `re` and the
//!   Rust regex crates. It uses `regex` when it can and `fancy-regex` when the
//!   translated pattern needs lookarounds.
//! - [`PyMatch`] mirrors `re.Match.group()`.
//! - [`py_int`] mirrors `int(s)` for the strings `expend.py` passes to it,
//!   including the `ValueError` messages.

use std::fmt::Write as _;
use std::sync::LazyLock;

use super::Error;
use super::py_tables::{DIGIT, MN, SPACE, WORD};

/// The body of a character class (without brackets) for a range table.
fn class_body(table: &[(u32, u32)]) -> String {
  let mut out = String::new();
  for &(a, b) in table {
    if a == b {
      let _ = write!(out, "\\x{{{a:X}}}");
    } else {
      let _ = write!(out, "\\x{{{a:X}}}-\\x{{{b:X}}}");
    }
  }
  out
}

static DIGIT_BODY: LazyLock<String> = LazyLock::new(|| class_body(DIGIT));
static SPACE_BODY: LazyLock<String> = LazyLock::new(|| class_body(SPACE));
static WORD_BODY: LazyLock<String> = LazyLock::new(|| class_body(WORD));

/// Python `\b` for `str` patterns: a position between a `\w` char and a
/// non-`\w` char (or the start or end of the string).
static WORD_BOUNDARY: LazyLock<String> = LazyLock::new(|| {
  let w = format!("[{}]", *WORD_BODY);
  format!("(?:(?<={w})(?!{w})|(?<!{w})(?={w}))")
});

/// Whether `c` is in a range table.
fn in_table(table: &[(u32, u32)], c: char) -> bool {
  let c = c as u32;
  table
    .binary_search_by(|&(a, b)| {
      if b < c {
        std::cmp::Ordering::Less
      } else if a > c {
        std::cmp::Ordering::Greater
      } else {
        std::cmp::Ordering::Equal
      }
    })
    .is_ok()
}

/// Python `str.isspace()` (also `re` `\s`).
pub fn py_isspace(c: char) -> bool {
  in_table(SPACE, c)
}

/// `unicodedata.category(c) == "Mn"` (Unicode 14.0.0, as in Python 3.11).
pub fn py_category_is_mn(c: char) -> bool {
  in_table(MN, c)
}

/// Translate a Python `re` pattern source into an equivalent Rust regex source.
///
/// - `\d`, `\s` and `\w` become explicit classes of the Python 3.11 code points
///   (`str.isdecimal`, `str.isspace`, `str.isalnum` or `_`), so matching does
///   not depend on the Unicode tables of the Rust regex crates. Rust's `\s`
///   lacks U+001C..U+001F, and its `\w` differs from Python's in marks and
///   other numbers.
/// - `\b` (outside a class) becomes lookarounds on that `\w` class.
/// - `\<non-ASCII char>` (Python treats it as the literal char, Rust rejects
///   it) becomes the bare char.
/// - Inside a character class, `[`, `&` and `~` are escaped: Rust gives them a
///   meaning there (nested classes and set operations), Python does not.
pub fn translate(src: &str) -> String {
  let mut out = String::with_capacity(src.len());
  let chars: Vec<char> = src.chars().collect();
  let mut i = 0;
  let mut in_class = false;
  // Index of the first char of the current class body (after `[` and `^`).
  let mut class_body_start = 0;
  while i < chars.len() {
    let c = chars[i];
    if c == '\\' && i + 1 < chars.len() {
      let n = chars[i + 1];
      let body = match n {
        'd' => Some(&*DIGIT_BODY),
        's' => Some(&*SPACE_BODY),
        'w' => Some(&*WORD_BODY),
        _ => None,
      };
      if let Some(body) = body {
        if in_class {
          out.push_str(body);
        } else {
          out.push('[');
          out.push_str(body);
          out.push(']');
        }
      } else if n == 'b' && !in_class {
        out.push_str(&WORD_BOUNDARY);
      } else if !n.is_ascii() {
        out.push(n);
      } else {
        out.push('\\');
        out.push(n);
      }
      i += 2;
      continue;
    }
    if in_class {
      match c {
        ']' if i != class_body_start => {
          in_class = false;
          out.push(']');
        }
        '[' | '&' | '~' => {
          out.push('\\');
          out.push(c);
        }
        _ => out.push(c),
      }
      i += 1;
      continue;
    }
    if c == '[' {
      in_class = true;
      out.push('[');
      i += 1;
      if i < chars.len() && chars[i] == '^' {
        out.push('^');
        i += 1;
      }
      class_body_start = i;
      continue;
    }
    out.push(c);
    i += 1;
  }
  out
}

/// Whether a translated pattern needs `fancy-regex` (lookarounds).
fn needs_fancy(src: &str) -> bool {
  src.contains("(?=") || src.contains("(?!") || src.contains("(?<=") || src.contains("(?<!")
}

enum Engine {
  Plain(regex::Regex),
  Fancy(fancy_regex::Regex),
}

/// A compiled Python `re` pattern (`re.compile(pattern)`).
pub struct PyRegex {
  engine: Engine,
}

/// A match of a [`PyRegex`], like Python's `re.Match`.
pub struct PyMatch<'h> {
  haystack: &'h str,
  spans: Vec<Option<(usize, usize)>>,
}

impl<'h> PyMatch<'h> {
  /// `match.group(i)`: `None` when the group did not take part in the match.
  pub fn group(&self, i: usize) -> Option<&'h str> {
    self.spans[i].map(|(s, e)| &self.haystack[s..e])
  }

  /// `match.group(0)`.
  pub fn group0(&self) -> &'h str {
    self.group(0).expect("group 0 always matches")
  }

  /// `match.group(i)` for a group that always takes part in the match.
  pub fn group_str(&self, i: usize) -> &'h str {
    self
      .group(i)
      .unwrap_or_else(|| panic!("group {i} did not match"))
  }

  fn span(&self) -> (usize, usize) {
    self.spans[0].expect("group 0 always matches")
  }
}

impl PyRegex {
  /// `re.compile(pattern)`.
  pub fn compile(pattern: &str) -> Self {
    let src = translate(pattern);
    let engine = if needs_fancy(&src) {
      Engine::Fancy(
        fancy_regex::Regex::new(&src)
          .unwrap_or_else(|e| panic!("invalid pattern {pattern:?}: {e}")),
      )
    } else {
      Engine::Plain(
        regex::Regex::new(&src).unwrap_or_else(|e| panic!("invalid pattern {pattern:?}: {e}")),
      )
    };
    Self { engine }
  }

  /// All non-overlapping matches, left to right (`pattern.finditer`). None of
  /// the `expend.py` patterns can match the empty string, so the Python and
  /// Rust rules for empty matches never come into play.
  pub fn finditer<'h>(&self, text: &'h str) -> Vec<PyMatch<'h>> {
    match &self.engine {
      Engine::Plain(re) => re
        .captures_iter(text)
        .map(|c| PyMatch {
          haystack: text,
          spans: c.iter().map(|m| m.map(|m| (m.start(), m.end()))).collect(),
        })
        .collect(),
      Engine::Fancy(re) => re
        .captures_iter(text)
        .map(|c| {
          let c = c.expect("fancy-regex backtrack limit");
          PyMatch {
            haystack: text,
            spans: c.iter().map(|m| m.map(|m| (m.start(), m.end()))).collect(),
          }
        })
        .collect(),
    }
  }

  /// `pattern.search(text) is not None`.
  pub fn search(&self, text: &str) -> bool {
    match &self.engine {
      Engine::Plain(re) => re.is_match(text),
      Engine::Fancy(re) => re.is_match(text).expect("fancy-regex backtrack limit"),
    }
  }

  /// `pattern.sub(repl, text)` with a function `repl`. An error from `repl`
  /// propagates, as a Python exception does.
  pub fn sub<F>(&self, text: &str, mut repl: F) -> Result<String, Error>
  where
    F: FnMut(&PyMatch<'_>) -> Result<String, Error>,
  {
    let mut out = String::with_capacity(text.len());
    let mut last = 0;
    for m in self.finditer(text) {
      let (s, e) = m.span();
      out.push_str(&text[last..s]);
      out.push_str(&repl(&m)?);
      last = e;
    }
    out.push_str(&text[last..]);
    Ok(out)
  }
}

/// CPython's `sys.int_info.default_max_str_digits` (Python 3.11).
const MAX_STR_DIGITS: usize = 4300;

/// `repr(s)` for a string of printable ASCII without quotes or backslashes,
/// which is all `expend.py` can pass to `int()`.
fn py_repr(s: &str) -> String {
  debug_assert!(
    s.chars()
      .all(|c| c.is_ascii_graphic() && c != '\'' && c != '\\')
  );
  format!("'{s}'")
}

/// `int(s)` for a string of ASCII digits, `.`, `,` and `(`, which is all
/// `expend.py` passes to `int()`. Returns `str(int(s))`: the digits without
/// leading zeros.
///
/// The errors match CPython 3.11: `int()` first reads the leading digits, then
/// checks the 4300 digit limit, then rejects any other character. The
/// `invalid literal` message truncates the `repr` to 200 characters
/// (`%.200R`).
pub fn py_int(s: &str) -> Result<String, Error> {
  let n_digits = s.bytes().take_while(u8::is_ascii_digit).count();
  if n_digits == 0 || n_digits < s.len() && n_digits <= MAX_STR_DIGITS {
    let repr: String = py_repr(s).chars().take(200).collect();
    return Err(Error::ValueError(format!(
      "invalid literal for int() with base 10: {repr}"
    )));
  }
  if n_digits > MAX_STR_DIGITS {
    return Err(Error::ValueError(format!(
      "Exceeds the limit ({MAX_STR_DIGITS} digits) for integer string conversion: value has \
       {n_digits} digits; use sys.set_int_max_str_digits() to increase the limit"
    )));
  }
  let trimmed = s.trim_start_matches('0');
  Ok(if trimmed.is_empty() {
    "0".to_string()
  } else {
    trimmed.to_string()
  })
}

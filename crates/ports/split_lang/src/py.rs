//! Python `str` semantics that the ported code relies on (Python 3.11,
//! Unicode 14.0).

use std::sync::LazyLock;

/// Python `str.isspace()` for one character: bidirectional class `WS`, `B`
/// or `S`, or category `Zs`. This is Rust's `White_Space` set plus
/// U+001C..U+001F.
pub(crate) fn isspace(c: char) -> bool {
  c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

static ISDIGIT: LazyLock<Vec<(u32, u32)>> = LazyLock::new(|| {
  include_str!("../data/python_isdigit.txt")
    .lines()
    .filter(|l| !l.is_empty())
    .map(|l| {
      let (a, b) = l.split_once("..").expect("range");
      (
        u32::from_str_radix(a, 16).expect("hex"),
        u32::from_str_radix(b, 16).expect("hex"),
      )
    })
    .collect()
});

/// Python `str.isdigit()` for one character (`Numeric_Type` `Digit` or
/// `Decimal`, from `data/python_isdigit.txt`).
pub(crate) fn isdigit(c: char) -> bool {
  let cp = c as u32;
  let table = &*ISDIGIT;
  let i = table.partition_point(|&(_, end)| end < cp);
  i < table.len() && table[i].0 <= cp
}

/// Python `str.strip()` without arguments.
pub(crate) fn strip(text: &str) -> &str {
  text.trim_matches(isspace)
}

/// Python `len(str)`.
pub(crate) fn len(text: &str) -> usize {
  text.chars().count()
}

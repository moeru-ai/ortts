//! Checks the CPython builtin emulation in `nltk::python`.

mod common;

use std::collections::HashMap;

use nltk::python;
use serde::Deserialize;

use common::{all_chars, read_json};

#[derive(Deserialize)]
struct LowerFixture {
  chars: Vec<(u32, String)>,
  strings: Vec<(String, String)>,
}

#[test]
fn lower_matches_python() {
  let fixture: LowerFixture = read_json("python_lower.json");
  let table: HashMap<u32, String> = fixture.chars.into_iter().collect();
  for c in all_chars() {
    let expected = table
      .get(&(c as u32))
      .cloned()
      .unwrap_or_else(|| c.to_string());
    assert_eq!(python::lower(&c.to_string()), expected, "{:#x}", c as u32);
  }
  for (s, expected) in &fixture.strings {
    assert_eq!(&python::lower(s), expected, "{s:?}");
  }
  eprintln!(
    "python_lower.json: {} code points and {} strings, all exact",
    0x110000 - 0x800,
    fixture.strings.len()
  );
}

#[test]
fn lower_final_sigma() {
  // CPython applies the Final_Sigma rule, like Rust.
  assert_eq!(python::lower("ΟΔΥΣΣΕΥΣ"), "οδυσσευς");
  assert_eq!(python::lower("Σ"), "σ");
  assert_eq!(python::lower("ΑΣ."), "ας.");
  assert_eq!(python::lower("İ"), "i\u{307}");
}

#[derive(Deserialize)]
struct StrTables {
  isdigit: Vec<(u32, u32)>,
  isspace: Vec<(u32, u32)>,
  decimal_zeros: Vec<u32>,
}

#[test]
fn str_tables() {
  let t: StrTables = serde_json::from_str(include_str!("../data/python_str.json")).unwrap();
  for c in all_chars() {
    let cp = c as u32;
    assert_eq!(python::char_isdigit(c), common::in_ranges(&t.isdigit, cp));
    assert_eq!(python::char_isspace(c), common::in_ranges(&t.isspace, cp));
    let dec = t
      .decimal_zeros
      .iter()
      .find(|&&z| z <= cp && cp < z + 10)
      .map(|z| cp - z);
    assert_eq!(python::decimal(c), dec, "{cp:#x}");
  }
  assert!(python::isdigit("\u{0661}\u{0662}\u{00b2}"));
  assert!(!python::isdigit(""));
  assert!(!python::isdigit("1a"));
}

#[test]
fn int_matches_python() {
  // Cases checked against CPython 3.11 `int(text, base)`.
  let cases: &[(&str, u32, Option<i64>)] = &[
    ("\x1c65", 10, None),
    (" 65", 10, Some(65)),
    (" 65", 16, Some(101)),
    ("65 ", 10, Some(65)),
    ("0x41", 10, None),
    ("0x41", 16, Some(65)),
    ("0X_41", 16, Some(65)),
    ("0x", 16, None),
    ("_41", 10, None),
    ("4_1", 10, Some(41)),
    ("4_1", 16, Some(65)),
    ("4__1", 10, None),
    ("41_", 16, None),
    ("+41", 10, Some(41)),
    ("-41", 16, Some(-65)),
    ("\u{0664}\u{0661}", 10, Some(41)),
    ("\u{0664}\u{0661}", 16, Some(65)),
    ("\u{3000}", 10, None),
    ("1\u{3000}41", 10, None),
    ("\x7f1", 10, None),
    ("ff", 10, None),
    ("ff", 16, Some(255)),
    ("0x0x1", 16, None),
    ("\u{3000}41\u{3000}", 10, Some(41)),
    ("\u{2028}1", 16, Some(1)),
    ("", 10, None),
    ("+", 10, None),
  ];
  for &(text, base, expected) in cases {
    assert_eq!(
      python::int(text, base).ok(),
      expected,
      "int({text:?}, {base})"
    );
  }
  assert!(python::int(&"1".repeat(4301), 10).is_err());
  assert_eq!(python::int(&"f".repeat(30), 16), Ok(i64::MAX));
}

#[test]
fn cp1252() {
  assert_eq!(python::cp1252_c1(0x80), Some('\u{20ac}'));
  assert_eq!(python::cp1252_c1(0x81), None);
  assert_eq!(python::cp1252_c1(0x9f), Some('\u{178}'));
  assert_eq!(python::cp1252_c1(0x7f), None);
  assert_eq!(python::name2codepoint().len(), 252);
}

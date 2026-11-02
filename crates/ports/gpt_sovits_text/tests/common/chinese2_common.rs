//! Helpers shared by `tests/chinese2.rs` and `tests/chinese2_g2pw.rs`. It lives in
//! `tests/common/` so Cargo does not build it as its own test target.

#![allow(dead_code)]

use gpt_sovits_text::chinese2::{Chinese2, Error};
use serde_json::{Value, json};

pub fn lines(src: &str) -> Vec<Value> {
  src
    .lines()
    .filter(|l| !l.is_empty())
    .map(|l| serde_json::from_str(l).unwrap())
    .collect()
}

pub fn strings(v: &Value) -> Vec<String> {
  v.as_array()
    .unwrap()
    .iter()
    .map(|s| s.as_str().unwrap().to_string())
    .collect()
}

/// `{"e": [type, args]}` as the generator writes it. G2PW errors are
/// checked by name only (their messages are checked by `tests/g2pw.rs`), so
/// their args are `null` here and [`same`] ignores the Python args.
pub fn error_json(e: &Error) -> Value {
  let args = match e {
    Error::AssertionError(args) => json!(args),
    Error::IndexError(m) | Error::TypeError(m) | Error::ValueError(m) => json!([m]),
    Error::KeyError(k) => json!([k]),
    Error::G2PW(_) => Value::Null,
  };
  json!({"e": [e.name().unwrap_or("<port error>"), args]})
}

/// `g2p(text)` in the fixture format.
pub fn g2p_json(c: &Chinese2, text: &str) -> Value {
  match c.g2p(text) {
    Ok((phones, word2ph)) => {
      let w: String = word2ph.iter().map(|w| w.to_string()).collect();
      json!({"o": [phones.join(" "), w]})
    }
    Err(e) => error_json(&e),
  }
}

/// Exact equality, except that a `null` args of a port error matches any
/// Python args (see [`error_json`]).
pub fn same(expected: &Value, got: &Value) -> bool {
  if got["e"].is_array() && got["e"][1].is_null() {
    return expected["e"].is_array() && expected["e"][0] == got["e"][0];
  }
  expected == got
}

/// Checks every record, prints the first mismatches and the count, and
/// fails when there is any.
pub fn check_all(name: &str, records: &[Value], mut f: impl FnMut(&Value) -> (Value, Value)) {
  let mut bad = 0;
  for r in records {
    let (expected, got) = f(r);
    if !same(&expected, &got) {
      bad += 1;
      if bad <= 10 {
        eprintln!(
          "{name} mismatch for {}:\n  python: {expected}\n  rust:   {got}",
          r["t"]
            .as_str()
            .map_or_else(|| r["a"].to_string(), |t| format!("{t:?}"))
        );
      }
    }
  }
  eprintln!("{name}: {}/{} exact", records.len() - bad, records.len());
  assert_eq!(bad, 0, "{name}: {bad} mismatches");
}

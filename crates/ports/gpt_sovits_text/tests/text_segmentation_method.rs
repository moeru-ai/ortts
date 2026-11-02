//! Parity tests for `TTS_infer_pack/text_segmentation_method.py`.
//!
//! Fixtures: `parity/text_segmentation_method/gen_fixtures.py`.

use gpt_sovits_text::tts_infer_pack::text_segmentation_method::{
  self as tsm, IndexError, SplitBigTextOptions,
};
use serde_json::{Value, json};

const CASES: &str = include_str!("fixtures/text_segmentation_method/cases.jsonl");
const CONSTANTS: &str = include_str!("fixtures/text_segmentation_method/constants.json");

/// Converts a Rust result to the fixture encoding: the value, or
/// `{"error": "<exception name>"}`.
fn encode<T: serde::Serialize>(r: Result<T, IndexError>) -> Value {
  match r {
    Ok(v) => json!(v),
    Err(_) => json!({ "error": "IndexError" }),
  }
}

fn sorted_chars(set: &[char]) -> Vec<String> {
  let mut v: Vec<String> = set.iter().map(|c| c.to_string()).collect();
  v.sort();
  v
}

#[test]
fn constants() {
  let c: Value = serde_json::from_str(CONSTANTS).unwrap();
  assert_eq!(json!(sorted_chars(tsm::PUNCTUATION)), c["punctuation"]);
  assert_eq!(json!(sorted_chars(tsm::SPLITS)), c["splits"]);
  assert_eq!(json!(tsm::get_method_names()), c["method_names"]);
  for (name, message) in c["get_method_errors"].as_object().unwrap() {
    let Err(err) = tsm::get_method(name) else {
      panic!("{name:?} should be unknown");
    };
    assert_eq!(err.0, message.as_str().unwrap());
  }
}

#[test]
fn cases() {
  let names = ["cut0", "cut1", "cut2", "cut3", "cut4", "cut5"];
  let mut total = 0;
  let mut matched = std::collections::BTreeMap::<&str, usize>::new();
  let mut failures = Vec::new();
  for line in CASES.lines() {
    let case: Value = serde_json::from_str(line).unwrap();
    let text = case["text"].as_str().unwrap();
    total += 1;

    let mut check = |key: &'static str, got: Value| {
      if got == case[key] {
        *matched.entry(key).or_default() += 1;
      } else if failures.len() < 20 {
        failures.push(format!(
          "{key}({text:?}): got {got}, expected {}",
          case[key]
        ));
      }
    };

    check("split", encode(tsm::split(text)));
    for name in names {
      let method = tsm::get_method(name).unwrap();
      check(name, encode(method(text)));
    }
    check(
      "split_big_text",
      json!(tsm::split_big_text(text, SplitBigTextOptions::default())),
    );
    let max_len = case["small_max_len"].as_u64().unwrap() as usize;
    check(
      "split_big_text_small",
      json!(tsm::split_big_text(text, SplitBigTextOptions { max_len })),
    );
  }
  eprintln!("{total} cases, matched per function: {matched:?}");
  assert!(failures.is_empty(), "{}", failures.join("\n"));
  assert!(total >= 1500);
}

#[test]
fn python_digit_predicates() {
  // `\d` / isdecimal: ASCII, fullwidth and Arabic-Indic digits only.
  for ch in ['0', '9', '３', '٣'] {
    assert!(tsm::py_isdecimal(ch) && tsm::py_isdigit(ch), "{ch}");
  }
  // isdigit but not isdecimal.
  for ch in ['²', '①'] {
    assert!(!tsm::py_isdecimal(ch) && tsm::py_isdigit(ch), "{ch}");
  }
  // Numeric but neither.
  for ch in ['Ⅳ', '½', '一', 'a', '.'] {
    assert!(!tsm::py_isdecimal(ch) && !tsm::py_isdigit(ch), "{ch}");
  }
}

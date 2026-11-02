//! Parity tests for `text/symbols.py`, `text/symbols2.py` and `text/__init__.py`.
//!
//! Fixtures: `parity/symbols/gen_fixtures.py`.

use gpt_sovits_text::{
  KeyError, cleaned_text_to_sequence, symbol_to_id_v1, symbol_to_id_v2, symbols, symbols2,
};
use serde_json::Value;

fn fixture() -> Value {
  serde_json::from_str(include_str!("fixtures/symbols/symbols.json")).unwrap()
}

fn strings(v: &Value) -> Vec<String> {
  v.as_array()
    .unwrap()
    .iter()
    .map(|s| s.as_str().unwrap().to_string())
    .collect()
}

fn owned(v: &[&str]) -> Vec<String> {
  v.iter().map(|s| s.to_string()).collect()
}

fn sorted(v: &[&str]) -> Vec<String> {
  let mut out = owned(v);
  out.sort();
  out.dedup();
  out
}

#[test]
fn symbols_v1_constants() {
  let f = fixture();
  let m = &f["symbols"];
  assert_eq!(owned(symbols::PUNCTUATION), strings(&m["punctuation"]));
  assert_eq!(owned(symbols::PU_SYMBOLS), strings(&m["pu_symbols"]));
  assert_eq!(symbols::PAD, m["pad"].as_str().unwrap());
  assert_eq!(owned(symbols::C), strings(&m["c"]));
  assert_eq!(owned(symbols::V), strings(&m["v"]));
  assert_eq!(
    owned(symbols::V_WITHOUT_TONE),
    strings(&m["v_without_tone"])
  );
  assert_eq!(owned(symbols::JA_SYMBOLS), strings(&m["ja_symbols"]));
  assert_eq!(sorted(symbols::ARPA), strings(&m["arpa_sorted"]));
  assert_eq!(owned(&symbols::SYMBOLS), strings(&m["symbols"]));
}

#[test]
fn symbols2_constants() {
  let f = fixture();
  let m = &f["symbols2"];
  assert_eq!(owned(symbols2::PUNCTUATION), strings(&m["punctuation"]));
  assert_eq!(owned(symbols2::PU_SYMBOLS), strings(&m["pu_symbols"]));
  assert_eq!(symbols2::PAD, m["pad"].as_str().unwrap());
  assert_eq!(owned(symbols2::C), strings(&m["c"]));
  assert_eq!(owned(symbols2::V), strings(&m["v"]));
  assert_eq!(
    owned(symbols2::V_WITHOUT_TONE),
    strings(&m["v_without_tone"])
  );
  assert_eq!(owned(symbols2::JA_SYMBOLS), strings(&m["ja_symbols"]));
  assert_eq!(sorted(symbols2::ARPA), strings(&m["arpa_sorted"]));
  assert_eq!(symbols2::KO_SYMBOLS, m["ko_symbols"].as_str().unwrap());
  assert_eq!(sorted(symbols2::YUE_SYMBOLS), strings(&m["yue_symbols"]));
  assert_eq!(owned(&symbols2::SYMBOLS), strings(&m["symbols"]));
}

fn check_map(map: &std::collections::HashMap<&'static str, usize>, items: &Value) {
  let items = items.as_array().unwrap();
  assert_eq!(map.len(), items.len());
  for item in items {
    let s = item[0].as_str().unwrap();
    let i = item[1].as_u64().unwrap() as usize;
    assert_eq!(map.get(s), Some(&i), "symbol {s:?}");
  }
}

#[test]
fn symbol_to_id_maps() {
  let f = fixture();
  check_map(symbol_to_id_v1(), &f["symbol_to_id_v1"]);
  check_map(symbol_to_id_v2(), &f["symbol_to_id_v2"]);
}

#[test]
fn cleaned_text_to_sequence_cases() {
  let f = fixture();
  // The generator ran with `version` unset; `None` cases depend on it.
  let env_version = std::env::var("version").ok();
  let mut checked = 0;
  for case in f["cleaned_text_to_sequence"].as_array().unwrap() {
    let version = case["version"].as_str();
    if version.is_none() && env_version.is_some() {
      continue;
    }
    let text = strings(&case["cleaned_text"]);
    let got = cleaned_text_to_sequence(&text, version);
    let expected = match case.get("ids") {
      Some(ids) => Ok(
        ids
          .as_array()
          .unwrap()
          .iter()
          .map(|i| i.as_u64().unwrap() as usize)
          .collect(),
      ),
      None => {
        assert_eq!(case["error"], "KeyError");
        Err(KeyError(case["key"].as_str().unwrap().to_string()))
      }
    };
    assert_eq!(got, expected, "case {case}");
    checked += 1;
  }
  assert!(checked >= 380, "checked {checked}");
}

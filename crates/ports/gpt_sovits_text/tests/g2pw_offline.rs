//! Parity tests of the `g2pw` port that do not need the G2PW model:
//! OpenCC s2tw, the tokenizer, `correct_pronunciation` and `pp_dict`.
//!
//! Fixtures: `parity/g2pw/gen_fixtures.py`.

use std::path::PathBuf;

use gpt_sovits_text::g2pw::g2pw::{PP_DICT, correct_pronunciation};
use gpt_sovits_text::g2pw::onnx_api::{TOKENIZER_JSON, tokenizer_from_json};
use gpt_sovits_text::g2pw::opencc::OpenCC;
use gpt_sovits_text::g2pw::utils::tokenize;
use serde_json::Value;

fn fixture(name: &str) -> PathBuf {
  PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    .join("tests/fixtures/g2pw")
    .join(name)
}

fn jsonl(name: &str) -> Vec<Value> {
  std::fs::read_to_string(fixture(name))
    .unwrap()
    .lines()
    .map(|l| serde_json::from_str(l).unwrap())
    .collect()
}

fn strings(v: &Value) -> Vec<String> {
  v.as_array()
    .unwrap()
    .iter()
    .map(|s| s.as_str().unwrap().to_string())
    .collect()
}

#[test]
fn opencc_s2tw() {
  let cc = OpenCC::s2tw();
  let records = jsonl("opencc.jsonl");
  let mut mismatches = Vec::new();
  for r in &records {
    let text = r["text"].as_str().unwrap();
    let got = cc.convert(text);
    if got != r["out"].as_str().unwrap() {
      mismatches.push((text.to_string(), got, r["out"].clone()));
    }
  }
  assert!(
    mismatches.is_empty(),
    "{} / {} mismatches, first: {:?}",
    mismatches.len(),
    records.len(),
    &mismatches[..mismatches.len().min(5)]
  );
  eprintln!("opencc: {} / {} exact", records.len(), records.len());
}

#[test]
fn tokenizer_tokenize() {
  let tokenizer = tokenizer_from_json(TOKENIZER_JSON).unwrap();
  let records = jsonl("tokenizer.jsonl");
  let mut mismatches = Vec::new();
  for r in &records {
    let word = r["word"].as_str().unwrap();
    let got = tokenize(&tokenizer, word).unwrap();
    if got != strings(&r["tokens"]) {
      mismatches.push((word.to_string(), got, r["tokens"].clone()));
    }
  }
  assert!(
    mismatches.is_empty(),
    "{} / {} mismatches, first: {:?}",
    mismatches.len(),
    records.len(),
    &mismatches[..mismatches.len().min(5)]
  );
  eprintln!("tokenizer: {} / {} exact", records.len(), records.len());
}

fn fnv1a64(s: &str) -> String {
  let mut h: u64 = 0xCBF29CE484222325;
  for b in s.bytes() {
    h ^= u64::from(b);
    h = h.wrapping_mul(0x100000001B3);
  }
  format!("{h:016x}")
}

#[test]
fn pp_dict_matches_read_dict() {
  let tables: Value =
    serde_json::from_str(&std::fs::read_to_string(fixture("tables.json")).unwrap()).unwrap();
  let mut items: Vec<(&String, &Vec<String>)> = PP_DICT.iter().collect();
  items.sort();
  let canonical = serde_json::to_string(&items).unwrap();
  assert_eq!(
    PP_DICT.len() as u64,
    tables["pp_dict_len"].as_u64().unwrap()
  );
  assert_eq!(
    fnv1a64(&canonical),
    tables["pp_dict_fnv1a64"].as_str().unwrap()
  );
}

#[test]
fn correct_pronunciation_cases() {
  let records = jsonl("correct_pronunciation.jsonl");
  for r in &records {
    let word = r["word"].as_str().unwrap();
    let got = correct_pronunciation(word, strings(&r["word_pinyins"]));
    match (&got, r.get("error")) {
      (Ok(out), None) => assert_eq!(*out, strings(&r["out"]), "{word}"),
      (Err(e), Some(name)) => assert_eq!(e.exception_name(), name.as_str(), "{word}"),
      _ => panic!("{word}: {got:?} vs {r}"),
    }
  }
  eprintln!(
    "correct_pronunciation: {} / {} exact",
    records.len(),
    records.len()
  );
}

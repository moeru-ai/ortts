//! Parity tests of `gpt_sovits_text::chinese2` that need no G2PW model:
//! the string functions, `normalizer`, `_merge_erhua`,
//! `_get_initials_finals`, `text_normalize` and `g2p` with `is_g2pw = False`
//! (the pypinyin branch). Fixtures:
//! `parity/chinese2/gen_fixtures.py`. The G2PW branch is in
//! `tests/chinese2_g2pw.rs`.

#[path = "common/chinese2_common.rs"]
mod chinese2_common;

use chinese2_common::{check_all, error_json, g2p_json, lines, strings};
use gpt_sovits_text::chinese2::{
  _get_initials_finals, _merge_erhua, Chinese2, PINYIN_TO_SYMBOL_MAP, normalizer, py_isalpha,
  replace_consecutive_punctuation, replace_punctuation, replace_punctuation_with_en,
  text_normalize,
};
use serde_json::{Value, json};

const SENTENCES: &str = include_str!("fixtures/chinese2/sentences.jsonl");
const PUNCT: &str = include_str!("fixtures/chinese2/punct.jsonl");
const UNITS: &str = include_str!("fixtures/chinese2/units.jsonl");

#[test]
fn pinyin_to_symbol_map_matches_file() {
  assert_eq!(PINYIN_TO_SYMBOL_MAP.len(), 429);
  assert_eq!(PINYIN_TO_SYMBOL_MAP["a"], "AA a");
  assert_eq!(PINYIN_TO_SYMBOL_MAP["zuo"], "z uo");
  assert_eq!(PINYIN_TO_SYMBOL_MAP["yi"], "y i");
}

#[test]
fn py_isalpha_matches_python() {
  for c in ['a', 'Z', 'é', 'ß', '中', '々', 'ー', 'あ', '한', 'İ', 'ʰ'] {
    assert!(py_isalpha(c), "{c:?}");
  }
  // Python's isalpha is the L* categories: not Nl (Ⅷ), Mn (U+0300),
  // digits, punctuation or whitespace.
  for c in [
    '1', '٣', 'Ⅷ', '\u{0300}', '，', ',', ' ', '\u{3000}', '〇', '_', '😀',
  ] {
    assert!(!py_isalpha(c), "{c:?}");
  }
}

#[test]
fn string_functions() {
  let records = lines(PUNCT);
  type StrFn = fn(&str) -> String;
  let fns: [(&str, StrFn); 3] = [
    ("rp", replace_punctuation),
    ("rpe", replace_punctuation_with_en),
    ("rcp", replace_consecutive_punctuation),
  ];
  for (key, f) in fns {
    check_all(&format!("punct.jsonl {key}"), &records, |r| {
      let t = r["t"].as_str().unwrap();
      (r[key].clone(), json!({"o": f(t)}))
    });
  }
}

#[test]
fn normalizer_an2cn() {
  let records = lines(PUNCT);
  check_all("punct.jsonl an2cn", &records, |r| {
    let t = r["t"].as_str().unwrap();
    let got = match normalizer(t) {
      Ok(s) => json!({"o": s}),
      Err(e) => json!({"e": ["ValueError", [e.to_string()]]}),
    };
    (r["an2cn"].clone(), got)
  });
}

#[test]
fn merge_erhua_and_get_initials_finals() {
  let records = lines(UNITS);
  let merge: Vec<Value> = records
    .iter()
    .filter(|r| r["f"] == "_merge_erhua")
    .cloned()
    .collect();
  let gif: Vec<Value> = records
    .iter()
    .filter(|r| r["f"] == "_get_initials_finals")
    .cloned()
    .collect();
  assert!(merge.len() > 1000 && gif.len() > 1000);
  check_all("units.jsonl _merge_erhua", &merge, |r| {
    let a = r["a"].as_array().unwrap();
    let got = match _merge_erhua(
      strings(&a[0]),
      strings(&a[1]),
      a[2].as_str().unwrap(),
      a[3].as_str().unwrap(),
    ) {
      Ok((i, f)) => json!({"o": [i, f]}),
      Err(e) => error_json(&e),
    };
    (r["r"].clone(), got)
  });
  check_all("units.jsonl _get_initials_finals", &gif, |r| {
    let (i, f) = _get_initials_finals(r["a"][0].as_str().unwrap());
    (r["r"].clone(), json!({"o": [i, f]}))
  });
}

#[test]
fn text_normalize_sentences() {
  let records = lines(SENTENCES);
  assert!(records.len() >= 5000);
  check_all("sentences.jsonl norm", &records, |r| {
    let got = match text_normalize(r["t"].as_str().unwrap()) {
      Ok(s) => json!({"o": s}),
      Err(e) => error_json(&e),
    };
    (r["norm"].clone(), got)
  });
}

#[test]
fn g2p_pypinyin_branch() {
  let records: Vec<Value> = lines(SENTENCES)
    .into_iter()
    .filter(|r| !r["py"].is_null())
    .collect();
  let c = Chinese2::without_g2pw();
  check_all("sentences.jsonl py (is_g2pw=False)", &records, |r| {
    let norm = r["norm"]["o"].as_str().unwrap();
    (r["py"].clone(), g2p_json(&c, norm))
  });
}

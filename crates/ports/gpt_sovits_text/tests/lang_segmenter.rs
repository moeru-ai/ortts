//! Parity tests for `text/LangSegmenter/langsegmenter.py`.
//!
//! Fixtures: `parity/lang_segmenter/gen_fixtures.py`.
//!
//! `getTexts` needs the fasttext model: set `FAST_LANGDETECT_MODEL` to the
//! path of `lid.176.bin` (the `lid-model` task of `pixi.toml` puts it in
//! `parity/.cache/fast_langdetect`). That test is skipped when it is unset.

use std::path::PathBuf;

use fast_langdetect::{LangDetectConfig, LangDetector, Model};
use gpt_sovits_text::lang_segmenter::{
  LangSegmenter, LangText, full_cjk, full_en, merge_lang, split_jako,
};
use serde_json::{Value, json};
use split_lang::SplitLangError;

/// `(default_lang, text)` pairs whose `getTexts` output is known to differ
/// (documented in `parity/README.md`). Empty: every case matches.
const KNOWN_MISMATCHES: &[(&str, &str)] = &[];

fn encode_texts(v: &[LangText]) -> Value {
  json!(
    v.iter()
      .map(|d| json!([d.lang, d.text]))
      .collect::<Vec<_>>()
  )
}

fn encode(r: Result<Vec<LangText>, SplitLangError>) -> Value {
  match r {
    Ok(v) => encode_texts(&v),
    Err(e) => json!({ "error": e.exception_name() }),
  }
}

#[test]
fn functions_match_python() {
  let mut n = 0;
  for line in include_str!("fixtures/lang_segmenter/functions.jsonl").lines() {
    let case: Value = serde_json::from_str(line).unwrap();
    let text = case["text"].as_str().unwrap();
    let item = LangText {
      lang: "x".into(),
      text: text.into(),
    };
    assert_eq!(json!(full_en(text)), case["full_en"], "full_en({text:?})");
    assert_eq!(
      json!(full_cjk(text)),
      case["full_cjk"],
      "full_cjk({text:?})"
    );
    assert_eq!(
      encode_texts(&split_jako("ja", &item)),
      case["split_jako_ja"],
      "split_jako('ja', {text:?})"
    );
    assert_eq!(
      encode_texts(&split_jako("ko", &item)),
      case["split_jako_ko"],
      "split_jako('ko', {text:?})"
    );
    n += 1;
  }
  eprintln!("functions: {n} texts matched");
  assert!(n > 1000);
}

#[test]
fn merge_lang_semantics() {
  let a = |l: &str, t: &str| LangText {
    lang: l.into(),
    text: t.into(),
  };
  let l = merge_lang(Vec::new(), a("zh", "你"));
  let l = merge_lang(l, a("zh", "好"));
  let l = merge_lang(l, a("en", "hi"));
  assert_eq!(l, vec![a("zh", "你好"), a("en", "hi")]);
}

#[test]
fn get_texts_matches_python() {
  let Some(path) = std::env::var_os("FAST_LANGDETECT_MODEL").map(PathBuf::from) else {
    eprintln!(
      "FAST_LANGDETECT_MODEL is not set; skipping (the lid-model task of \
       crates/ports/gpt_sovits_text/pixi.toml puts lid.176.bin in parity/.cache/fast_langdetect)"
    );
    return;
  };
  // Mirrors the module-level `_default_detector` replacement in
  // langsegmenter.py, with the model file given directly.
  fast_langdetect::set_default_detector(LangDetector::new(
    LangDetectConfig::new(None, Some(path), true, Some(80), Model::Auto).unwrap(),
  ));

  let mut stats = std::collections::BTreeMap::<String, (usize, usize)>::new();
  let mut unexpected = Vec::new();
  let mut seen_known = 0;
  for line in include_str!("fixtures/lang_segmenter/get_texts.jsonl").lines() {
    let case: Value = serde_json::from_str(line).unwrap();
    let text = case["text"].as_str().unwrap();
    for (default_lang, expected) in case["get_texts"].as_object().unwrap() {
      let got = encode(LangSegmenter::getTexts(text, default_lang));
      let entry = stats.entry(default_lang.clone()).or_default();
      entry.0 += 1;
      let known = KNOWN_MISMATCHES.contains(&(default_lang.as_str(), text));
      if &got == expected {
        entry.1 += 1;
        assert!(
          !known,
          "getTexts({text:?}, {default_lang:?}) is a known mismatch but matches"
        );
      } else if known {
        seen_known += 1;
      } else {
        unexpected.push(format!(
          "getTexts({text:?}, {default_lang:?}):\n  got      {got}\n  expected {expected}"
        ));
      }
    }
  }
  for (d, (total, matched)) in &stats {
    eprintln!("getTexts(default_lang={d:?}): {matched}/{total} matched");
  }
  assert!(
    unexpected.is_empty(),
    "{} mismatches:\n{}",
    unexpected.len(),
    unexpected[..unexpected.len().min(20)].join("\n")
  );
  assert_eq!(seen_known, KNOWN_MISMATCHES.len());
  assert_eq!(stats.len(), 4);
  assert!(stats[""].0 >= 3000);
}

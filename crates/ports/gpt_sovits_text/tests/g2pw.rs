//! Parity tests of the `g2pw` port that run the G2PW ONNX model.
//!
//! The model directory (`G2PWModel_1.1.zip`, extracted) is read from the
//! `G2PW_MODEL_DIR` environment variable (the `g2pw-model` task of
//! `pixi.toml` puts it in `parity/.cache/G2PWModel`); without it every test
//! here is skipped (and says so). Fixtures:
//! `parity/g2pw/gen_fixtures.py`.

use std::collections::{BTreeMap, BTreeSet};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;
use std::sync::{LazyLock, Mutex};

use gpt_sovits_text::g2pw::G2PWPinyin;
use gpt_sovits_text::g2pw::correct_pronunciation;
use gpt_sovits_text::g2pw::dataset::prepare_onnx_input;
use gpt_sovits_text::g2pw::g2pw::G2PWPinyinOptions;
use gpt_sovits_text::g2pw::onnx_api::{G2PWOnnxConverter, G2PWOnnxConverterOptions, G2PWStyle};
use gpt_sovits_text::g2pw::utils::PyValue;
use pypinyin::{Errors, PinyinMethods, Style};
use serde_json::{Value, json};

/// Argmax flips caused by numeric drift between onnxruntime 1.31 (Python)
/// and the ONNX Runtime of the `ort` crate: `(fixture, variant, sentence)`.
/// See `parity/g2pw/README.md`. Every entry must still mismatch, and every
/// mismatch must be listed.
const KNOWN_ARGMAX_FLIPS: &[(&str, &str, &str)] = &[];

fn model_dir() -> Option<PathBuf> {
  match std::env::var_os("G2PW_MODEL_DIR") {
    Some(dir) => Some(PathBuf::from(dir)),
    None => {
      eprintln!(
        "skipped: set G2PW_MODEL_DIR to the extracted G2PWModel_1.1 directory \
       (the g2pw-model task of crates/ports/gpt_sovits_text/pixi.toml puts it in parity/.cache/G2PWModel)"
      );
      None
    }
  }
}

/// `chinese2.py`'s `g2pw = G2PWPinyin(model_dir=..., model_source=bert_path,
/// v_to_u=False, neutral_tone_with_five=True)`.
static G2PW: LazyLock<Option<G2PWPinyin>> = LazyLock::new(|| {
  let dir = model_dir()?;
  Some(G2PWPinyin::new(dir, &G2PWPinyinOptions::chinese2()).unwrap())
});

/// A second converter (same arguments as `G2PW._g2pw`) whose
/// `enable_sentence_dedup` / `polyphonic_context_chars` the tests change.
static VARIANT: LazyLock<Option<Mutex<G2PWOnnxConverter>>> = LazyLock::new(|| {
  let dir = model_dir()?;
  Some(Mutex::new(
    G2PWOnnxConverter::new(
      dir,
      &G2PWOnnxConverterOptions {
        style: G2PWStyle::Pinyin,
        model_source: None,
        enable_non_tradional_chinese: true,
      },
    )
    .unwrap(),
  ))
});

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

/// Expands the `{"unique": [...], "index": [...]}` lists of `model.jsonl`.
fn undedup(v: &Value) -> Vec<Value> {
  let unique = v["unique"].as_array().unwrap();
  v["index"]
    .as_array()
    .unwrap()
    .iter()
    .map(|i| unique[i.as_u64().unwrap() as usize].clone())
    .collect()
}

fn opt_results(v: &[Vec<Option<String>>]) -> Value {
  json!(v)
}

fn py_value_json(v: &PyValue) -> Value {
  match v {
    PyValue::None => Value::Null,
    PyValue::Bool(b) => json!(b),
    PyValue::Int(i) => json!(i),
    PyValue::Float(f) => json!(f),
    PyValue::Str(s) => json!(s),
    PyValue::List(l) => Value::Array(l.iter().map(py_value_json).collect()),
    PyValue::Dict(d) => Value::Object(
      d.iter()
        .map(|(k, v)| (k.clone(), py_value_json(v)))
        .collect(),
    ),
  }
}

/// A difference between the port and the fixture that is explained by an
/// argmax flip: the same sentence, the same set of `None`s, and only
/// polyphonic positions differ.
struct Mismatch {
  fixture: &'static str,
  variant: String,
  sentence: String,
  detail: String,
}

fn check_flips(mismatches: &[Mismatch], fixture_name: &str) {
  let got: BTreeSet<(String, String)> = mismatches
    .iter()
    .map(|m| (m.variant.clone(), m.sentence.clone()))
    .collect();
  let expected: BTreeSet<(String, String)> = KNOWN_ARGMAX_FLIPS
    .iter()
    .filter(|(f, _, _)| *f == fixture_name)
    .map(|(_, v, s)| (v.to_string(), s.to_string()))
    .collect();
  if got != expected {
    for m in mismatches {
      eprintln!(
        "MISMATCH {} [{}] {:?}: {}",
        m.fixture, m.variant, m.sentence, m.detail
      );
    }
    panic!(
      "{fixture_name}: mismatches {:?} differ from KNOWN_ARGMAX_FLIPS {:?}",
      got, expected
    );
  }
}

#[test]
fn tables() {
  let Some(g2pw) = G2PW.as_ref() else { return };
  let conv = &g2pw._g2pw;
  let t: Value =
    serde_json::from_str(&std::fs::read_to_string(fixture("tables.json")).unwrap()).unwrap();

  let config: serde_json::Map<String, Value> = conv
    .config
    .attrs
    .iter()
    .map(|(k, v)| (k.clone(), py_value_json(v)))
    .collect();
  assert_eq!(Value::Object(config), t["config"]);
  assert_eq!(conv.model_source, t["model_source"].as_str().unwrap());
  assert_eq!(conv.labels, strings(&t["labels"]));
  assert_eq!(conv.chars, strings(&t["chars"]));
  let c2p: BTreeMap<String, Vec<usize>> = conv
    .char2phonemes
    .iter()
    .map(|(k, v)| (k.clone(), v.clone()))
    .collect();
  let c2p_expected: BTreeMap<String, Vec<usize>> =
    serde_json::from_value(t["char2phonemes"].clone()).unwrap();
  assert_eq!(c2p, c2p_expected);
  let mut new: Vec<String> = conv.polyphonic_chars_new.iter().cloned().collect();
  new.sort();
  assert_eq!(new, strings(&t["polyphonic_chars_new"]));
  assert_eq!(
    conv.monophonic_chars_dict.len() as u64,
    t["monophonic_chars_dict_len"].as_u64().unwrap()
  );
  for (k, v) in t["monophonic_chars_dict_sample"].as_object().unwrap() {
    assert_eq!(conv.monophonic_chars_dict[k], v.as_str().unwrap());
  }
  assert_eq!(
    conv.bopomofo_convert_dict.len() as u64,
    t["bopomofo_convert_dict_len"].as_u64().unwrap()
  );
  assert_eq!(
    conv.char_bopomofo_dict.len() as u64,
    t["char_bopomofo_dict_len"].as_u64().unwrap()
  );
  assert_eq!(
    conv.enable_sentence_dedup,
    t["enable_sentence_dedup"].as_bool().unwrap()
  );
  assert_eq!(
    conv.polyphonic_context_chars as u64,
    t["polyphonic_context_chars"].as_u64().unwrap()
  );
  assert_eq!(conv.enable_opencc, t["enable_opencc"].as_bool().unwrap());
  for (b, expected) in t["convert_bopomofo_to_pinyin"].as_object().unwrap() {
    match conv._convert_bopomofo_to_pinyin(b) {
      Ok(got) => assert_eq!(json!(got), *expected, "{b}"),
      Err(e) => assert_eq!(e.exception_name(), expected["error"].as_str(), "{b}"),
    }
  }
}

#[test]
fn call() {
  let Some(variant) = VARIANT.as_ref() else {
    return;
  };
  let mut conv = variant.lock().unwrap();
  let records = jsonl("call.jsonl");
  let mut counts: BTreeMap<String, (usize, usize)> = BTreeMap::new();
  let mut mismatches = Vec::new();
  for r in &records {
    let name = r["variant"].as_str().unwrap();
    (conv.enable_sentence_dedup, conv.polyphonic_context_chars) = match name {
      "default" => (true, 16),
      "nodedup" => (false, 16),
      "ctx0" => (true, 0),
      "ctx3" => (true, 3),
      "nodedup_ctx0" => (false, 0),
      _ => panic!("unknown variant {name}"),
    };
    let sentences = strings(&r["sentences"]);
    let got = conv.call(&sentences);
    let entry = counts.entry(name.to_string()).or_default();
    entry.0 += 1;
    match (&got, r.get("error")) {
      (Ok(out), None) if opt_results(out) == r["result"] => entry.1 += 1,
      (Ok(out), None) => {
        let expected = &r["result"];
        for (i, (g, e)) in out.iter().zip(expected.as_array().unwrap()).enumerate() {
          if json!(g) != *e {
            mismatches.push(Mismatch {
              fixture: "call",
              variant: name.to_string(),
              sentence: sentences[i].clone(),
              detail: format!("{g:?} vs {e}"),
            });
          }
        }
      }
      (Err(e), Some(err)) if e.exception_name() == err.as_str() => entry.1 += 1,
      _ => panic!("{name} {sentences:?}: {got:?} vs {r}"),
    }
  }
  eprintln!("call: {counts:?}");
  check_flips(&mismatches, "call");
  conv.enable_sentence_dedup = true;
  conv.polyphonic_context_chars = 16;
}

#[test]
fn model() {
  let Some(g2pw) = G2PW.as_ref() else { return };
  let conv = &g2pw._g2pw;
  let cc = conv.cc.unwrap();
  let records = jsonl("model.jsonl");
  let (mut rows, mut label_matches, mut max_conf_diff, mut max_prob_diff) =
    (0usize, 0usize, 0f64, 0f64);
  let mut mismatches = Vec::new();
  for r in &records {
    let sentence = r["sentence"].as_str().unwrap();
    let prepared = conv._prepare_data(&[cc.convert(sentence)]);
    let (texts, mq, rq, sids, partial) = match prepared {
      Ok(p) => p,
      Err(e) => {
        assert_eq!(e.exception_name(), r["error"].as_str(), "{sentence}");
        continue;
      }
    };
    assert_eq!(json!(texts), json!(undedup(&r["texts"])), "{sentence}");
    assert_eq!(json!(mq), r["model_query_ids"], "{sentence}");
    assert_eq!(json!(rq), r["result_query_ids"], "{sentence}");
    assert_eq!(json!(sids), r["sent_ids"], "{sentence}");
    assert_eq!(json!(partial), r["partial_results"], "{sentence}");
    if texts.is_empty() {
      continue;
    }
    let mi = prepare_onnx_input(
      &conv.tokenizer,
      &conv.labels,
      &conv.char2phonemes,
      &conv.chars,
      &texts,
      &mq,
      conv.config.use_mask(),
      None,
      512,
      Some(&conv.char2id),
      conv.char_phoneme_masks.as_ref(),
    );
    let mi = match mi {
      Ok(mi) => mi,
      Err(e) => {
        assert_eq!(e.exception_name(), r["error"].as_str(), "{sentence}");
        continue;
      }
    };
    let Some(mi) = mi else {
      assert!(r["onnx_input"].is_null(), "{sentence}");
      continue;
    };
    let oi = &r["onnx_input"];
    // The fixture stores token_type_ids (all 0) and attention_masks (1 for
    // the first attention_lengths[i] columns, 0 after) by their lengths, and
    // input_ids without the padding.
    assert_eq!(json!(mi.input_ids.shape()), oi["shape"], "{sentence}");
    assert!(mi.token_type_ids.iter().all(|&v| v == 0), "{sentence}");
    let lengths: Vec<usize> = oi["attention_lengths"]
      .as_array()
      .unwrap()
      .iter()
      .map(|v| v.as_u64().unwrap() as usize)
      .collect();
    for (row, (ids, mask)) in mi
      .input_ids
      .outer_iter()
      .zip(mi.attention_masks.outer_iter())
      .enumerate()
    {
      let n = lengths[row];
      let expected_mask: Vec<i64> = (0..mask.len()).map(|j| i64::from(j < n)).collect();
      assert_eq!(mask.to_vec(), expected_mask, "{sentence}");
      assert!(ids.iter().skip(n).all(|&v| v == 0), "{sentence}");
      assert_eq!(
        json!(ids.iter().take(n).collect::<Vec<_>>()),
        undedup(&oi["input_ids"])[row],
        "{sentence}"
      );
    }
    let nonzero: Vec<Vec<usize>> = mi
      .phoneme_masks
      .outer_iter()
      .map(|row| {
        row
          .iter()
          .enumerate()
          .filter(|(_, v)| **v != 0.0)
          .map(|(i, _)| i)
          .collect()
      })
      .collect();
    assert_eq!(json!(nonzero), oi["phoneme_mask_nonzero"], "{sentence}");
    assert_eq!(json!(mi.char_ids.to_vec()), oi["char_ids"], "{sentence}");
    assert_eq!(
      json!(mi.position_ids.to_vec()),
      oi["position_ids"],
      "{sentence}"
    );

    let probs = conv._predict_probs(&mi).unwrap();
    let (preds, confs) = conv._predict(&mi).unwrap();
    let expected_preds = strings(&r["preds"]);
    for (i, (p, c)) in preds.iter().zip(&confs).enumerate() {
      rows += 1;
      if *p == expected_preds[i] {
        label_matches += 1;
      } else {
        mismatches.push(Mismatch {
          fixture: "model",
          variant: "default".to_string(),
          sentence: sentence.to_string(),
          detail: format!("row {i}: {p} vs {}", expected_preds[i]),
        });
      }
      let ec = r["confidences"][i].as_f64().unwrap();
      max_conf_diff = max_conf_diff.max((c - ec).abs());
    }
    if let Some(expected_probs) = r.get("probs") {
      for (i, row) in expected_probs.as_array().unwrap().iter().enumerate() {
        for (j, p) in row.as_array().unwrap().iter().enumerate() {
          max_prob_diff = max_prob_diff.max((f64::from(probs[[i, j]]) - p.as_f64().unwrap()).abs());
        }
      }
    }
  }
  eprintln!(
    "model: {label_matches} / {rows} argmax labels match, max |confidence diff| {max_conf_diff:e}, max |prob diff| {max_prob_diff:e}"
  );
  assert!(max_conf_diff < 1e-4, "confidence diff {max_conf_diff}");
  assert!(max_prob_diff < 1e-4, "prob diff {max_prob_diff}");
  check_flips(&mismatches, "model");
}

#[test]
fn pinyin() {
  let Some(g2pw) = G2PW.as_ref() else { return };
  let records = jsonl("pinyin.jsonl");
  let mut mismatches = Vec::new();
  let mut cases = 0;
  for r in &records {
    let text = r["text"].as_str().unwrap();
    let errors = Errors::Default;
    let run = |key: &str| -> Value {
      match key {
        "pinyin_tone3" => json!(g2pw.pinyin(text, Style::TONE3, false, &errors, true)),
        "lazy_tone3" => json!(g2pw.lazy_pinyin(text, Style::TONE3, &errors, true)),
        "lazy_initials" => json!(g2pw.lazy_pinyin(text, Style::INITIALS, &errors, true)),
        "lazy_finals_tone3" => json!(g2pw.lazy_pinyin(text, Style::FINALS_TONE3, &errors, true)),
        "pinyin_tone_heteronym" => json!(g2pw.pinyin(text, Style::TONE, true, &errors, true)),
        "lazy_normal" => json!(g2pw.lazy_pinyin(text, Style::NORMAL, &errors, true)),
        _ => unreachable!(),
      }
    };
    if r.get("error").is_some() {
      let res = catch_unwind(AssertUnwindSafe(|| run("pinyin_tone3")));
      assert!(res.is_err(), "{text}: expected {}", r["error"]);
      cases += 1;
      continue;
    }
    for key in [
      "pinyin_tone3",
      "lazy_tone3",
      "lazy_initials",
      "lazy_finals_tone3",
      "pinyin_tone_heteronym",
      "lazy_normal",
    ] {
      let Some(expected) = r.get(key) else { continue };
      cases += 1;
      let got = run(key);
      if got != *expected {
        mismatches.push(Mismatch {
          fixture: "pinyin",
          variant: key.to_string(),
          sentence: text.to_string(),
          detail: format!("{got} vs {expected}"),
        });
      }
    }
  }
  eprintln!("pinyin: {} / {cases} exact", cases - mismatches.len());
  check_flips(&mismatches, "pinyin");
}

#[test]
fn chinese2_calls() {
  let Some(g2pw) = G2PW.as_ref() else { return };
  let records = jsonl("chinese2.jsonl");
  let (mut calls, mut cps) = (0, 0);
  let mut mismatches = Vec::new();
  for r in &records {
    for c in r["g2pw_calls"].as_array().unwrap() {
      calls += 1;
      let sentences = strings(&c["sentences"]);
      let got = g2pw._g2pw.call(&sentences).unwrap();
      for (i, (g, e)) in got.iter().zip(c["result"].as_array().unwrap()).enumerate() {
        if json!(g) != *e {
          mismatches.push(Mismatch {
            fixture: "chinese2",
            variant: "g2pw".to_string(),
            sentence: sentences[i].clone(),
            detail: format!("{g:?} vs {e}"),
          });
        }
      }
    }
    for c in r["correct_pronunciation"].as_array().unwrap() {
      cps += 1;
      let word = c["word"].as_str().unwrap();
      let got = correct_pronunciation(word, strings(&c["word_pinyins"]));
      match (&got, c.get("error")) {
        (Ok(out), None) => assert_eq!(*out, strings(&c["out"]), "{word}"),
        (Err(e), Some(name)) => assert_eq!(e.exception_name(), name.as_str(), "{word}"),
        _ => panic!("{word}: {got:?} vs {c}"),
      }
    }
  }
  eprintln!(
    "chinese2: {} texts, {calls} g2pw batches, {cps} correct_pronunciation calls",
    records.len()
  );
  check_flips(&mismatches, "chinese2");
}

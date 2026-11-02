//! Helpers shared by `tests/cleaner.rs`, `tests/text_preprocessor.rs` and
//! `tests/e2e.rs`: the fake BERT model of
//! `parity/e2e/e2e_common.py`, its feature summary, the
//! JSON encoding of outputs, and the model setup from environment variables.

#![allow(dead_code)]

use std::path::PathBuf;

use gpt_sovits_text::tts_infer_pack::text_preprocessor::{
  BertError, BertFeatureModel, BertInputs, Error, PreprocessResult,
};
use gpt_sovits_text::{Cleaner, TextPreprocessor};
use ndarray::Array2;
use serde_json::{Value, json};

/// `FakeBert` of `e2e_common.py`: hidden state `[t, d]` is
/// `((id_t * 1009 + t * 31 + d * 17) % 65521) / 64`, exact in `f32`.
pub struct FakeBert;

impl BertFeatureModel for FakeBert {
  fn hidden_states(&self, inputs: &BertInputs) -> Result<Array2<f32>, BertError> {
    let ids = &inputs.input_ids;
    Ok(Array2::from_shape_fn((ids.len(), 1024), |(t, d)| {
      let v = (ids[t] * 1009 + t as i64 * 31 + d as i64 * 17) % 65521;
      v as f32 / 64.0
    }))
  }
}

/// `bert_summary` of `e2e_common.py`: `[[rows, cols], checksum]`.
pub fn bert_summary(bert: &Array2<f32>) -> Value {
  let (rows, cols) = bert.dim();
  let mut checksum: i64 = 0;
  for ((i, j), &v) in bert.indexed_iter() {
    let k = (v * 64.0) as i64;
    assert_eq!(k as f32, v * 64.0, "fake BERT values are multiples of 1/64");
    checksum += k * (1 + ((i * cols + j) % 997) as i64);
  }
  json!([[rows, cols], checksum])
}

/// `{"error": "<exception class>"}`.
pub fn error_json(name: &str) -> Value {
  json!({ "error": name })
}

/// A `get_phones_and_bert` output in the fixture format.
pub fn gpb_json(r: Result<(Vec<usize>, Array2<f32>, String), Error>) -> Value {
  match r {
    Ok((phones, bert, norm)) => {
      json!({"phones": phones, "norm": norm, "bert": bert_summary(&bert)})
    }
    Err(e) => error_json(e.name()),
  }
}

/// A `preprocess` output in the fixture format.
pub fn preprocess_json(r: Result<Vec<PreprocessResult>, Error>) -> Value {
  match r {
    Ok(segs) => Value::Array(
      segs
        .iter()
        .map(|s| {
          json!({"phones": s.phones, "norm": s.norm_text, "bert": bert_summary(&s.bert_features)})
        })
        .collect(),
    ),
    Err(e) => error_json(e.name()),
  }
}

/// The per-method outputs of a record, with the `"cutN"` back-references
/// of the generator's `dedupe` resolved.
pub fn resolve_methods(out: &Value) -> Vec<(String, Value)> {
  let obj = out.as_object().unwrap();
  let mut res: Vec<(String, Value)> = Vec::new();
  for (m, v) in obj {
    let v = match v.as_str() {
      Some(r) => res.iter().find(|(k, _)| k == r).unwrap().1.clone(),
      None => v.clone(),
    };
    res.push((m.clone(), v));
  }
  res
}

pub fn lines(src: &str) -> Vec<Value> {
  src
    .lines()
    .filter(|l| !l.is_empty())
    .map(|l| serde_json::from_str(l).unwrap())
    .collect()
}

/// `G2PW_MODEL_DIR`, or `None` with a message.
pub fn g2pw_model_dir() -> Option<PathBuf> {
  let dir = std::env::var_os("G2PW_MODEL_DIR").map(PathBuf::from);
  if dir.is_none() {
    eprintln!(
      "skipped: set G2PW_MODEL_DIR to the extracted G2PWModel_1.1 directory \
       (the g2pw-model task of crates/ports/gpt_sovits_text/pixi.toml puts it in parity/.cache/G2PWModel)"
    );
  }
  dir
}

/// The full `TextPreprocessor` (G2PW and fast-langdetect from
/// `G2PW_MODEL_DIR` and `FAST_LANGDETECT_MODEL`), or `None` with a message.
pub fn full_text_preprocessor() -> Option<TextPreprocessor<FakeBert>> {
  let model = std::env::var_os("FAST_LANGDETECT_MODEL").map(PathBuf::from);
  let dir = g2pw_model_dir();
  let Some(model) = model else {
    eprintln!(
      "skipped: set FAST_LANGDETECT_MODEL to the path of lid.176.bin \
       (the lid-model task of crates/ports/gpt_sovits_text/pixi.toml puts it in parity/.cache/fast_langdetect)"
    );
    return None;
  };
  let dir = dir?;
  Some(TextPreprocessor::from_paths(FakeBert, dir, model).unwrap())
}

/// A `TextPreprocessor` without models (`chinese2` without G2PW); enough for
/// the functions that do not run a language frontend.
pub fn bare_text_preprocessor() -> TextPreprocessor<FakeBert> {
  TextPreprocessor::new(
    FakeBert,
    TextPreprocessor::<FakeBert>::default_tokenizer(),
    Cleaner::from_chinese2(gpt_sovits_text::chinese2::Chinese2::without_g2pw()),
  )
}

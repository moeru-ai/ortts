//! Parity tests of `gpt_sovits_text::chinese2::Chinese2::g2p` with the G2PW
//! model (`is_g2pw = True`, as GPT-SoVITS runs it).
//!
//! The model directory (`G2PWModel_1.1.zip`, extracted) is read from the
//! `G2PW_MODEL_DIR` environment variable (the `g2pw-model` task of
//! `pixi.toml` puts it in `parity/.cache/G2PWModel`); without it the tests
//! are skipped (and say so). Fixtures: `parity/chinese2/gen_fixtures.py`.

#[path = "common/chinese2_common.rs"]
mod chinese2_common;

use std::path::PathBuf;
use std::sync::LazyLock;

use chinese2_common::{check_all, g2p_json, lines};
use gpt_sovits_text::chinese2::Chinese2;
use serde_json::Value;

const SENTENCES: &str = include_str!("fixtures/chinese2/sentences.jsonl");

static CHINESE2: LazyLock<Option<Chinese2>> = LazyLock::new(|| {
  let Some(dir) = std::env::var_os("G2PW_MODEL_DIR").map(PathBuf::from) else {
    eprintln!(
      "skipped: set G2PW_MODEL_DIR to the extracted G2PWModel_1.1 directory \
       (the g2pw-model task of crates/ports/gpt_sovits_text/pixi.toml puts it in parity/.cache/G2PWModel)"
    );
    return None;
  };
  Some(Chinese2::new(dir).unwrap())
});

/// `g2p(text_normalize(text))`, what `cleaner.clean_text` computes for "zh".
#[test]
fn g2p_normalized() {
  let Some(c) = CHINESE2.as_ref() else { return };
  let records: Vec<Value> = lines(SENTENCES)
    .into_iter()
    .filter(|r| !r["g2p"].is_null())
    .collect();
  assert!(records.len() >= 5000);
  check_all("sentences.jsonl g2p", &records, |r| {
    let norm = r["norm"]["o"].as_str().unwrap();
    (r["g2p"].clone(), g2p_json(c, norm))
  });
}

/// `g2p(text)` on the unnormalized text (error branches).
#[test]
fn g2p_raw() {
  let Some(c) = CHINESE2.as_ref() else { return };
  let records: Vec<Value> = lines(SENTENCES)
    .into_iter()
    .filter(|r| r.get("raw").is_some())
    .collect();
  check_all("sentences.jsonl raw", &records, |r| {
    (r["raw"].clone(), g2p_json(c, r["t"].as_str().unwrap()))
  });
}

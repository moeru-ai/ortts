//! Parity tests against fixtures from `parity/gen_fixtures.py` (`pixi run fixtures`).

use std::collections::BTreeMap;

use g2p_en::G2p;
use serde::Deserialize;

#[derive(Deserialize)]
struct PredictCase {
  src: String,
  word: String,
  phones: Vec<String>,
}

#[derive(Deserialize)]
struct Vocab {
  graphemes: Vec<String>,
  phonemes: Vec<String>,
}

/// Fixture words whose Rust prediction is allowed to differ from Python.
/// Must stay in sync with the crate README. Empty: every case matches.
const KNOWN_MISMATCHES: &[&str] = &[];

#[test]
fn predict_matches_python() {
  let g2p = G2p::new();
  let cases: Vec<PredictCase> = include_str!("fixtures/predict.jsonl")
    .lines()
    .map(|l| serde_json::from_str(l).expect("bad fixture line"))
    .collect();
  assert!(
    cases.len() >= 6000,
    "too few fixture cases: {}",
    cases.len()
  );

  let mut per_src: BTreeMap<&str, (usize, usize)> = BTreeMap::new();
  let mut mismatches = Vec::new();
  let mut mismatched_words = Vec::new();
  for case in &cases {
    let got = g2p.predict(&case.word);
    let entry = per_src.entry(&case.src).or_default();
    entry.0 += 1;
    if got == case.phones {
      entry.1 += 1;
    } else {
      mismatches.push(format!(
        "{:?} ({}): python {:?}, rust {:?}",
        case.word, case.src, case.phones, got
      ));
      mismatched_words.push(case.word.as_str());
    }
  }
  for (src, (n, ok)) in &per_src {
    eprintln!("{src}: {ok}/{n}");
  }
  assert_eq!(
    mismatched_words,
    KNOWN_MISMATCHES,
    "mismatches:\n{}",
    mismatches.join("\n")
  );
}

#[test]
fn vocab_matches_python() {
  let g2p = G2p::new();
  let vocab: Vocab = serde_json::from_str(include_str!("fixtures/vocab.json")).unwrap();
  assert_eq!(g2p.graphemes, vocab.graphemes);
  assert_eq!(g2p.phonemes, vocab.phonemes);
  for (i, g) in vocab.graphemes.iter().enumerate() {
    assert_eq!(g2p.g2idx[g], i);
    assert_eq!(&g2p.idx2g[&i], g);
  }
  for (i, p) in vocab.phonemes.iter().enumerate() {
    assert_eq!(g2p.p2idx[p], i);
    assert_eq!(&g2p.idx2p[&i], p);
  }
  assert_eq!(g2p.g2idx.len(), vocab.graphemes.len());
  assert_eq!(g2p.p2idx.len(), vocab.phonemes.len());
}

#[test]
fn homograph2features_matches_python() {
  let g2p = G2p::new();
  let expected: Vec<(String, Vec<String>, Vec<String>, String)> =
    serde_json::from_str(include_str!("fixtures/homograph2features.json")).unwrap();
  assert_eq!(g2p.homograph2features.len(), expected.len());
  for (headword, pron1, pron2, pos1) in expected {
    let got = g2p
      .homograph2features
      .get(&headword)
      .unwrap_or_else(|| panic!("missing {headword}"));
    assert_eq!(got, &(pron1, pron2, pos1), "{headword}");
  }
}

#[test]
fn weights_have_numpy_shapes() {
  let g2p = G2p::new();
  let shapes = [
    (&g2p.enc_emb, 29, 256),
    (&g2p.enc_w_ih, 768, 256),
    (&g2p.enc_w_hh, 768, 256),
    (&g2p.dec_emb, 74, 256),
    (&g2p.dec_w_ih, 768, 256),
    (&g2p.dec_w_hh, 768, 256),
    (&g2p.fc_w, 74, 256),
  ];
  for (a, rows, cols) in shapes {
    assert_eq!(
      (a.rows(), a.cols(), a.data().len()),
      (rows, cols, rows * cols)
    );
  }
  for b in [&g2p.enc_b_ih, &g2p.enc_b_hh, &g2p.dec_b_ih, &g2p.dec_b_hh] {
    assert_eq!(b.len(), 768);
  }
  assert_eq!(g2p.fc_b.len(), 74);
}

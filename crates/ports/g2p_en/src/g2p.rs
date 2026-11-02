//! Port of `g2p_en/g2p.py`: the `G2p` class without `__call__` and the
//! nltk CMUdict / POS tagger setup (see `parity/README.md`).

use std::collections::HashMap;

use crate::npz;

/// The `homographs.en` file shipped with g2p_en 2.1.0.
const HOMOGRAPHS_EN: &str = include_str!("../data/homographs.en");

/// The `checkpoint20.npz` file shipped with g2p_en 2.1.0.
const CHECKPOINT20_NPZ: &[u8] = include_bytes!("../data/checkpoint20.npz");

/// The `(pron1, pron2, pos1)` tuple stored in `homograph2features`.
pub type HomographFeatures = (Vec<String>, Vec<String>, String);

/// A 2-D C-order `float32` array, the numpy arrays that
/// `G2p.load_variables` reads (`enc_emb`, `enc_w_ih`, ...).
#[derive(Debug, Clone, PartialEq)]
pub struct Array2 {
  rows: usize,
  cols: usize,
  data: Vec<f32>,
  /// `a.T` in C order, `(cols, rows)`, for [`matmul_t_add`].
  data_t: Vec<f32>,
}

impl Array2 {
  /// Wraps C-order `data` of shape `(rows, cols)`.
  pub fn new(rows: usize, cols: usize, data: Vec<f32>) -> Self {
    assert_eq!(
      data.len(),
      rows * cols,
      "data does not match shape ({rows}, {cols})"
    );
    let mut data_t = vec![0f32; data.len()];
    for r in 0..rows {
      for c in 0..cols {
        data_t[c * rows + r] = data[r * cols + c];
      }
    }
    Self {
      rows,
      cols,
      data,
      data_t,
    }
  }

  /// `a.shape[0]`.
  pub fn rows(&self) -> usize {
    self.rows
  }

  /// `a.shape[1]`.
  pub fn cols(&self) -> usize {
    self.cols
  }

  /// The C-order values.
  pub fn data(&self) -> &[f32] {
    &self.data
  }

  /// Row `i`, as `a[i]` in numpy.
  pub fn row(&self, i: usize) -> &[f32] {
    &self.data[i * self.cols..(i + 1) * self.cols]
  }
}

/// Python `str.isspace` for one character, used by `str.strip()`.
fn is_py_whitespace(c: char) -> bool {
  matches!(
    c,
    '\t'
      | '\n'
      | '\x0b'
      | '\x0c'
      | '\r'
      | '\x1c'
      | '\x1d'
      | '\x1e'
      | '\x1f'
      | ' '
      | '\u{85}'
      | '\u{a0}'
      | '\u{1680}'
  ) || ('\u{2000}'..='\u{200a}').contains(&c)
    || matches!(
      c,
      '\u{2028}' | '\u{2029}' | '\u{202f}' | '\u{205f}' | '\u{3000}'
    )
}

/// Python `str.splitlines()` (without `keepends`).
fn splitlines(s: &str) -> Vec<&str> {
  let mut lines = Vec::new();
  let mut start = 0;
  let mut chars = s.char_indices().peekable();
  while let Some((i, c)) = chars.next() {
    let is_break = matches!(
      c,
      '\n' | '\r' | '\x0b' | '\x0c' | '\x1c' | '\x1d' | '\x1e' | '\u{85}' | '\u{2028}' | '\u{2029}'
    );
    if !is_break {
      continue;
    }
    lines.push(&s[start..i]);
    start = i + c.len_utf8();
    if c == '\r' && matches!(chars.peek(), Some((_, '\n'))) {
      chars.next();
      start += 1;
    }
  }
  if start < s.len() {
    lines.push(&s[start..]);
  }
  lines
}

/// Port of `construct_homograph_dictionary()`: parses `homographs.en` into
/// `{headword.lower(): (pron1.split(), pron2.split(), pos1)}`. Later lines
/// overwrite earlier ones with the same headword, as with a Python dict.
pub fn construct_homograph_dictionary() -> HashMap<String, HomographFeatures> {
  let mut homograph2features = HashMap::new();
  for line in splitlines(HOMOGRAPHS_EN) {
    if line.starts_with('#') {
      continue; // comment
    }
    let fields: Vec<&str> = line.trim_matches(is_py_whitespace).split('|').collect();
    // Python unpacks into four names and raises ValueError otherwise.
    let [headword, pron1, pron2, pos1] = fields[..] else {
      panic!("ValueError: homographs.en line does not have 4 fields: {line:?}");
    };
    let split = |s: &str| {
      s.split(is_py_whitespace)
        .filter(|p| !p.is_empty())
        .map(str::to_owned)
        .collect()
    };
    homograph2features.insert(
      headword.to_lowercase(),
      (split(pron1), split(pron2), pos1.to_owned()),
    );
  }
  homograph2features
}

/// Port of `class G2p` from `g2p_en/g2p.py`.
///
/// Holds what `G2p.__init__` sets up except `self.cmu` (nltk CMUdict, which
/// GPT-SoVITS replaces with its own dictionary) and `self.variables` (the
/// npz handle; its arrays are the `enc_*`, `dec_*` and `fc_*` fields).
#[derive(Debug, Clone)]
pub struct G2p {
  /// Port of `self.graphemes`.
  pub graphemes: Vec<String>,
  /// Port of `self.phonemes`.
  pub phonemes: Vec<String>,
  /// Port of `self.g2idx`.
  pub g2idx: HashMap<String, usize>,
  /// Port of `self.idx2g`.
  pub idx2g: HashMap<usize, String>,
  /// Port of `self.p2idx`.
  pub p2idx: HashMap<String, usize>,
  /// Port of `self.idx2p`.
  pub idx2p: HashMap<usize, String>,
  /// Port of `self.homograph2features`. Public and mutable because
  /// GPT-SoVITS `en_G2p.__init__` overrides entries.
  pub homograph2features: HashMap<String, HomographFeatures>,
  /// Port of `self.enc_emb`, shape `(29, 256)`.
  pub enc_emb: Array2,
  /// Port of `self.enc_w_ih`, shape `(768, 256)`.
  pub enc_w_ih: Array2,
  /// Port of `self.enc_w_hh`, shape `(768, 256)`.
  pub enc_w_hh: Array2,
  /// Port of `self.enc_b_ih`, shape `(768,)`.
  pub enc_b_ih: Vec<f32>,
  /// Port of `self.enc_b_hh`, shape `(768,)`.
  pub enc_b_hh: Vec<f32>,
  /// Port of `self.dec_emb`, shape `(74, 256)`.
  pub dec_emb: Array2,
  /// Port of `self.dec_w_ih`, shape `(768, 256)`.
  pub dec_w_ih: Array2,
  /// Port of `self.dec_w_hh`, shape `(768, 256)`.
  pub dec_w_hh: Array2,
  /// Port of `self.dec_b_ih`, shape `(768,)`.
  pub dec_b_ih: Vec<f32>,
  /// Port of `self.dec_b_hh`, shape `(768,)`.
  pub dec_b_hh: Vec<f32>,
  /// Port of `self.fc_w`, shape `(74, 256)`.
  pub fc_w: Array2,
  /// Port of `self.fc_b`, shape `(74,)`.
  pub fc_b: Vec<f32>,
}

impl Default for G2p {
  fn default() -> Self {
    Self::new()
  }
}

/// Port of `np.matmul(x, w.T) + b` for a single row `x` of shape `(1, k)`
/// and `w` of shape `(n, k)`, in `float32`.
///
/// Every output is summed in index order, `out[j] = (((0 + x[0] * w[j, 0]) +
/// x[1] * w[j, 1]) + ...) + b[j]`, without fused multiply-add. numpy hands
/// the dot products to OpenBLAS, whose summation order depends on the CPU
/// kernel and is not reproduced bit for bit (see `parity/README.md`). The loop
/// runs over the transposed weights so that it vectorizes across `j`.
fn matmul_t_add(x: &[f32], w: &Array2, b: &[f32]) -> Vec<f32> {
  debug_assert_eq!(x.len(), w.cols);
  let n = w.rows;
  let mut out = vec![0f32; n];
  for (k, &xk) in x.iter().enumerate() {
    let wt_k = &w.data_t[k * n..(k + 1) * n];
    for (o, &wkj) in out.iter_mut().zip(wt_k) {
      *o += xk * wkj;
    }
  }
  for (o, &bj) in out.iter_mut().zip(b) {
    *o += bj;
  }
  out
}

/// Port of `ndarray.argmax()`: the index of the first maximum. A NaN counts
/// as the maximum, as in numpy.
fn argmax(xs: &[f32]) -> usize {
  let mut best = 0;
  for (i, &x) in xs.iter().enumerate() {
    if x.is_nan() {
      return i;
    }
    if x > xs[best] {
      best = i;
    }
  }
  best
}

impl G2p {
  /// Port of `G2p.__init__` (without `self.cmu = cmudict.dict()`).
  pub fn new() -> Self {
    let graphemes: Vec<String> = ["<pad>", "<unk>", "</s>"]
      .into_iter()
      .map(str::to_owned)
      .chain("abcdefghijklmnopqrstuvwxyz".chars().map(String::from))
      .collect();
    let phonemes: Vec<String> = [
      "<pad>", "<unk>", "<s>", "</s>", "AA0", "AA1", "AA2", "AE0", "AE1", "AE2", "AH0", "AH1",
      "AH2", "AO0", "AO1", "AO2", "AW0", "AW1", "AW2", "AY0", "AY1", "AY2", "B", "CH", "D", "DH",
      "EH0", "EH1", "EH2", "ER0", "ER1", "ER2", "EY0", "EY1", "EY2", "F", "G", "HH", "IH0", "IH1",
      "IH2", "IY0", "IY1", "IY2", "JH", "K", "L", "M", "N", "NG", "OW0", "OW1", "OW2", "OY0",
      "OY1", "OY2", "P", "R", "S", "SH", "T", "TH", "UH0", "UH1", "UH2", "UW", "UW0", "UW1", "UW2",
      "V", "W", "Y", "Z", "ZH",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect();
    let g2idx = graphemes
      .iter()
      .enumerate()
      .map(|(i, g)| (g.clone(), i))
      .collect();
    let idx2g = graphemes
      .iter()
      .enumerate()
      .map(|(i, g)| (i, g.clone()))
      .collect();
    let p2idx = phonemes
      .iter()
      .enumerate()
      .map(|(i, p)| (p.clone(), i))
      .collect();
    let idx2p = phonemes
      .iter()
      .enumerate()
      .map(|(i, p)| (i, p.clone()))
      .collect();

    let mut variables = npz::load(CHECKPOINT20_NPZ);
    let mut take2 = |name: &str| {
      let a = variables
        .remove(name)
        .unwrap_or_else(|| panic!("KeyError: {name} is not in checkpoint20.npz"));
      assert_eq!(a.shape.len(), 2, "{name} is not 2-D");
      Array2::new(a.shape[0], a.shape[1], a.data)
    };
    let enc_emb = take2("enc_emb");
    let enc_w_ih = take2("enc_w_ih");
    let enc_w_hh = take2("enc_w_hh");
    let dec_emb = take2("dec_emb");
    let dec_w_ih = take2("dec_w_ih");
    let dec_w_hh = take2("dec_w_hh");
    let fc_w = take2("fc_w");
    let mut take1 = |name: &str| {
      let a = variables
        .remove(name)
        .unwrap_or_else(|| panic!("KeyError: {name} is not in checkpoint20.npz"));
      assert_eq!(a.shape.len(), 1, "{name} is not 1-D");
      a.data
    };

    Self {
      graphemes,
      phonemes,
      g2idx,
      idx2g,
      p2idx,
      idx2p,
      homograph2features: construct_homograph_dictionary(),
      enc_emb,
      enc_w_ih,
      enc_w_hh,
      enc_b_ih: take1("enc_b_ih"),
      enc_b_hh: take1("enc_b_hh"),
      dec_emb,
      dec_w_ih,
      dec_w_hh,
      dec_b_ih: take1("dec_b_ih"),
      dec_b_hh: take1("dec_b_hh"),
      fc_w,
      fc_b: take1("fc_b"),
    }
  }

  /// Port of `G2p.sigmoid`: `1 / (1 + np.exp(-x))` in `float32`.
  pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
  }

  /// Port of `G2p.grucell` for batch size 1. `x` has the input size and `h`
  /// the hidden size; returns the new hidden state.
  pub fn grucell(
    x: &[f32],
    h: &[f32],
    w_ih: &Array2,
    w_hh: &Array2,
    b_ih: &[f32],
    b_hh: &[f32],
  ) -> Vec<f32> {
    let rzn_ih = matmul_t_add(x, w_ih, b_ih);
    let rzn_hh = matmul_t_add(h, w_hh, b_hh);

    let split = rzn_ih.len() * 2 / 3;
    let (rz_ih, n_ih) = rzn_ih.split_at(split);
    let (rz_hh, n_hh) = rzn_hh.split_at(rzn_hh.len() * 2 / 3);

    let rz: Vec<f32> = rz_ih
      .iter()
      .zip(rz_hh)
      .map(|(a, b)| Self::sigmoid(a + b))
      .collect();
    let (r, z) = rz.split_at(rz.len() / 2);

    (0..n_ih.len())
      .map(|i| {
        let n = (n_ih[i] + r[i] * n_hh[i]).tanh();
        (1.0 - z[i]) * n + z[i] * h[i]
      })
      .collect()
  }

  /// Port of `G2p.gru` for batch size 1. `x` is `(steps, input)`; returns
  /// the hidden state after every step, `(steps, hidden)`. `h0 = None`
  /// starts from zeros, as in Python.
  pub fn gru(
    x: &Array2,
    steps: usize,
    w_ih: &Array2,
    w_hh: &Array2,
    b_ih: &[f32],
    b_hh: &[f32],
    h0: Option<&[f32]>,
  ) -> Array2 {
    let hidden = w_hh.cols;
    let mut h = h0.map_or_else(|| vec![0f32; hidden], <[f32]>::to_vec);
    let mut outputs = vec![0f32; steps * hidden];
    for t in 0..steps {
      h = Self::grucell(x.row(t), &h, w_ih, w_hh, b_ih, b_hh);
      outputs[t * hidden..(t + 1) * hidden].copy_from_slice(&h);
    }
    Array2::new(steps, hidden, outputs)
  }

  /// Port of `G2p.encode`: embeds `list(word) + ["</s>"]`, mapping every
  /// character outside the graphemes to `<unk>`. Returns `(len + 1, emb)`
  /// (Python adds a leading batch axis of 1).
  pub fn encode(&self, word: &str) -> Array2 {
    let unk = self.g2idx["<unk>"];
    let mut x: Vec<usize> = word
      .chars()
      .map(|c| self.g2idx.get(&c.to_string()).copied().unwrap_or(unk))
      .collect();
    x.push(self.g2idx.get("</s>").copied().unwrap_or(unk));
    let cols = self.enc_emb.cols;
    let mut data = Vec::with_capacity(x.len() * cols);
    for idx in &x {
      data.extend_from_slice(self.enc_emb.row(*idx));
    }
    Array2::new(x.len(), cols, data)
  }

  /// Port of `G2p.predict`: GRU encoder over the characters of `word`, then
  /// greedy GRU decoding of at most 20 phonemes, stopping at `</s>`.
  pub fn predict(&self, word: &str) -> Vec<String> {
    // encoder
    let enc = self.encode(word);
    let h0 = vec![0f32; self.enc_w_hh.cols];
    let enc = Self::gru(
      &enc,
      word.chars().count() + 1,
      &self.enc_w_ih,
      &self.enc_w_hh,
      &self.enc_b_ih,
      &self.enc_b_hh,
      Some(&h0),
    );
    let last_hidden = enc.row(enc.rows - 1);

    // decoder
    let mut dec = self.dec_emb.row(2); // 2: <s>
    let mut h = last_hidden.to_vec();

    let mut preds = Vec::new();
    for _ in 0..20 {
      h = Self::grucell(
        dec,
        &h,
        &self.dec_w_ih,
        &self.dec_w_hh,
        &self.dec_b_ih,
        &self.dec_b_hh,
      );
      let logits = matmul_t_add(&h, &self.fc_w, &self.fc_b);
      let pred = argmax(&logits);
      if pred == 3 {
        break; // 3: </s>
      }
      preds.push(pred);
      dec = self.dec_emb.row(pred);
    }

    preds
      .into_iter()
      .map(|idx| {
        self
          .idx2p
          .get(&idx)
          .cloned()
          .unwrap_or_else(|| "<unk>".to_owned())
      })
      .collect()
  }
}

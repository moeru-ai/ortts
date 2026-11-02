//! Port of `GPT_SoVITS/text/g2pw/dataset.py` (GPT-SoVITS @ 04a44eb).
//!
//! Credits: this code is modified from <https://github.com/GitYCC/g2pW>
//! (PaddleSpeech, Apache-2.0).

use std::collections::{BTreeSet, HashMap};

use ndarray::{Array1, Array2};
use tokenizers::Tokenizer;

use super::G2PWError;
use super::utils::tokenize_and_map;

/// Port of `ANCHOR_CHAR`.
pub const ANCHOR_CHAR: &str = "▁";

/// The dict that `prepare_onnx_input` returns, with the same names.
#[derive(Clone, Debug, PartialEq)]
pub struct OnnxInput {
  /// `input_ids` (int64, `[batch, max_token_length]`).
  pub input_ids: Array2<i64>,
  /// `token_type_ids` (int64, `[batch, max_token_length]`).
  pub token_type_ids: Array2<i64>,
  /// `attention_masks` (int64, `[batch, max_token_length]`).
  pub attention_masks: Array2<i64>,
  /// `phoneme_masks` (float32, `[batch, len(labels)]`).
  pub phoneme_masks: Array2<f32>,
  /// `char_ids` (int64, `[batch]`).
  pub char_ids: Array1<i64>,
  /// `position_ids` (int64, `[batch]`).
  pub position_ids: Array1<i64>,
}

impl OnnxInput {
  /// `{name: value[indices] for name, value in model_input.items()}`.
  pub fn select(&self, indices: &[usize]) -> OnnxInput {
    use ndarray::Axis;
    OnnxInput {
      input_ids: self.input_ids.select(Axis(0), indices),
      token_type_ids: self.token_type_ids.select(Axis(0), indices),
      attention_masks: self.attention_masks.select(Axis(0), indices),
      phoneme_masks: self.phoneme_masks.select(Axis(0), indices),
      char_ids: self.char_ids.select(Axis(0), indices),
      position_ids: self.position_ids.select(Axis(0), indices),
    }
  }

  /// The batch size (`len(char_ids)`).
  pub fn len(&self) -> usize {
    self.char_ids.len()
  }

  /// `len(char_ids) == 0`.
  pub fn is_empty(&self) -> bool {
    self.char_ids.is_empty()
  }
}

/// A tokenized text in the `tokenized_cache` of `prepare_onnx_input`.
struct Cached {
  is_short: bool,
  tokens: Vec<String>,
  text2token: Vec<Option<usize>>,
  token2text: Vec<(usize, usize)>,
  input_id: Vec<i64>,
}

/// `tokenizer.convert_tokens_to_ids(tokens)` of the fast tokenizer: the
/// vocabulary id, or `unk_token_id` for a token that is not in it.
pub fn convert_tokens_to_ids(tokenizer: &Tokenizer, tokens: &[String]) -> Vec<i64> {
  let unk = tokenizer.token_to_id("[UNK]");
  tokens
    .iter()
    .map(|t| tokenizer.token_to_id(t).or(unk).map_or(-1, i64::from))
    .collect()
}

/// Port of `prepare_onnx_input(tokenizer, labels, char2phonemes, chars,
/// texts, query_ids, use_mask=False, window_size=None, max_len=512,
/// char2id=None, char_phoneme_masks=None)`.
///
/// Returns `Ok(None)` where Python returns `{}` (a text that
/// `tokenize_and_map` cannot tokenize; Python prints a warning). A query
/// character missing from `char2id` / `char_phoneme_masks` gives `KeyError`,
/// and an empty `texts` gives `ValueError` (`max()` of an empty sequence).
#[allow(clippy::too_many_arguments)]
pub fn prepare_onnx_input(
  tokenizer: &Tokenizer,
  labels: &[String],
  char2phonemes: &HashMap<String, Vec<usize>>,
  chars: &[String],
  texts: &[String],
  query_ids: &[usize],
  use_mask: bool,
  window_size: Option<usize>,
  max_len: usize,
  char2id: Option<&HashMap<String, usize>>,
  char_phoneme_masks: Option<&HashMap<String, Vec<u8>>>,
) -> Result<Option<OnnxInput>, G2PWError> {
  let truncated = window_size
    .filter(|&w| w != 0)
    .map(|w| _truncate_texts(w, texts, query_ids));

  let mut input_ids: Vec<Vec<i64>> = Vec::new();
  let mut phoneme_masks: Vec<Vec<u8>> = Vec::new();
  let mut char_ids: Vec<i64> = Vec::new();
  let mut position_ids: Vec<i64> = Vec::new();
  let mut tokenized_cache: HashMap<Vec<char>, Cached> = HashMap::new();

  let owned_char2id;
  let char2id = match char2id {
    Some(c) => c,
    None => {
      owned_char2id = chars
        .iter()
        .enumerate()
        .map(|(idx, c)| (c.clone(), idx))
        .collect::<HashMap<_, _>>();
      &owned_char2id
    }
  };
  let owned_masks;
  let char_phoneme_masks = if use_mask {
    match char_phoneme_masks {
      Some(m) => Some(m),
      None => {
        owned_masks = build_char_phoneme_masks(char2phonemes, labels.len());
        Some(&owned_masks)
      }
    }
  } else {
    None
  };
  let full_phoneme_mask = vec![1u8; labels.len()];

  for idx in 0..texts.len() {
    let (text, query_id) = match &truncated {
      Some((t, q)) => (&t[idx], q[idx]),
      None => (&texts[idx], query_ids[idx]),
    };
    // .lower()
    let text: Vec<char> = text.to_lowercase().chars().collect();

    if !tokenized_cache.contains_key(&text) {
      let (tokens, text2token, token2text) = match tokenize_and_map(tokenizer, &text) {
        Ok(v) => v,
        Err(_) => {
          // print(f'warning: text "{text}" is invalid')
          return Ok(None);
        }
      };
      let cached = if tokens.len() + 2 <= max_len {
        let processed_tokens = processed(&tokens);
        Cached {
          is_short: true,
          input_id: convert_tokens_to_ids(tokenizer, &processed_tokens),
          tokens,
          text2token,
          token2text,
        }
      } else {
        Cached {
          is_short: false,
          input_id: Vec::new(),
          tokens,
          text2token,
          token2text,
        }
      };
      tokenized_cache.insert(text.clone(), cached);
    }
    let cached = &tokenized_cache[&text];

    let (text_for_query, query_id_for_query, text2token_for_query, input_id) = if cached.is_short {
      (
        text.clone(),
        query_id,
        cached
          .text2token
          .iter()
          .map(|i| i.map(|i| i as i64))
          .collect::<Vec<_>>(),
        cached.input_id.clone(),
      )
    } else {
      let (text_for_query, query_id_for_query, tokens_for_query, text2token_for_query, _token2text) =
        _truncate(
          max_len,
          &text,
          query_id,
          &cached.tokens,
          &cached.text2token,
          &cached.token2text,
        )?;
      let processed_tokens = processed(&tokens_for_query);
      (
        text_for_query,
        query_id_for_query,
        text2token_for_query,
        convert_tokens_to_ids(tokenizer, &processed_tokens),
      )
    };

    let query_char: String = text_for_query
      .get(query_id_for_query)
      .ok_or_else(|| G2PWError::IndexError("string index out of range".to_string()))?
      .to_string();
    let phoneme_mask = match char_phoneme_masks {
      Some(masks) => masks
        .get(&query_char)
        .ok_or_else(|| G2PWError::KeyError(query_char.clone()))?
        .clone(),
      None => full_phoneme_mask.clone(),
    };
    let char_id = *char2id
      .get(&query_char)
      .ok_or_else(|| G2PWError::KeyError(query_char.clone()))?;
    // [CLS] token locate at first place
    let position_id = text2token_for_query
      .get(query_id_for_query)
      .ok_or_else(|| G2PWError::IndexError("list index out of range".to_string()))?
      .ok_or_else(|| {
        G2PWError::TypeError("unsupported operand type(s) for +: 'NoneType' and 'int'".to_string())
      })?
      + 1;

    input_ids.push(input_id);
    phoneme_masks.push(phoneme_mask);
    char_ids.push(char_id as i64);
    position_ids.push(position_id);
  }

  let max_token_length = input_ids
    .iter()
    .map(Vec::len)
    .max()
    .ok_or_else(|| G2PWError::ValueError("max() arg is an empty sequence".to_string()))?;

  let batch = input_ids.len();
  let mut ids = Array2::<i64>::zeros((batch, max_token_length));
  let token_types = Array2::<i64>::zeros((batch, max_token_length));
  let mut attention = Array2::<i64>::zeros((batch, max_token_length));
  for (row, seq) in input_ids.iter().enumerate() {
    for (col, &id) in seq.iter().enumerate() {
      ids[[row, col]] = id;
      attention[[row, col]] = 1;
    }
  }
  let mut masks = Array2::<f32>::zeros((batch, labels.len()));
  for (row, mask) in phoneme_masks.iter().enumerate() {
    for (col, &m) in mask.iter().enumerate() {
      masks[[row, col]] = f32::from(m);
    }
  }

  Ok(Some(OnnxInput {
    input_ids: ids,
    token_type_ids: token_types,
    attention_masks: attention,
    phoneme_masks: masks,
    char_ids: Array1::from(char_ids),
    position_ids: Array1::from(position_ids),
  }))
}

/// `["[CLS]"] + tokens + ["[SEP]"]`.
fn processed(tokens: &[String]) -> Vec<String> {
  let mut out = Vec::with_capacity(tokens.len() + 2);
  out.push("[CLS]".to_string());
  out.extend(tokens.iter().cloned());
  out.push("[SEP]".to_string());
  out
}

/// `{char: [1 if i in char2phonemes[char] else 0 for i in range(len(labels))]
/// for char in char2phonemes}`.
pub fn build_char_phoneme_masks(
  char2phonemes: &HashMap<String, Vec<usize>>,
  num_labels: usize,
) -> HashMap<String, Vec<u8>> {
  char2phonemes
    .iter()
    .map(|(char, phonemes)| {
      let mut mask = vec![0u8; num_labels];
      for &i in phonemes {
        if i < num_labels {
          mask[i] = 1;
        }
      }
      (char.clone(), mask)
    })
    .collect()
}

/// Port of `_truncate_texts(window_size, texts, query_ids)`.
pub fn _truncate_texts(
  window_size: usize,
  texts: &[String],
  query_ids: &[usize],
) -> (Vec<String>, Vec<usize>) {
  let mut truncated_texts = Vec::new();
  let mut truncated_query_ids = Vec::new();
  for (text, &query_id) in texts.iter().zip(query_ids.iter()) {
    let chars: Vec<char> = text.chars().collect();
    let start = query_id.saturating_sub(window_size / 2);
    let end = chars.len().min(query_id + window_size / 2);
    let truncated_text: String = if start < end {
      chars[start..end].iter().collect()
    } else {
      String::new()
    };
    truncated_texts.push(truncated_text);
    truncated_query_ids.push(query_id - start);
  }
  (truncated_texts, truncated_query_ids)
}

/// Port of `_truncate(max_len, text, query_id, tokens, text2token,
/// token2text)`.
///
/// Returns `(text, query_id, tokens, text2token, token2text)` for the
/// window of `max_len - 2` tokens around the query token. The text-to-token
/// map is signed: a position that `tokenize_and_map` left at its word index
/// can be shifted below zero, as in Python.
#[allow(clippy::type_complexity)]
pub fn _truncate(
  max_len: usize,
  text: &[char],
  query_id: usize,
  tokens: &[String],
  text2token: &[Option<usize>],
  token2text: &[(usize, usize)],
) -> Result<
  (
    Vec<char>,
    usize,
    Vec<String>,
    Vec<Option<i64>>,
    Vec<(usize, usize)>,
  ),
  G2PWError,
> {
  let truncate_len = (max_len as i64) - 2;
  if (tokens.len() as i64) <= truncate_len {
    return Ok((
      text.to_vec(),
      query_id,
      tokens.to_vec(),
      text2token.iter().map(|i| i.map(|i| i as i64)).collect(),
      token2text.to_vec(),
    ));
  }

  let token_position = text2token.get(query_id).copied().flatten().ok_or_else(|| {
    G2PWError::TypeError("unsupported operand type(s) for -: 'NoneType' and 'int'".to_string())
  })? as i64;

  let mut token_start = token_position - truncate_len / 2;
  let mut token_end = token_start + truncate_len;
  let font_exceed_dist = -token_start;
  let back_exceed_dist = token_end - tokens.len() as i64;
  if font_exceed_dist > 0 {
    token_start += font_exceed_dist;
    token_end += font_exceed_dist;
  } else if back_exceed_dist > 0 {
    token_start -= back_exceed_dist;
    token_end -= back_exceed_dist;
  }
  let token_start = token_start as usize;
  let token_end = token_end as usize;

  let start = token2text[token_start].0;
  let end = token2text[token_end - 1].1;

  let slice_end = end.min(text.len());
  let slice_start = start.min(slice_end);
  Ok((
    text[slice_start..slice_end].to_vec(),
    query_id - start,
    tokens[token_start..token_end].to_vec(),
    text2token[start.min(text2token.len())..end.min(text2token.len())]
      .iter()
      .map(|i| i.map(|i| i as i64 - token_start as i64))
      .collect(),
    token2text[token_start..token_end]
      .iter()
      .map(|&(s, e)| (s - start, e - start))
      .collect(),
  ))
}

/// Port of `get_phoneme_labels(polyphonic_chars)`.
pub fn get_phoneme_labels(
  polyphonic_chars: &[(String, String)],
) -> (Vec<String>, HashMap<String, Vec<usize>>) {
  let labels: Vec<String> = polyphonic_chars
    .iter()
    .map(|(_, phoneme)| phoneme.clone())
    .collect::<BTreeSet<_>>()
    .into_iter()
    .collect();
  let index: HashMap<&str, usize> = labels
    .iter()
    .enumerate()
    .map(|(i, l)| (l.as_str(), i))
    .collect();
  let mut char2phonemes: HashMap<String, Vec<usize>> = HashMap::new();
  for (char, phoneme) in polyphonic_chars {
    char2phonemes
      .entry(char.clone())
      .or_default()
      .push(index[phoneme.as_str()]);
  }
  (labels, char2phonemes)
}

/// Port of `get_char_phoneme_labels(polyphonic_chars)`.
pub fn get_char_phoneme_labels(
  polyphonic_chars: &[(String, String)],
) -> (Vec<String>, HashMap<String, Vec<usize>>) {
  let labels: Vec<String> = polyphonic_chars
    .iter()
    .map(|(char, phoneme)| format!("{char} {phoneme}"))
    .collect::<BTreeSet<_>>()
    .into_iter()
    .collect();
  let index: HashMap<&str, usize> = labels
    .iter()
    .enumerate()
    .map(|(i, l)| (l.as_str(), i))
    .collect();
  let mut char2phonemes: HashMap<String, Vec<usize>> = HashMap::new();
  for (char, phoneme) in polyphonic_chars {
    char2phonemes
      .entry(char.clone())
      .or_default()
      .push(index[format!("{char} {phoneme}").as_str()]);
  }
  (labels, char2phonemes)
}

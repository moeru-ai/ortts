//! Port of `GPT_SoVITS/text/g2pw/onnx_api.py` (GPT-SoVITS @ 04a44eb).
//!
//! Credits: this code is modified from PaddleSpeech
//! (`paddlespeech/t2s/frontend/g2pw`, Apache-2.0) and
//! <https://github.com/GitYCC/g2pW>.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use ndarray::{Array2, Axis};
use ort::ep::{CPU, CUDA, ExecutionProvider};
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Tensor;
use tokenizers::Tokenizer;

use super::G2PWError;
use super::dataset::{
  OnnxInput, build_char_phoneme_masks, get_char_phoneme_labels, get_phoneme_labels,
  prepare_onnx_input,
};
use super::opencc::OpenCC;
use super::utils::{Config, load_config};
use crate::zh_normalization::char_convert::tranditional_to_simplified;
use pypinyin::{PinyinOptions, Style};

/// The `chinese-roberta-wwm-ext-large` tokenizer (`tokenizer.json` of the
/// `bert_path` that GPT-SoVITS passes as `model_source`), embedded.
pub const TOKENIZER_JSON: &str = include_str!("../../data/g2pw/tokenizer.json");

/// `AutoTokenizer.from_pretrained(model_source)` for a fast tokenizer: the
/// `tokenizer.json`, with truncation and padding off (transformers calls
/// `no_truncation()` / `no_padding()` for `tokenize`).
pub fn tokenizer_from_json(json: &str) -> Result<Tokenizer, G2PWError> {
  let mut tokenizer =
    Tokenizer::from_bytes(json.as_bytes()).map_err(|e| G2PWError::Tokenizer(e.to_string()))?;
  tokenizer
    .with_truncation(None)
    .map_err(|e| G2PWError::Tokenizer(e.to_string()))?;
  tokenizer.with_padding(None);
  Ok(tokenizer)
}

/// Port of `model_version`.
pub const MODEL_VERSION: &str = "1.1";

/// The `session.run(...)[0]` part of [`predict`]: the probabilities,
/// `[batch, len(labels)]`.
pub fn run_session(
  session: &mut Session,
  onnx_input: &OnnxInput,
) -> Result<Array2<f32>, G2PWError> {
  let outputs = session.run(ort::inputs![
    "input_ids" => Tensor::from_array(onnx_input.input_ids.clone())?,
    "token_type_ids" => Tensor::from_array(onnx_input.token_type_ids.clone())?,
    "attention_mask" => Tensor::from_array(onnx_input.attention_masks.clone())?,
    "phoneme_mask" => Tensor::from_array(onnx_input.phoneme_masks.clone())?,
    "char_ids" => Tensor::from_array(onnx_input.char_ids.clone())?,
    "position_ids" => Tensor::from_array(onnx_input.position_ids.clone())?,
  ])?;
  let probs = outputs[0].try_extract_array::<f32>()?;
  let probs = probs
    .into_dimensionality::<ndarray::Ix2>()
    .map_err(|e| G2PWError::ValueError(e.to_string()))?
    .to_owned();
  Ok(probs)
}

/// `np.argmax(arr)`: the first index of the maximum (NaN counts as the
/// maximum, as in NumPy).
pub fn argmax(row: &[f32]) -> usize {
  let mut best = 0;
  for (i, &v) in row.iter().enumerate() {
    if row[best].is_nan() {
      break;
    }
    if v.is_nan() || v > row[best] {
      best = i;
    }
  }
  best
}

/// Port of `predict(session, onnx_input, labels)`: returns the label and
/// probability of the argmax of each row.
pub fn predict(
  session: &mut Session,
  onnx_input: &OnnxInput,
  labels: &[String],
) -> Result<(Vec<String>, Vec<f64>), G2PWError> {
  let probs = run_session(session, onnx_input)?;
  Ok(argmax_labels(&probs, labels))
}

/// The part of [`predict`] after `session.run`.
pub fn argmax_labels(probs: &Array2<f32>, labels: &[String]) -> (Vec<String>, Vec<f64>) {
  let mut all_preds = Vec::new();
  let mut all_confidences = Vec::new();
  for row in probs.axis_iter(Axis(0)) {
    let row = row.to_vec();
    let index = argmax(&row);
    all_preds.push(labels[index].clone());
    all_confidences.push(f64::from(row[index]));
  }
  (all_preds, all_confidences)
}

/// Port of `_load_json_from_candidates(filename, candidate_dirs)`. `None`
/// entries are skipped (the `if not candidate_dir` check).
pub fn _load_json_from_candidates(
  filename: &str,
  candidate_dirs: &[Option<&Path>],
) -> Result<serde_json::Value, G2PWError> {
  for candidate_dir in candidate_dirs.iter().flatten() {
    if candidate_dir.as_os_str().is_empty() {
      continue;
    }
    let json_path = candidate_dir.join(filename);
    if json_path.exists() {
      let text = std::fs::read_to_string(&json_path)
        .map_err(|e| G2PWError::FileNotFoundError(format!("{}: {e}", json_path.display())))?;
      return serde_json::from_str(&text).map_err(|e| G2PWError::Json(e.to_string()));
    }
  }
  Err(G2PWError::FileNotFoundError(format!(
    "Cannot locate {filename} in candidate dirs: {candidate_dirs:?}"
  )))
}

/// Port of `_find_first_existing_file(*paths)`.
pub fn _find_first_existing_file(paths: &[PathBuf]) -> Result<PathBuf, G2PWError> {
  for path in paths {
    if !path.as_os_str().is_empty() && path.exists() {
      return Ok(path.clone());
    }
  }
  Err(G2PWError::FileNotFoundError(format!(
    "Files not found: {paths:?}"
  )))
}

/// `download_and_decompress(model_dir)` without the download: Python
/// downloads and extracts `G2PWModel_1.1.zip` when `model_dir` does not
/// exist; the port expects the extracted model and gives
/// `FileNotFoundError`.
pub fn download_and_decompress(model_dir: &Path) -> Result<PathBuf, G2PWError> {
  if !model_dir.exists() {
    return Err(G2PWError::FileNotFoundError(format!(
      "G2PW model dir {} does not exist (download G2PWModel_1.1.zip)",
      model_dir.display()
    )));
  }
  Ok(model_dir.to_path_buf())
}

/// `str.strip()` (Python whitespace: Unicode `White_Space` plus
/// U+001C..U+001F).
pub(crate) fn py_strip(s: &str) -> &str {
  s.trim_matches(|c: char| c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c))
}

/// `int(s)` for a decimal string (surrounding whitespace, a sign, and
/// underscores between digits). Gives `ValueError` otherwise.
fn py_int(s: &str) -> Result<i64, G2PWError> {
  let err = || G2PWError::ValueError(format!("invalid literal for int() with base 10: {s:?}"));
  let t = py_strip(s);
  let (neg, digits) = match t.strip_prefix('-') {
    Some(rest) => (true, rest),
    None => (false, t.strip_prefix('+').unwrap_or(t)),
  };
  if digits.is_empty() || digits.starts_with('_') || digits.ends_with('_') || digits.contains("__")
  {
    return Err(err());
  }
  let mut value: i64 = 0;
  for c in digits.chars() {
    if c == '_' {
      continue;
    }
    let d = c.to_digit(10).ok_or_else(err)?;
    value = value.saturating_mul(10).saturating_add(i64::from(d));
  }
  Ok(if neg { -value } else { value })
}

/// Reads `<file>` as `[line.split("\t") for line in
/// open(file).read().strip().split("\n")]`, keeping the lines that the
/// Python code then unpacks as `char, phoneme` (a line with another number
/// of fields gives `ValueError`, as the unpacking does).
fn read_char_table(path: &Path) -> Result<Vec<(String, String)>, G2PWError> {
  let text = std::fs::read_to_string(path)
    .map_err(|e| G2PWError::FileNotFoundError(format!("{}: {e}", path.display())))?;
  py_strip(&text)
    .split('\n')
    .map(|line| {
      let fields: Vec<&str> = line.split('\t').collect();
      match fields.as_slice() {
        [char, phoneme] => Ok((char.to_string(), phoneme.to_string())),
        _ => Err(G2PWError::ValueError(format!(
          "expected 2 values to unpack, got {} in {line:?}",
          fields.len()
        ))),
      }
    })
    .collect()
}

/// The `style` argument (`self.style_convert_func`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum G2PWStyle {
  /// `"bopomofo"`: `lambda x: x`.
  Bopomofo,
  /// `"pinyin"`: `self._convert_bopomofo_to_pinyin`.
  Pinyin,
}

impl G2PWStyle {
  /// `{"bopomofo": ..., "pinyin": ...}[style]` (`KeyError` otherwise).
  pub fn from_name(style: &str) -> Result<Self, G2PWError> {
    match style {
      "bopomofo" => Ok(G2PWStyle::Bopomofo),
      "pinyin" => Ok(G2PWStyle::Pinyin),
      _ => Err(G2PWError::KeyError(style.to_string())),
    }
  }
}

/// Keyword arguments of `G2PWOnnxConverter(model_dir, style="bopomofo",
/// model_source=None, enable_non_tradional_chinese=False)`.
#[derive(Clone, Debug)]
pub struct G2PWOnnxConverterOptions {
  /// `style`.
  pub style: G2PWStyle,
  /// `model_source`: a directory with the `tokenizer.json` that
  /// `AutoTokenizer.from_pretrained(model_source)` loads. `None` uses the
  /// embedded `chinese-roberta-wwm-ext-large` tokenizer ([`TOKENIZER_JSON`]),
  /// which is the `bert_path` that GPT-SoVITS passes. (Python falls back to
  /// `config.model_source`, a Hugging Face hub name, for `None`.)
  pub model_source: Option<PathBuf>,
  /// `enable_non_tradional_chinese` (upstream spelling).
  pub enable_non_tradional_chinese: bool,
}

impl Default for G2PWOnnxConverterOptions {
  fn default() -> Self {
    Self {
      style: G2PWStyle::Bopomofo,
      model_source: None,
      enable_non_tradional_chinese: false,
    }
  }
}

/// `self.non_polyphonic`.
pub const NON_POLYPHONIC: [&str; 19] = [
  "一", "不", "和", "咋", "嗲", "剖", "差", "攢", "倒", "難", "奔", "勁", "拗", "肖", "瘙", "誒",
  "泊", "听", "噢",
];

/// `self.non_monophonic`.
pub const NON_MONOPHONIC: [&str; 2] = ["似", "攢"];

/// The return value of `_prepare_data`: `(texts, model_query_ids,
/// result_query_ids, sent_ids, partial_results)`.
pub type PreparedData = (
  Vec<String>,
  Vec<usize>,
  Vec<usize>,
  Vec<usize>,
  Vec<Vec<Option<String>>>,
);

/// Port of `_G2PWBaseOnnxConverter` and its subclass `G2PWOnnxConverter`
/// (the only subclass; it adds the ONNX Runtime session and `_predict`).
pub struct G2PWOnnxConverter {
  /// `self.model_dir`.
  pub model_dir: PathBuf,
  /// `self.config` (`load_config(model_dir/config.py, use_default=True)`).
  pub config: Config,
  /// `self.model_source` (the tokenizer source, see
  /// [`G2PWOnnxConverterOptions::model_source`]).
  pub model_source: String,
  /// `self.enable_opencc`.
  pub enable_opencc: bool,
  /// `self.tokenizer`.
  pub tokenizer: Tokenizer,
  /// `self.polyphonic_chars` (`POLYPHONIC_CHARS.txt`).
  pub polyphonic_chars: Vec<(String, String)>,
  /// `self.monophonic_chars` (`MONOPHONIC_CHARS.txt`).
  pub monophonic_chars: Vec<(String, String)>,
  /// `self.labels`.
  pub labels: Vec<String>,
  /// `self.char2phonemes`.
  pub char2phonemes: HashMap<String, Vec<usize>>,
  /// `self.chars` (sorted).
  pub chars: Vec<String>,
  /// `self.char2id`.
  pub char2id: HashMap<String, usize>,
  /// `self.char_phoneme_masks` (`None` without `config.use_mask`).
  pub char_phoneme_masks: Option<HashMap<String, Vec<u8>>>,
  /// `self.polyphonic_chars_new`.
  pub polyphonic_chars_new: HashSet<String>,
  /// `self.monophonic_chars_dict`.
  pub monophonic_chars_dict: HashMap<String, String>,
  /// `self.bopomofo_convert_dict` (`bopomofo_to_pinyin_wo_tune_dict.json`).
  pub bopomofo_convert_dict: HashMap<String, serde_json::Value>,
  /// The keys of `self.char_bopomofo_dict` (`char_bopomofo_dict.json`);
  /// only membership is used.
  pub char_bopomofo_dict: HashSet<String>,
  /// `self.style_convert_func`.
  pub style: G2PWStyle,
  /// `self.cc` (`OpenCC("s2tw")` when `enable_opencc`).
  pub cc: Option<&'static OpenCC>,
  /// `self.enable_sentence_dedup` (env `g2pw_sentence_dedup`, default on).
  pub enable_sentence_dedup: bool,
  /// `self.polyphonic_context_chars` (env `g2pw_polyphonic_context_chars`,
  /// default 16; 0 disables the crop).
  pub polyphonic_context_chars: usize,
  /// `self.session_g2pw`.
  pub session_g2pw: Mutex<Session>,
}

impl G2PWOnnxConverter {
  /// Port of `G2PWOnnxConverter.__init__` (with
  /// `_G2PWBaseOnnxConverter.__init__`).
  ///
  /// The session options mirror Python: `ORT_ENABLE_ALL`,
  /// `ORT_SEQUENTIAL`, `intra_op_num_threads = 2`, and the providers
  /// `[CUDA, CPU]` when ONNX Runtime has the CUDA provider, else `[CPU]`.
  pub fn new(
    model_dir: impl AsRef<Path>,
    options: &G2PWOnnxConverterOptions,
  ) -> Result<Self, G2PWError> {
    let model_dir = download_and_decompress(model_dir.as_ref())?;
    let config = load_config(&model_dir.join("config.py"), true)?;

    let (tokenizer_json, model_source) = match &options.model_source {
      Some(dir) => {
        let path = dir.join("tokenizer.json");
        let text = std::fs::read_to_string(&path)
          .map_err(|e| G2PWError::FileNotFoundError(format!("{}: {e}", path.display())))?;
        (text, dir.display().to_string())
      }
      None => (
        TOKENIZER_JSON.to_string(),
        "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large".to_string(),
      ),
    };
    let tokenizer = tokenizer_from_json(&tokenizer_json)?;

    let polyphonic_chars = read_char_table(&model_dir.join("POLYPHONIC_CHARS.txt"))?;
    let monophonic_chars = read_char_table(&model_dir.join("MONOPHONIC_CHARS.txt"))?;
    let (labels, char2phonemes) = if config.use_char_phoneme() {
      get_char_phoneme_labels(&polyphonic_chars)
    } else {
      get_phoneme_labels(&polyphonic_chars)
    };

    let mut chars: Vec<String> = char2phonemes.keys().cloned().collect();
    chars.sort();
    let char2id: HashMap<String, usize> = chars
      .iter()
      .enumerate()
      .map(|(idx, c)| (c.clone(), idx))
      .collect();
    let char_phoneme_masks = if config.use_mask() {
      Some(build_char_phoneme_masks(&char2phonemes, labels.len()))
    } else {
      None
    };

    let mut polyphonic_chars_new: HashSet<String> = chars.iter().cloned().collect();
    for char in NON_POLYPHONIC {
      polyphonic_chars_new.remove(char);
    }

    let mut monophonic_chars_dict: HashMap<String, String> = HashMap::new();
    for (char, phoneme) in &monophonic_chars {
      monophonic_chars_dict.insert(char.clone(), phoneme.clone());
    }
    for char in NON_MONOPHONIC {
      monophonic_chars_dict.remove(char);
    }

    // candidate_asset_dirs = [self.model_dir, default_asset_dir]; the second
    // is GPT_SoVITS/text/G2PWModel next to the Python sources, which the
    // port does not have.
    let candidate_asset_dirs = [Some(model_dir.as_path())];
    let bopomofo_convert_dict = match _load_json_from_candidates(
      "bopomofo_to_pinyin_wo_tune_dict.json",
      &candidate_asset_dirs,
    )? {
      serde_json::Value::Object(map) => map.into_iter().collect(),
      _ => {
        return Err(G2PWError::Json(
          "bopomofo_to_pinyin_wo_tune_dict.json is not an object".to_string(),
        ));
      }
    };
    let char_bopomofo_dict =
      match _load_json_from_candidates("char_bopomofo_dict.json", &candidate_asset_dirs)? {
        serde_json::Value::Object(map) => map.into_iter().map(|(k, _)| k).collect(),
        _ => {
          return Err(G2PWError::Json(
            "char_bopomofo_dict.json is not an object".to_string(),
          ));
        }
      };

    let enable_opencc = options.enable_non_tradional_chinese;
    let cc = enable_opencc.then(OpenCC::s2tw);
    let enable_sentence_dedup = matches!(
      py_strip(&std::env::var("g2pw_sentence_dedup").unwrap_or_else(|_| "true".to_string()))
        .to_lowercase()
        .as_str(),
      "1" | "true" | "yes" | "y" | "on"
    );
    // 聚焦到多音字附近上下文，默认左右各16字；设为0表示关闭裁剪（整句）。
    let polyphonic_context_chars =
      py_int(&std::env::var("g2pw_polyphonic_context_chars").unwrap_or_else(|_| "16".to_string()))?
        .max(0) as usize;

    let onnx_path =
      _find_first_existing_file(&[model_dir.join("g2pW.onnx"), model_dir.join("g2pw.onnx")])?;
    let mut providers = Vec::new();
    if CUDA::default().is_available().unwrap_or(false) {
      providers.push(CUDA::default().build());
    }
    providers.push(CPU::default().build());
    let session = Session::builder()?
      .with_optimization_level(GraphOptimizationLevel::All)?
      .with_parallel_execution(false)?
      .with_intra_threads(2)?
      .with_execution_providers(providers)?
      .commit_from_file(&onnx_path)?;

    Ok(Self {
      model_dir,
      config,
      model_source,
      enable_opencc,
      tokenizer,
      polyphonic_chars,
      monophonic_chars,
      labels,
      char2phonemes,
      chars,
      char2id,
      char_phoneme_masks,
      polyphonic_chars_new,
      monophonic_chars_dict,
      bopomofo_convert_dict,
      char_bopomofo_dict,
      style: options.style,
      cc,
      enable_sentence_dedup,
      polyphonic_context_chars,
      session_g2pw: Mutex::new(session),
    })
  }

  /// Port of `_convert_bopomofo_to_pinyin(bopomofo)`. `None` where Python
  /// prints a warning and returns `None`; a last character outside `12345`
  /// gives `AssertionError`.
  pub fn _convert_bopomofo_to_pinyin(&self, bopomofo: &str) -> Result<Option<String>, G2PWError> {
    let mut chars: Vec<char> = bopomofo.chars().collect();
    let tone = chars
      .pop()
      .ok_or_else(|| G2PWError::IndexError("string index out of range".to_string()))?;
    if !"12345".contains(tone) {
      return Err(G2PWError::AssertionError(format!(
        "tone {tone:?} of {bopomofo:?}"
      )));
    }
    let key: String = chars.into_iter().collect();
    match self.bopomofo_convert_dict.get(&key) {
      // `if component:` (a non-empty string; the JSON values are strings)
      Some(serde_json::Value::String(component)) if !component.is_empty() => {
        Ok(Some(format!("{component}{tone}")))
      }
      Some(v) if py_json_truthy(v) => Err(G2PWError::TypeError(format!(
        "can only concatenate str to {v} of {bopomofo:?}"
      ))),
      _ => {
        // print(f'Warning: "{bopomofo}" cannot convert to pinyin')
        Ok(None)
      }
    }
  }

  /// `self.style_convert_func(x)`.
  pub fn style_convert_func(&self, x: &str) -> Result<Option<String>, G2PWError> {
    match self.style {
      G2PWStyle::Bopomofo => Ok(Some(x.to_string())),
      G2PWStyle::Pinyin => self._convert_bopomofo_to_pinyin(x),
    }
  }

  /// Port of `__call__(sentences)`. Python also accepts one `str`; pass a
  /// one-element slice.
  ///
  /// With OpenCC enabled, a sentence whose `s2tw` conversion has another
  /// length gives `AssertionError`.
  pub fn call<S: AsRef<str>>(
    &self,
    sentences: &[S],
  ) -> Result<Vec<Vec<Option<String>>>, G2PWError> {
    let mut sentences: Vec<String> = sentences.iter().map(|s| s.as_ref().to_string()).collect();

    if let Some(cc) = self.cc {
      let mut translated_sentences = Vec::with_capacity(sentences.len());
      for sent in &sentences {
        let translated_sent = cc.convert(sent);
        if translated_sent.chars().count() != sent.chars().count() {
          return Err(G2PWError::AssertionError(format!(
            "OpenCC s2tw changed the length of {sent:?} to {translated_sent:?}"
          )));
        }
        translated_sentences.push(translated_sent);
      }
      sentences = translated_sentences;
    }

    let (texts, model_query_ids, result_query_ids, sent_ids, partial_results) =
      self._prepare_data(&sentences)?;
    if texts.is_empty() {
      return Ok(partial_results);
    }

    let Some(model_input) = prepare_onnx_input(
      &self.tokenizer,
      &self.labels,
      &self.char2phonemes,
      &self.chars,
      &texts,
      &model_query_ids,
      self.config.use_mask(),
      None,
      512,
      Some(&self.char2id),
      self.char_phoneme_masks.as_ref(),
    )?
    else {
      return Ok(partial_results);
    };

    let (mut preds, _confidences) = if self.enable_sentence_dedup {
      self._predict_with_sentence_dedup(&model_input, &texts)?
    } else {
      self._predict(&model_input)?
    };

    if self.config.use_char_phoneme() {
      preds = preds
        .iter()
        .map(|pred| {
          pred
            .split(' ')
            .nth(1)
            .map(str::to_string)
            .ok_or_else(|| G2PWError::IndexError("list index out of range".to_string()))
        })
        .collect::<Result<_, _>>()?;
    }

    let mut results = partial_results;
    for ((sent_id, query_id), pred) in sent_ids
      .iter()
      .zip(result_query_ids.iter())
      .zip(preds.iter())
    {
      results[*sent_id][*query_id] = self.style_convert_func(pred)?;
    }

    Ok(results)
  }

  /// Port of `_prepare_data(sentences)`.
  ///
  /// `pinyin(sent_s, ...)` groups each run of non-Han characters into one
  /// item, so `pypinyin_result[i]` is not aligned with `sent[i]` after such
  /// a run of two or more characters; this is kept, and an index past the
  /// end gives `IndexError` as in Python.
  // The `char_bopomofo_dict` branch has the same body as the `else` branch
  // upstream; both are kept.
  #[allow(clippy::if_same_then_else)]
  pub fn _prepare_data<S: AsRef<str>>(&self, sentences: &[S]) -> Result<PreparedData, G2PWError> {
    let mut texts = Vec::new();
    let mut model_query_ids = Vec::new();
    let mut result_query_ids = Vec::new();
    let mut sent_ids = Vec::new();
    let mut partial_results = Vec::new();
    let pinyin_options = PinyinOptions {
      style: Style::TONE3,
      neutral_tone_with_five: true,
      ..PinyinOptions::default()
    };
    for (sent_id, sent) in sentences.iter().enumerate() {
      let sent = sent.as_ref();
      let sent_s = tranditional_to_simplified(sent);
      let pypinyin_result = pypinyin::pinyin(sent_s.as_str(), &pinyin_options);
      let sent_chars: Vec<char> = sent.chars().collect();
      let mut partial_result: Vec<Option<String>> = vec![None; sent_chars.len()];
      let mut polyphonic_indices: Vec<usize> = Vec::new();
      let pypinyin_at = |i: usize| -> Result<Option<String>, G2PWError> {
        pypinyin_result
          .get(i)
          .and_then(|item| item.first())
          .cloned()
          .map(Some)
          .ok_or_else(|| G2PWError::IndexError("list index out of range".to_string()))
      };
      for (i, char) in sent_chars.iter().enumerate() {
        let char = char.to_string();
        if self.polyphonic_chars_new.contains(&char) {
          polyphonic_indices.push(i);
        } else if let Some(phoneme) = self.monophonic_chars_dict.get(&char) {
          partial_result[i] = self.style_convert_func(phoneme)?;
        } else if self.char_bopomofo_dict.contains(&char) {
          partial_result[i] = pypinyin_at(i)?;
        } else {
          partial_result[i] = pypinyin_at(i)?;
        }
      }

      if let (Some(&first), Some(&last)) = (polyphonic_indices.first(), polyphonic_indices.last()) {
        let (sent_for_predict, query_offset) = if self.polyphonic_context_chars > 0 {
          let left = first.saturating_sub(self.polyphonic_context_chars);
          let right = sent_chars
            .len()
            .min(last + self.polyphonic_context_chars + 1);
          (sent_chars[left..right].iter().collect::<String>(), left)
        } else {
          (sent.to_string(), 0)
        };

        for &index in &polyphonic_indices {
          texts.push(sent_for_predict.clone());
          model_query_ids.push(index - query_offset);
          result_query_ids.push(index);
          sent_ids.push(sent_id);
        }
      }

      partial_results.push(partial_result);
    }
    Ok((
      texts,
      model_query_ids,
      result_query_ids,
      sent_ids,
      partial_results,
    ))
  }

  /// Port of `G2PWOnnxConverter._predict(model_input)`.
  pub fn _predict(&self, model_input: &OnnxInput) -> Result<(Vec<String>, Vec<f64>), G2PWError> {
    let mut session = self.session_g2pw.lock().unwrap_or_else(|e| e.into_inner());
    predict(&mut session, model_input, &self.labels)
  }

  /// The `session.run(...)[0]` probabilities that [`Self::_predict`] takes
  /// the argmax of.
  pub fn _predict_probs(&self, model_input: &OnnxInput) -> Result<Array2<f32>, G2PWError> {
    let mut session = self.session_g2pw.lock().unwrap_or_else(|e| e.into_inner());
    run_session(&mut session, model_input)
  }

  /// Port of `_predict_with_sentence_dedup(model_input, texts)`: one model
  /// run per distinct text; for a text that occurs more than once, the
  /// token inputs are passed once (batch 1) with all its queries.
  pub fn _predict_with_sentence_dedup(
    &self,
    model_input: &OnnxInput,
    texts: &[String],
  ) -> Result<(Vec<String>, Vec<f64>), G2PWError> {
    if texts.len() <= 1 {
      return self._predict(model_input);
    }

    let groups = group_indices(texts);
    if groups.iter().all(|indices| indices.len() == 1) {
      return self._predict(model_input);
    }

    let mut preds: Vec<String> = vec![String::new(); texts.len()];
    let mut confidences: Vec<f64> = vec![0.0; texts.len()];
    for indices in &groups {
      let group_input = sentence_dedup_group_input(model_input, indices);
      let (group_preds, group_confidences) = self._predict(&group_input)?;
      for ((&output_idx, pred), confidence) in
        indices.iter().zip(group_preds).zip(group_confidences)
      {
        preds[output_idx] = pred;
        confidences[output_idx] = confidence;
      }
    }

    Ok((preds, confidences))
  }
}

/// `grouped_indices.setdefault(text, []).append(idx)`: the indices of each
/// distinct text, in order of first occurrence.
pub fn group_indices(texts: &[String]) -> Vec<Vec<usize>> {
  let mut order: HashMap<&str, usize> = HashMap::new();
  let mut groups: Vec<Vec<usize>> = Vec::new();
  for (idx, text) in texts.iter().enumerate() {
    let slot = *order.entry(text.as_str()).or_insert_with(|| {
      groups.push(Vec::new());
      groups.len() - 1
    });
    groups[slot].push(idx);
  }
  groups
}

/// The `group_input` of `_predict_with_sentence_dedup`.
pub fn sentence_dedup_group_input(model_input: &OnnxInput, indices: &[usize]) -> OnnxInput {
  let mut group_input = model_input.select(indices);
  if indices.len() > 1 {
    group_input.input_ids = group_input.input_ids.slice(ndarray::s![..1, ..]).to_owned();
    group_input.token_type_ids = group_input
      .token_type_ids
      .slice(ndarray::s![..1, ..])
      .to_owned();
    group_input.attention_masks = group_input
      .attention_masks
      .slice(ndarray::s![..1, ..])
      .to_owned();
  }
  group_input
}

/// Python truthiness of a JSON value.
fn py_json_truthy(v: &serde_json::Value) -> bool {
  match v {
    serde_json::Value::Null => false,
    serde_json::Value::Bool(b) => *b,
    serde_json::Value::Number(n) => n.as_f64() != Some(0.0),
    serde_json::Value::String(s) => !s.is_empty(),
    serde_json::Value::Array(a) => !a.is_empty(),
    serde_json::Value::Object(o) => !o.is_empty(),
  }
}

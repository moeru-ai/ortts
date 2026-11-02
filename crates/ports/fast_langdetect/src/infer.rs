//! Port of `fast_langdetect/infer.py` (fast-langdetect 1.1.0).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};

use crate::FastLangdetectError;
use crate::fasttext::{self, FastText};

/// Port of `FASTTEXT_LARGE_MODEL_NAME`.
pub const FASTTEXT_LARGE_MODEL_NAME: &str = "lid.176.bin";
/// Port of `FASTTEXT_LARGE_MODEL_URL` (informational; downloading is not ported).
pub const FASTTEXT_LARGE_MODEL_URL: &str =
  "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin";
/// Port of `FASTTEXT_LARGE_MODEL_MD5` (informational; not verified).
pub const FASTTEXT_LARGE_MODEL_MD5: &str = "01810bc59c6a3d2b79c79e6336612f65";

/// Port of `CACHE_DIRECTORY`: `$FTLANG_CACHE`, or `<tempdir>/fasttext-langdetect`.
pub fn cache_directory() -> PathBuf {
  match std::env::var_os("FTLANG_CACHE") {
    Some(v) => PathBuf::from(v),
    None => std::env::temp_dir().join("fasttext-langdetect"),
  }
}

/// Port of the `Literal["lite", "full", "auto"]` model selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Model {
  /// `"lite"`: the bundled quantized `lid.176.ftz` (not supported by this port).
  Lite,
  /// `"full"`: `lid.176.bin` in `cache_dir` (what GPT-SoVITS uses).
  Full,
  /// `"auto"`: `"full"`, falling back to `"lite"` on `MemoryError`.
  #[default]
  Auto,
}

/// Port of `class LangDetectConfig`.
#[derive(Debug, Clone)]
pub struct LangDetectConfig {
  /// `cache_dir`: directory holding `lid.176.bin`.
  pub cache_dir: PathBuf,
  /// `custom_model_path`.
  pub custom_model_path: Option<PathBuf>,
  /// `normalize_input`.
  pub normalize_input: bool,
  /// `max_input_length` (characters).
  pub max_input_length: Option<usize>,
  /// `model`: default selection when `detect()` is called without one.
  pub model: Model,
}

impl Default for LangDetectConfig {
  /// `LangDetectConfig()` with the Python defaults.
  fn default() -> Self {
    Self {
      cache_dir: cache_directory(),
      custom_model_path: None,
      normalize_input: true,
      max_input_length: Some(80),
      model: Model::Auto,
    }
  }
}

impl LangDetectConfig {
  /// Port of `LangDetectConfig.__init__`: raises `FileNotFoundError` when
  /// `custom_model_path` does not exist.
  pub fn new(
    cache_dir: Option<PathBuf>,
    custom_model_path: Option<PathBuf>,
    normalize_input: bool,
    max_input_length: Option<usize>,
    model: Model,
  ) -> Result<Self, FastLangdetectError> {
    if let Some(p) = &custom_model_path
      && !p.exists()
    {
      return Err(FastLangdetectError::FileNotFound(format!(
        "fast-langdetect: Target model file not found: {}",
        p.display()
      )));
    }
    Ok(Self {
      cache_dir: cache_dir.unwrap_or_else(cache_directory),
      custom_model_path,
      normalize_input,
      max_input_length,
      model,
    })
  }
}

/// Keyword arguments of `LangDetector.detect()` / `detect()`.
#[derive(Debug, Clone, Copy)]
pub struct DetectOptions {
  /// `model` (`None` uses `config.model`).
  pub model: Option<Model>,
  /// `k`.
  pub k: i32,
  /// `threshold`.
  pub threshold: f64,
}

impl Default for DetectOptions {
  fn default() -> Self {
    Self {
      model: None,
      k: 1,
      threshold: 0.0,
    }
  }
}

/// One `{"lang": ..., "score": ...}` dict returned by `detect()`.
#[derive(Debug, Clone, PartialEq)]
pub struct DetectResult {
  /// `"lang"`: the label without `__label__`.
  pub lang: String,
  /// `"score"`: `min(float(score), 1.0)`.
  pub score: f64,
}

/// Port of `class LangDetector`.
#[derive(Debug)]
pub struct LangDetector {
  /// `config`.
  pub config: LangDetectConfig,
  models: Mutex<HashMap<&'static str, Arc<FastText>>>,
}

/// Python `str.istitle` building block: characters of general category `Lt`.
fn is_titlecase(c: char) -> bool {
  matches!(c as u32,
        0x01C5 | 0x01C8 | 0x01CB | 0x01F2
        | 0x1F88..=0x1F8F | 0x1F98..=0x1F9F | 0x1FA8..=0x1FAF
        | 0x1FBC | 0x1FCC | 0x1FFC)
}

/// Port of Python `str.isupper()`.
fn py_isupper(text: &str) -> bool {
  let mut cased = false;
  for c in text.chars() {
    if c.is_lowercase() || is_titlecase(c) {
      return false;
    } else if !cased && c.is_uppercase() {
      cased = true;
    }
  }
  cased
}

impl LangDetector {
  /// Port of `LangDetector.__init__(config)`.
  pub fn new(config: LangDetectConfig) -> Self {
    Self {
      config,
      models: Mutex::new(HashMap::new()),
    }
  }

  /// Port of `LangDetector._preprocess_text`.
  pub fn _preprocess_text(&self, text: &str) -> String {
    let mut text = if text.contains('\n') {
      text.replace('\n', " ")
    } else {
      text.to_string()
    };
    if let Some(max) = self.config.max_input_length
      && text.chars().count() > max
    {
      text = text.chars().take(max).collect();
    }
    text
  }

  /// Port of `LangDetector._normalize_text`.
  pub fn _normalize_text(text: &str, should_normalize: bool) -> String {
    if !should_normalize {
      return text.to_string();
    }
    let upper = text.bytes().filter(u8::is_ascii_uppercase).count();
    let alpha = text.bytes().filter(u8::is_ascii_alphabetic).count();
    if py_isupper(text) || (upper as f64 > 0.8 * alpha as f64 && text.chars().count() > 5) {
      return text.to_lowercase();
    }
    text.to_string()
  }

  /// Port of `LangDetector._get_model(low_memory)`.
  ///
  /// The lite model is the quantized `lid.176.ftz` bundled with the Python
  /// package; it is not ported, so `low_memory=True` without a
  /// `custom_model_path` fails. The full model is never downloaded.
  pub fn _get_model(&self, low_memory: bool) -> Result<Arc<FastText>, FastLangdetectError> {
    let cache_key = if low_memory {
      "low_memory"
    } else {
      "high_memory"
    };
    let mut models = self.models.lock().expect("model cache poisoned");
    if let Some(m) = models.get(cache_key) {
      return Ok(m.clone());
    }
    let path: PathBuf = if let Some(p) = &self.config.custom_model_path {
      p.clone()
    } else if low_memory {
      return Err(FastLangdetectError::Unsupported(
        "fast-langdetect: the bundled lite model (lid.176.ftz) is not ported".into(),
      ));
    } else {
      self.config.cache_dir.join(FASTTEXT_LARGE_MODEL_NAME)
    };
    let model = Arc::new(load_local(&path)?);
    models.insert(cache_key, model.clone());
    Ok(model)
  }

  /// Port of `LangDetector.detect(text, *, model, k, threshold)`.
  pub fn detect(
    &self,
    text: &str,
    options: &DetectOptions,
  ) -> Result<Vec<DetectResult>, FastLangdetectError> {
    let sel_model = options.model.unwrap_or(self.config.model);
    let ft_model = match sel_model {
      Model::Lite => self._get_model(true)?,
      Model::Full | Model::Auto => self._get_model(false)?,
    };
    let text = self._preprocess_text(text);
    let normalized_text = Self::_normalize_text(&text, self.config.normalize_input);
    let (labels, scores) = ft_model.predict(&normalized_text, options.k, options.threshold)?;
    let mut results: Vec<DetectResult> = labels
      .into_iter()
      .zip(scores)
      .map(|(label, score)| DetectResult {
        lang: label.replace("__label__", ""),
        score: score.min(1.0),
      })
      .collect();
    // sorted(..., key=score, reverse=True): stable, descending.
    results.sort_by(|a, b| {
      b.score
        .partial_cmp(&a.score)
        .unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(results)
  }
}

/// Port of `ModelLoader.load_local` / `ModelLoader._load_unix`.
fn load_local(path: &Path) -> Result<FastText, FastLangdetectError> {
  if !path.exists() {
    return Err(FastLangdetectError::FileNotFound(format!(
      "Model file not found: {}",
      path.display()
    )));
  }
  fasttext::load_model(path).map_err(|e| match e {
    FastLangdetectError::FileNotFound(_) => e,
    other => FastLangdetectError::ModelLoad(format!(
      "fast-langdetect: Failed to load model: {}",
      other.message()
    )),
  })
}

static DEFAULT_DETECTOR: RwLock<Option<Arc<LangDetector>>> = RwLock::new(None);

/// Port of `_default_detector`: the global detector, created with
/// `LangDetectConfig()` on first use.
pub fn _default_detector() -> Arc<LangDetector> {
  if let Some(d) = DEFAULT_DETECTOR.read().expect("poisoned").as_ref() {
    return d.clone();
  }
  let mut w = DEFAULT_DETECTOR.write().expect("poisoned");
  w.get_or_insert_with(|| Arc::new(LangDetector::new(LangDetectConfig::default())))
    .clone()
}

/// Replaces `_default_detector`, like GPT-SoVITS does in
/// `LangSegmenter/langsegmenter.py`:
/// `fast_langdetect.infer._default_detector = LangDetector(LangDetectConfig(cache_dir=...))`.
pub fn set_default_detector(detector: LangDetector) {
  *DEFAULT_DETECTOR.write().expect("poisoned") = Some(Arc::new(detector));
}

/// Port of the module-level `detect(text, *, model, k, threshold, config)`.
///
/// With `config`, a fresh `LangDetector(config)` is created (and the model
/// reloaded) for this call, as in Python.
pub fn detect(
  text: &str,
  options: &DetectOptions,
  config: Option<&LangDetectConfig>,
) -> Result<Vec<DetectResult>, FastLangdetectError> {
  match config {
    None => _default_detector().detect(text, options),
    Some(c) => LangDetector::new(c.clone()).detect(text, options),
  }
}

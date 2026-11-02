//! Port of `split_lang/detect_lang/detector.py`.
//!
//! Python calls the module-level `fast_langdetect.detect`, which uses
//! `fast_langdetect.infer._default_detector`. Every function here takes a
//! `detector`: `None` does the same (it uses
//! [`fast_langdetect::infer::_default_detector`], which GPT-SoVITS replaces
//! with [`fast_langdetect::set_default_detector`]); `Some` uses that detector
//! instead.

use fast_langdetect::{DetectOptions, DetectResult, FastLangdetectError, LangDetector, Model};

use crate::model::LangSectionType;
use crate::split::utils::contains_ja_kana;
use crate::{SplitLangError, py};

fn detect(
  text: &str,
  options: &DetectOptions,
  detector: Option<&LangDetector>,
) -> Result<Vec<DetectResult>, FastLangdetectError> {
  match detector {
    Some(d) => d.detect(text, options),
    None => fast_langdetect::detect(text, options, None),
  }
}

/// Port of `fast_lang_detect(text)`:
/// `str(fast_langdetect.detect(text, model="full")[0]["lang"]).lower()`.
pub fn fast_lang_detect(
  text: &str,
  detector: Option<&LangDetector>,
) -> Result<String, SplitLangError> {
  let options = DetectOptions {
    model: Some(Model::Full),
    k: 1,
    threshold: 0.0,
  };
  let result = detect(text, &options, detector)?;
  let first = result.first().ok_or_else(crate::index_error)?;
  Ok(first.lang.to_lowercase())
}

/// Port of `detect_lang_combined(text, lang_section_type)`.
pub fn detect_lang_combined(
  text: &str,
  lang_section_type: LangSectionType,
  detector: Option<&LangDetector>,
) -> Result<String, SplitLangError> {
  if lang_section_type == LangSectionType::ZhJa {
    if contains_ja_kana(text) {
      return Ok("ja".to_string());
    }
    return fast_lang_detect(text, detector);
  }
  fast_lang_detect(text, detector)
}

/// Port of `possible_detection_list(text)`: the languages of
/// `fast_langdetect.detect(text.replace("\n", "").strip(), model="full", k=5, threshold=0.01)`.
pub fn possible_detection_list(
  text: &str,
  detector: Option<&LangDetector>,
) -> Result<Vec<String>, SplitLangError> {
  let text = text.replace('\n', "");
  let text = py::strip(&text);
  let options = DetectOptions {
    model: Some(Model::Full),
    k: 5,
    threshold: 0.01,
  };
  Ok(
    detect(text, &options, detector)?
      .into_iter()
      .map(|r| r.lang)
      .collect(),
  )
}

//! Port of `GPT_SoVITS/TTS_infer_pack/TextPreprocessor.py` (GPT-SoVITS @ 04a44eb).
//!
//! # Ported scope
//!
//! `punctuation`, `get_first`, `merge_short_text_in_array` and the
//! `TextPreprocessor` class: `__init__` (as [`TextPreprocessor::new`] and
//! [`TextPreprocessor::from_paths`]), `preprocess`, `pre_seg_text`,
//! `segment_and_extract_feature_for_text`, `get_phones_and_bert` (every
//! language mode), `get_bert_feature`, `clean_text_inf`, `get_bert_inf`,
//! `filter_text` and `replace_consecutive_punctuation`.
//!
//! # BERT model
//!
//! Python calls `self.bert_model(**inputs, output_hidden_states=True)` and
//! keeps `torch.cat(res["hidden_states"][-3:-2], -1)[0]`: the third hidden
//! state from the end, one row per token. The port puts that call behind the
//! [`BertFeatureModel`] trait, so a backend can plug in an ONNX export of
//! `chinese-roberta-wwm-ext-large`. The tokenizer is that model's
//! `tokenizer.json`, embedded in the `g2pw` module
//! ([`crate::g2pw::onnx_api::TOKENIZER_JSON`]). Tensors are `ndarray`
//! arrays of `f32`; a BERT feature is `[hidden, n_phones]` (`[1024, n]`).
//!
//! # Differences in form (not in output)
//!
//! - Where Python raises, the methods return an [`Error`] that names the
//!   Python exception ([`Error::name`]). `torch.cat` of an empty list (a
//!   `zh` segment whose normalized text is empty, or an empty `textlist`)
//!   is `ValueError` (`torch.cat(): expected a non-empty list of Tensors`,
//!   torch 2.x); `res[i]` past the end of the BERT output is `IndexError`.
//! - Korean (`ko`, `all_ko`, and `ko` segments that `LangSegmenter` finds in
//!   `auto` and `auto_yue` text) is not ported: [`crate::cleaner`] returns
//!   [`crate::cleaner::Error::NotPorted`] when a `ko` segment reaches
//!   `clean_text`. Modes that never produce such a segment work as upstream.
//! - The `print` calls, the `tqdm` progress bar, `self.device` and
//!   `self.bert_lock` are left out. The `ValueError` of `filter_text` holds
//!   the untranslated i18n key `请输入有效文本`.

use std::fmt;
use std::path::Path;

use fast_langdetect::{LangDetectConfig, LangDetector, Model};
use ndarray::{Array2, ArrayView2, Axis, concatenate, s};
use split_lang::SplitLangError;
use tokenizers::Tokenizer;

use crate::chinese2::py_isalpha;
use crate::cleaner::{self, Cleaner};
use crate::g2pw::onnx_api::{TOKENIZER_JSON, tokenizer_from_json};
use crate::lang_segmenter::LangSegmenter;
use crate::tone_sandhi::py_isnumeric;
use crate::tts_infer_pack::text_segmentation_method::{
  self, SPLITS, SplitBigTextOptions, get_method as get_seg_method, split_big_text,
};
use crate::zh_normalization::pyre::py_strip;
use crate::{KeyError, cleaned_text_to_sequence};

/// Port of the module-level `punctuation` set (`{"!", "?", "…", ",", ".",
/// "-"}`, no space, unlike `text_segmentation_method.punctuation`).
pub const PUNCTUATION: &[char] = &['!', '?', '…', ',', '.', '-'];

/// A boxed error from a [`BertFeatureModel`].
pub type BertError = Box<dyn std::error::Error + Send + Sync>;

/// The tokenizer output for one text (`self.tokenizer(text,
/// return_tensors="pt")`, batch size 1): `input_ids`, `token_type_ids` and
/// `attention_mask`, with `[CLS]` and `[SEP]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BertInputs {
  /// `input_ids[0]`.
  pub input_ids: Vec<i64>,
  /// `token_type_ids[0]`.
  pub token_type_ids: Vec<i64>,
  /// `attention_mask[0]`.
  pub attention_mask: Vec<i64>,
}

/// The BERT model of `TextPreprocessor` (`self.bert_model`).
pub trait BertFeatureModel {
  /// `torch.cat(self.bert_model(**inputs, output_hidden_states=True)
  /// ["hidden_states"][-3:-2], -1)[0]`: the third hidden state from the
  /// end, shape `[n_tokens, hidden]` (`hidden` is 1024 for
  /// `chinese-roberta-wwm-ext-large`), one row per token of `inputs`.
  fn hidden_states(&self, inputs: &BertInputs) -> Result<Array2<f32>, BertError>;
}

impl<T: BertFeatureModel + ?Sized> BertFeatureModel for &T {
  fn hidden_states(&self, inputs: &BertInputs) -> Result<Array2<f32>, BertError> {
    (**self).hidden_states(inputs)
  }
}

impl<T: BertFeatureModel + ?Sized> BertFeatureModel for Box<T> {
  fn hidden_states(&self, inputs: &BertInputs) -> Result<Array2<f32>, BertError> {
    (**self).hidden_states(inputs)
  }
}

/// An exception raised by `TextPreprocessor` or the code it calls.
#[derive(Debug)]
pub enum Error {
  /// An exception of `clean_text` (see [`cleaner::Error`]).
  Cleaner(cleaner::Error),
  /// `KeyError(symbol)` from `cleaned_text_to_sequence`.
  KeyError(KeyError),
  /// An exception of `LangSegmenter.getTexts`.
  LangSegmenter(SplitLangError),
  /// `IndexError` from `text_segmentation_method` (`cut1("")`, ...) or from
  /// `res[i]` in `get_bert_feature`.
  IndexError(String),
  /// `ValueError`: `filter_text` on a list of empty texts, `get_method` of
  /// an unknown method name, or `torch.cat` of an empty list.
  ValueError(String),
  /// `AssertionError` from `assert len(word2ph) == len(text)`.
  AssertionError(&'static str),
  /// `TypeError`: `len(word2ph)` on `None` (`get_bert_feature` of a
  /// language without `word2ph`; upstream never reaches it).
  TypeError(String),
  /// `RuntimeError` from `torch.cat` of tensors whose sizes do not match.
  RuntimeError(String),
  /// The tokenizer failed (the port's own error).
  Tokenizer(String),
  /// The BERT model failed (the port's own error).
  Bert(BertError),
}

impl Error {
  /// The Python exception class name; `"<port error>"` for the tokenizer
  /// and BERT model errors of the port.
  pub fn name(&self) -> &'static str {
    match self {
      Error::Cleaner(e) => e.name(),
      Error::KeyError(_) => "KeyError",
      Error::LangSegmenter(e) => e.exception_name(),
      Error::IndexError(_) => "IndexError",
      Error::ValueError(_) => "ValueError",
      Error::AssertionError(_) => "AssertionError",
      Error::TypeError(_) => "TypeError",
      Error::RuntimeError(_) => "RuntimeError",
      Error::Tokenizer(_) | Error::Bert(_) => "<port error>",
    }
  }
}

impl fmt::Display for Error {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      Error::Cleaner(e) => write!(f, "{e}"),
      Error::KeyError(e) => write!(f, "{e}"),
      Error::LangSegmenter(e) => write!(f, "{e}"),
      Error::IndexError(m) => write!(f, "IndexError: {m}"),
      Error::ValueError(m) => write!(f, "ValueError: {m}"),
      Error::AssertionError(c) => write!(f, "AssertionError: {c}"),
      Error::TypeError(m) => write!(f, "TypeError: {m}"),
      Error::RuntimeError(m) => write!(f, "RuntimeError: {m}"),
      Error::Tokenizer(m) => write!(f, "tokenizer error: {m}"),
      Error::Bert(e) => write!(f, "BERT model error: {e}"),
    }
  }
}

impl std::error::Error for Error {}

impl From<cleaner::Error> for Error {
  fn from(e: cleaner::Error) -> Self {
    Error::Cleaner(e)
  }
}

impl From<KeyError> for Error {
  fn from(e: KeyError) -> Self {
    Error::KeyError(e)
  }
}

impl From<SplitLangError> for Error {
  fn from(e: SplitLangError) -> Self {
    Error::LangSegmenter(e)
  }
}

impl From<text_segmentation_method::IndexError> for Error {
  fn from(e: text_segmentation_method::IndexError) -> Self {
    Error::IndexError(e.0.to_string())
  }
}

impl From<text_segmentation_method::ValueError> for Error {
  fn from(e: text_segmentation_method::ValueError) -> Self {
    Error::ValueError(e.0)
  }
}

/// Python `str.isalnum()` for one character (what `\w` matches, with `_`).
fn py_isalnum(c: char) -> bool {
  // isalnum = isalpha or isdecimal or isdigit or isnumeric, and isnumeric
  // includes the other two.
  py_isalpha(c) || py_isnumeric(c)
}

/// `re.sub("\W+", "", text)` is empty: no character is a word character.
fn has_no_word_char(text: &str) -> bool {
  !text.chars().any(|c| c == '_' || py_isalnum(c))
}

/// Port of `get_first(text)`: the text before the first character of
/// `splits`, stripped.
pub fn get_first(text: &str) -> String {
  let first = match text.find(|c: char| SPLITS.contains(&c)) {
    Some(i) => &text[..i],
    None => text,
  };
  py_strip(first).to_string()
}

/// Port of `merge_short_text_in_array(texts, threshold)`: joins consecutive
/// texts until each piece has at least `threshold` characters; a short
/// remainder is appended to the last piece.
pub fn merge_short_text_in_array(texts: Vec<String>, threshold: usize) -> Vec<String> {
  if texts.len() < 2 {
    return texts;
  }
  let mut result: Vec<String> = Vec::new();
  let mut text = String::new();
  for ele in &texts {
    text.push_str(ele);
    if text.chars().count() >= threshold {
      result.push(std::mem::take(&mut text));
    }
  }
  if !text.is_empty() {
    match result.last_mut() {
      None => result.push(text),
      Some(last) => last.push_str(&text),
    }
  }
  result
}

/// `re.sub(r' {2,}', ' ', text)`.
fn collapse_spaces(text: &str) -> String {
  let mut out = String::with_capacity(text.len());
  let mut prev_space = false;
  for c in text.chars() {
    if c == ' ' && prev_space {
      continue;
    }
    prev_space = c == ' ';
    out.push(c);
  }
  out
}

/// `{"phones": ..., "bert_features": ..., "norm_text": ...}`, one element
/// of the `preprocess` result.
#[derive(Debug, Clone, PartialEq)]
pub struct PreprocessResult {
  /// `phones`: symbol ids.
  pub phones: Vec<usize>,
  /// `bert_features`: `[hidden, len(phones)]`.
  pub bert_features: Array2<f32>,
  /// `norm_text`.
  pub norm_text: String,
}

/// Port of `class TextPreprocessor`.
///
/// It owns the BERT model, its tokenizer, and the language modules of
/// `clean_text` ([`Cleaner`], which holds `chinese2` with its G2PW model).
/// `LangSegmenter` uses fast-langdetect's process-wide default detector, as
/// in Python; [`TextPreprocessor::from_paths`] sets it.
pub struct TextPreprocessor<M> {
  /// `self.bert_model`.
  pub bert_model: M,
  /// `self.tokenizer`.
  pub tokenizer: Tokenizer,
  /// The language modules that `clean_text` imports.
  pub cleaner: Cleaner,
}

impl<M> fmt::Debug for TextPreprocessor<M> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("TextPreprocessor")
      .field("cleaner", &self.cleaner)
      .finish_non_exhaustive()
  }
}

/// An error of [`TextPreprocessor::from_paths`].
#[derive(Debug)]
pub enum InitError {
  /// Loading the G2PW model of `chinese2` failed.
  Chinese2(crate::chinese2::Error),
  /// The fasttext model file does not exist.
  FastLangdetect(fast_langdetect::FastLangdetectError),
  /// The embedded tokenizer did not load.
  Tokenizer(String),
}

impl fmt::Display for InitError {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      InitError::Chinese2(e) => write!(f, "G2PW model: {e}"),
      InitError::FastLangdetect(e) => write!(f, "fast-langdetect model: {e}"),
      InitError::Tokenizer(m) => write!(f, "tokenizer: {m}"),
    }
  }
}

impl std::error::Error for InitError {}

impl<M: BertFeatureModel> TextPreprocessor<M> {
  /// Port of `TextPreprocessor.__init__(bert_model, tokenizer, device)`,
  /// with the language modules as an argument (Python imports them).
  pub fn new(bert_model: M, tokenizer: Tokenizer, cleaner: Cleaner) -> Self {
    Self {
      bert_model,
      tokenizer,
      cleaner,
    }
  }

  /// Builds everything GPT-SoVITS sets up at import time:
  ///
  /// - the `chinese-roberta-wwm-ext-large` tokenizer (embedded);
  /// - `chinese2` with G2PW from `g2pw_model_dir`, the extracted
  ///   `G2PWModel_1.1` directory (upstream `GPT_SoVITS/text/G2PWModel`);
  /// - fast-langdetect's default detector with the model file
  ///   `fast_langdetect_model` (`lid.176.bin`, upstream
  ///   `GPT_SoVITS/pretrained_models/fast_langdetect/lid.176.bin`). This
  ///   replaces the process-wide default detector, as `langsegmenter.py`
  ///   does when it is imported.
  ///
  /// The English and Japanese modules build their state on first use.
  pub fn from_paths(
    bert_model: M,
    g2pw_model_dir: impl AsRef<Path>,
    fast_langdetect_model: impl AsRef<Path>,
  ) -> Result<Self, InitError> {
    let defaults = LangDetectConfig::default();
    let config = LangDetectConfig::new(
      None,
      Some(fast_langdetect_model.as_ref().to_path_buf()),
      defaults.normalize_input,
      defaults.max_input_length,
      Model::Auto,
    )
    .map_err(InitError::FastLangdetect)?;
    let tokenizer =
      tokenizer_from_json(TOKENIZER_JSON).map_err(|e| InitError::Tokenizer(e.to_string()))?;
    let cleaner = Cleaner::new(g2pw_model_dir).map_err(InitError::Chinese2)?;
    fast_langdetect::set_default_detector(LangDetector::new(config));
    Ok(Self::new(bert_model, tokenizer, cleaner))
  }

  /// The embedded `chinese-roberta-wwm-ext-large` tokenizer, as
  /// `AutoTokenizer.from_pretrained(bert_path)` loads it.
  pub fn default_tokenizer() -> Tokenizer {
    tokenizer_from_json(TOKENIZER_JSON).expect("the embedded tokenizer.json loads")
  }

  /// Port of `preprocess(text, lang, text_split_method, version="v2")`:
  /// [`Self::replace_consecutive_punctuation`], [`Self::pre_seg_text`],
  /// then [`Self::segment_and_extract_feature_for_text`] for each segment,
  /// skipping segments whose `norm_text` is empty.
  pub fn preprocess(
    &self,
    text: &str,
    lang: &str,
    text_split_method: &str,
    version: &str,
  ) -> Result<Vec<PreprocessResult>, Error> {
    let text = self.replace_consecutive_punctuation(text);
    let texts = self.pre_seg_text(&text, lang, text_split_method)?;
    let mut result = Vec::new();
    for text in &texts {
      let (phones, bert_features, norm_text) =
        self.segment_and_extract_feature_for_text(text, lang, version)?;
      if norm_text.is_empty() {
        continue;
      }
      result.push(PreprocessResult {
        phones,
        bert_features,
        norm_text,
      });
    }
    Ok(result)
  }

  /// Port of `pre_seg_text(text, lang, text_split_method)`.
  pub fn pre_seg_text(
    &self,
    text: &str,
    lang: &str,
    text_split_method: &str,
  ) -> Result<Vec<String>, Error> {
    let text = text.trim_matches('\n');
    if text.is_empty() {
      return Ok(Vec::new());
    }
    let first = text.chars().next().expect("text is not empty");
    let mut text = text.to_string();
    if !SPLITS.contains(&first) && get_first(&text).chars().count() < 4 {
      text = if lang != "en" {
        format!("。{text}")
      } else {
        format!(".{text}")
      };
    }

    let seg_method = get_seg_method(text_split_method)?;
    let mut text = seg_method(&text)?;

    while text.contains("\n\n") {
      text = text.replace("\n\n", "\n");
    }

    let _texts: Vec<String> = text.split('\n').map(str::to_string).collect();
    let _texts = self.filter_text(_texts)?;
    let _texts = merge_short_text_in_array(_texts, 5);
    let mut texts = Vec::new();

    for mut text in _texts {
      // 解决输入目标文本的空行导致报错的问题
      if py_strip(&text).is_empty() {
        continue;
      }
      if has_no_word_char(&text) {
        // 检测一下，如果是纯符号，就跳过。
        continue;
      }
      let last = text.chars().next_back().expect("text is not empty");
      if !SPLITS.contains(&last) {
        text.push(if lang != "en" { '。' } else { '.' });
      }

      // 解决句子过长导致Bert报错的问题
      if text.chars().count() > 510 {
        texts.extend(split_big_text(&text, SplitBigTextOptions::default()));
      } else {
        texts.push(text);
      }
    }
    Ok(texts)
  }

  /// Port of `segment_and_extract_feature_for_text(text, language,
  /// version="v1")`: [`Self::get_phones_and_bert`].
  pub fn segment_and_extract_feature_for_text(
    &self,
    text: &str,
    language: &str,
    version: &str,
  ) -> Result<(Vec<usize>, Array2<f32>, String), Error> {
    self.get_phones_and_bert(text, language, version, false)
  }

  /// Port of `get_phones_and_bert(text, language, version, final=False)`.
  ///
  /// Splits the text by language according to the mode (`all_zh`,
  /// `all_yue`, `all_ja`, `all_ko`, `en`, `auto`, `auto_yue`, or any other
  /// value such as `zh`, `yue`, `ja`, `ko`, which keeps English segments and
  /// labels the rest with the mode), runs [`Self::clean_text_inf`] and
  /// [`Self::get_bert_inf`] on each segment, and concatenates the phone ids,
  /// the BERT features (along the phone axis) and the normalized texts.
  /// With fewer than 6 phones and `final` unset, it runs again on
  /// `"." + text` with `final` set. Returns `(phones, bert, norm_text)`.
  pub fn get_phones_and_bert(
    &self,
    text: &str,
    language: &str,
    version: &str,
    r#final: bool,
  ) -> Result<(Vec<usize>, Array2<f32>, String), Error> {
    let text = collapse_spaces(text);
    let mut textlist: Vec<String> = Vec::new();
    let mut langlist: Vec<String> = Vec::new();
    match language {
      "all_zh" | "all_ja" | "all_ko" => {
        let default_lang = &language["all_".len()..];
        for tmp in LangSegmenter::getTexts(&text, default_lang)? {
          langlist.push(tmp.lang);
          textlist.push(tmp.text);
        }
      }
      "all_yue" => {
        for mut tmp in LangSegmenter::getTexts(&text, "zh")? {
          if tmp.lang == "zh" {
            tmp.lang = "yue".to_string();
          }
          langlist.push(tmp.lang);
          textlist.push(tmp.text);
        }
      }
      "en" => {
        langlist.push("en".to_string());
        textlist.push(text.clone());
      }
      "auto" => {
        for tmp in LangSegmenter::getTexts(&text, "")? {
          langlist.push(tmp.lang);
          textlist.push(tmp.text);
        }
      }
      "auto_yue" => {
        for mut tmp in LangSegmenter::getTexts(&text, "")? {
          if tmp.lang == "zh" {
            tmp.lang = "yue".to_string();
          }
          langlist.push(tmp.lang);
          textlist.push(tmp.text);
        }
      }
      _ => {
        for tmp in LangSegmenter::getTexts(&text, "")? {
          if let Some(last) = langlist.last()
            && ((tmp.lang == "en" && last == "en") || (tmp.lang != "en" && last != "en"))
          {
            textlist
              .last_mut()
              .expect("textlist and langlist have the same length")
              .push_str(&tmp.text);
            continue;
          }
          if tmp.lang == "en" {
            langlist.push(tmp.lang);
          } else {
            // 因无法区别中日韩文汉字,以用户输入为准
            langlist.push(language.to_string());
          }
          textlist.push(tmp.text);
        }
      }
    }

    let mut phones_list: Vec<Vec<usize>> = Vec::new();
    let mut bert_list: Vec<Array2<f32>> = Vec::new();
    let mut norm_text_list: Vec<String> = Vec::new();
    for (segment, lang) in textlist.iter().zip(&langlist) {
      let (phones, word2ph, norm_text) = self.clean_text_inf(segment, lang, version)?;
      let bert = self.get_bert_inf(&phones, word2ph.as_deref(), &norm_text, lang)?;
      phones_list.push(phones);
      norm_text_list.push(norm_text);
      bert_list.push(bert);
    }
    let bert = torch_cat(&bert_list, Axis(1))?;
    let phones: Vec<usize> = phones_list.concat();
    let norm_text: String = norm_text_list.concat();

    if !r#final && phones.len() < 6 {
      return self.get_phones_and_bert(&format!(".{text}"), language, version, true);
    }

    Ok((phones, bert, norm_text))
  }

  /// Port of `get_bert_feature(text, word2ph)`: runs the BERT model on
  /// `text`, drops the `[CLS]` and `[SEP]` rows (`[1:-1]`), repeats row `i`
  /// `word2ph[i]` times, and returns the transpose (`[hidden, sum(word2ph)]`).
  pub fn get_bert_feature(
    &self,
    text: &str,
    word2ph: Option<&[usize]>,
  ) -> Result<Array2<f32>, Error> {
    let encoding = self
      .tokenizer
      .encode(text, true)
      .map_err(|e| Error::Tokenizer(e.to_string()))?;
    let to_i64 = |v: &[u32]| v.iter().map(|&x| i64::from(x)).collect::<Vec<_>>();
    let inputs = BertInputs {
      input_ids: to_i64(encoding.get_ids()),
      token_type_ids: to_i64(encoding.get_type_ids()),
      attention_mask: to_i64(encoding.get_attention_mask()),
    };
    let hidden = self
      .bert_model
      .hidden_states(&inputs)
      .map_err(Error::Bert)?;
    // [1:-1]
    let n = hidden.nrows();
    let res = if n >= 2 {
      hidden.slice(s![1..n - 1, ..]).to_owned()
    } else {
      hidden.slice(s![0..0, ..]).to_owned()
    };
    let Some(word2ph) = word2ph else {
      return Err(Error::TypeError(
        "object of type 'NoneType' has no len()".to_string(),
      ));
    };
    if word2ph.len() != text.chars().count() {
      return Err(Error::AssertionError("len(word2ph) == len(text)"));
    }
    let mut phone_level_feature: Vec<Array2<f32>> = Vec::with_capacity(word2ph.len());
    for (i, &w) in word2ph.iter().enumerate() {
      if i >= res.nrows() {
        return Err(Error::IndexError(format!(
          "index {i} is out of bounds for dimension 0 with size {}",
          res.nrows()
        )));
      }
      // res[i].repeat(word2ph[i], 1)
      let row = res.row(i);
      let repeat_feature = row
        .broadcast((w, row.len()))
        .expect("a row broadcasts to (w, hidden)")
        .to_owned();
      phone_level_feature.push(repeat_feature);
    }
    let phone_level_feature = torch_cat(&phone_level_feature, Axis(0))?;
    Ok(
      phone_level_feature
        .reversed_axes()
        .as_standard_layout()
        .to_owned(),
    )
  }

  /// Port of `clean_text_inf(text, language, version="v2")`: drops `all_`
  /// from the language, runs `clean_text`, and maps the phones to ids with
  /// `cleaned_text_to_sequence`. Returns `(phones, word2ph, norm_text)`.
  #[allow(clippy::type_complexity)]
  pub fn clean_text_inf(
    &self,
    text: &str,
    language: &str,
    version: &str,
  ) -> Result<(Vec<usize>, Option<Vec<usize>>, String), Error> {
    let language = language.replace("all_", "");
    let (phones, word2ph, norm_text) = self.cleaner.clean_text(text, &language, Some(version))?;
    let phones = cleaned_text_to_sequence(&phones, Some(version))?;
    Ok((phones, word2ph, norm_text))
  }

  /// Port of `get_bert_inf(phones, word2ph, norm_text, language)`:
  /// [`Self::get_bert_feature`] for `zh`, zeros of shape
  /// `[1024, len(phones)]` for every other language.
  pub fn get_bert_inf(
    &self,
    phones: &[usize],
    word2ph: Option<&[usize]>,
    norm_text: &str,
    language: &str,
  ) -> Result<Array2<f32>, Error> {
    let language = language.replace("all_", "");
    if language == "zh" {
      self.get_bert_feature(norm_text, word2ph)
    } else {
      Ok(Array2::zeros((1024, phones.len())))
    }
  }

  /// Port of `filter_text(texts)`: drops `" "` and `""`; raises
  /// `ValueError` when every text is `" "`, `"\n"` or `""`.
  pub fn filter_text(&self, texts: Vec<String>) -> Result<Vec<String>, Error> {
    if texts.iter().all(|t| matches!(t.as_str(), " " | "\n" | "")) {
      return Err(Error::ValueError("请输入有效文本".to_string()));
    }
    Ok(
      texts
        .into_iter()
        .filter(|t| !matches!(t.as_str(), " " | ""))
        .collect(),
    )
  }

  /// Port of `replace_consecutive_punctuation(text)`: a run of two or more
  /// characters of [`PUNCTUATION`] becomes its first character.
  pub fn replace_consecutive_punctuation(&self, text: &str) -> String {
    replace_consecutive_punctuation(text)
  }
}

/// [`TextPreprocessor::replace_consecutive_punctuation`] as a free function
/// (it does not read `self`).
pub fn replace_consecutive_punctuation(text: &str) -> String {
  let mut result = String::with_capacity(text.len());
  let mut prev_punct = false;
  for c in text.chars() {
    let punct = PUNCTUATION.contains(&c);
    if !(punct && prev_punct) {
      result.push(c);
    }
    prev_punct = punct;
  }
  result
}

/// `torch.cat(tensors, dim)` for 2-D tensors: `ValueError` for an empty
/// list (torch 2.x), `RuntimeError` for mismatched sizes.
fn torch_cat(tensors: &[Array2<f32>], axis: Axis) -> Result<Array2<f32>, Error> {
  if tensors.is_empty() {
    return Err(Error::ValueError(
      "torch.cat(): expected a non-empty list of Tensors".to_string(),
    ));
  }
  let views: Vec<ArrayView2<f32>> = tensors.iter().map(|t| t.view()).collect();
  concatenate(axis, &views).map_err(|e| Error::RuntimeError(e.to_string()))
}

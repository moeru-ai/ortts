//! Port of `GPT_SoVITS/text/cleaner.py` (GPT-SoVITS @ 04a44eb).
//!
//! # Module state
//!
//! `cleaner.py` imports a language module by name
//! (`__import__("text." + language_module_map[language])`). The language
//! modules of the port keep their state as follows, and [`Cleaner`] holds
//! the one that needs a constructor argument:
//!
//! | `language_module_map` value | Rust |
//! |---|---|
//! | `chinese2` (v2 `zh`) | [`Cleaner::chinese2`] ([`Chinese2`], built with the G2PW model dir) |
//! | `english` (`en`) | [`crate::english`] (module-level `_g2p`, built on first use) |
//! | `japanese` (`ja`) | [`crate::japanese`] (global `OpenJTalk`, built on first use) |
//! | `cantonese` (v2 `yue`) | [`crate::cantonese`] (no state) |
//! | `chinese` (v1 `zh`) | not ported: [`Error::NotPorted`] |
//! | `korean` (v2 `ko`) | not ported: [`Error::NotPorted`] |
//!
//! # Ported scope
//!
//! `special`, `clean_text`, `clean_special` and `text_to_sequence`. Left out:
//! the `__main__` demo.
//!
//! # Differences in form (not in output)
//!
//! - Where Python raises, the methods return an [`Error`] that names the
//!   Python exception ([`Error::name`]).
//! - `text/chinese.py` (the v1 Chinese frontend) and `text/korean.py` are not
//!   ported. Where Python would import one of them, the methods return
//!   [`Error::NotPorted`] instead of a result. They never fall back to
//!   another module. Every other language and version combination is handled
//!   as in Python, including the fallback of an unknown language to
//!   `("en", " ")` (so v1 `yue` and v1 `ko` become `en` as upstream).
//! - `clean_special` reads `phones[0]` and `phones[1]` of `g2p`, which is
//!   `(phones, word2ph)` for `zh` and `yue`. Upstream only calls it for
//!   `zh` (the `special` list has no other language). For `en` and `ja`,
//!   whose `g2p` returns a flat phone list, the Python code would iterate the
//!   characters of the first phone; the port returns
//!   [`Error::NotPorted`] for these two instead.

use std::fmt;
use std::path::Path;

use crate::chinese2::{self, Chinese2};
use crate::{cantonese, en_normalization, english, japanese, symbol_to_id_v1, symbol_to_id_v2};

/// Port of `special`: `(special_s, special_l, target_symbol)` triples. The
/// commented-out `%` and `@` entries upstream are not part of the list.
pub const SPECIAL: &[(&str, &str, &str)] = &[("￥", "zh", "SP2"), ("^", "zh", "SP3")];

/// An exception raised by `cleaner.py` or by the language module it calls.
#[derive(Debug)]
pub enum Error {
  /// An exception of `chinese2.text_normalize` or `chinese2.g2p`.
  Chinese2(chinese2::Error),
  /// An exception of `english.text_normalize`.
  EnNormalization(en_normalization::Error),
  /// An exception of `english.g2p`.
  English(english::Error),
  /// An exception of `japanese.g2p`. The message is the Python exception
  /// name (`AssertionError`, `IndexError`) when the port raises one, and
  /// a `jpreprocess` error otherwise.
  Japanese(anyhow::Error),
  /// An exception of `cantonese.text_normalize` or `cantonese.g2p`.
  Cantonese(cantonese::Error),
  /// `AssertionError` from `assert len(phones) == sum(word2ph)`,
  /// `assert len(norm_text) == len(word2ph)` or `assert ph in symbols`.
  /// Holds the failed condition (Python's message is empty).
  AssertionError(&'static str),
  /// `KeyError(language)` from `language_module_map[language]` in
  /// `clean_special`.
  KeyError(String),
  /// `TypeError(message)`: `text_to_sequence` calls `clean_text(text)`
  /// without `language`.
  TypeError(String),
  /// The language module that Python would import is not ported
  /// (`chinese` for v1 `zh`, `korean` for `ko`), or `clean_special` is
  /// called for a language whose `g2p` result it cannot index (see the
  /// module docs).
  NotPorted {
    /// The module name from `language_module_map`.
    module: &'static str,
    /// The language code.
    language: String,
    /// The resolved version.
    version: String,
  },
}

impl Error {
  /// The Python exception class name (`type(e).__name__`). [`Error::NotPorted`]
  /// gives `"NotPortedError"`, the name the parity generators use for the
  /// same case; a `jpreprocess` failure gives `"<port error>"`.
  pub fn name(&self) -> &'static str {
    match self {
      Error::Chinese2(e) => e.name().unwrap_or("<port error>"),
      Error::EnNormalization(e) => e.name(),
      Error::English(e) => e.name(),
      Error::Japanese(e) => match e.to_string().as_str() {
        "AssertionError" => "AssertionError",
        "IndexError" => "IndexError",
        _ => "<port error>",
      },
      Error::Cantonese(e) => e.name(),
      Error::AssertionError(_) => "AssertionError",
      Error::KeyError(_) => "KeyError",
      Error::TypeError(_) => "TypeError",
      Error::NotPorted { .. } => "NotPortedError",
    }
  }
}

impl fmt::Display for Error {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      Error::Chinese2(e) => write!(f, "{e}"),
      Error::EnNormalization(e) => write!(f, "{}: {}", e.name(), e.message()),
      Error::English(e) => write!(f, "{e}"),
      Error::Japanese(e) => write!(f, "{e}"),
      Error::Cantonese(e) => write!(f, "{e}"),
      Error::AssertionError(cond) => write!(f, "AssertionError: {cond}"),
      Error::KeyError(k) => write!(f, "KeyError: {k:?}"),
      Error::TypeError(m) => write!(f, "TypeError: {m}"),
      Error::NotPorted {
        module,
        language,
        version,
      } => write!(
        f,
        "NotPortedError: text.{module} (language {language:?}, version {version:?}) is not ported"
      ),
    }
  }
}

impl std::error::Error for Error {}

impl From<chinese2::Error> for Error {
  fn from(e: chinese2::Error) -> Self {
    Error::Chinese2(e)
  }
}

impl From<en_normalization::Error> for Error {
  fn from(e: en_normalization::Error) -> Self {
    Error::EnNormalization(e)
  }
}

impl From<english::Error> for Error {
  fn from(e: english::Error) -> Self {
    Error::English(e)
  }
}

impl From<cantonese::Error> for Error {
  fn from(e: cantonese::Error) -> Self {
    Error::Cantonese(e)
  }
}

/// The modules that `language_module_map` names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LanguageModule {
  Chinese,
  Chinese2,
  Japanese,
  English,
  Korean,
  Cantonese,
}

impl LanguageModule {
  fn name(self) -> &'static str {
    match self {
      LanguageModule::Chinese => "chinese",
      LanguageModule::Chinese2 => "chinese2",
      LanguageModule::Japanese => "japanese",
      LanguageModule::English => "english",
      LanguageModule::Korean => "korean",
      LanguageModule::Cantonese => "cantonese",
    }
  }
}

/// `version = os.environ.get("version", "v2")` when `version is None`.
fn resolve_version(version: Option<&str>) -> String {
  match version {
    Some(v) => v.to_string(),
    None => std::env::var("version").unwrap_or_else(|_| "v2".to_string()),
  }
}

/// `symbols` and `language_module_map[language]` for a version. `None` when
/// the language is not in the map.
fn language_module(version: &str, language: &str) -> Option<LanguageModule> {
  if version == "v1" {
    match language {
      "zh" => Some(LanguageModule::Chinese),
      "ja" => Some(LanguageModule::Japanese),
      "en" => Some(LanguageModule::English),
      _ => None,
    }
  } else {
    match language {
      "zh" => Some(LanguageModule::Chinese2),
      "ja" => Some(LanguageModule::Japanese),
      "en" => Some(LanguageModule::English),
      "ko" => Some(LanguageModule::Korean),
      "yue" => Some(LanguageModule::Cantonese),
      _ => None,
    }
  }
}

/// `ph in symbols` for `symbols_v1.symbols` or `symbols_v2.symbols`.
fn in_symbols(version: &str, ph: &str) -> bool {
  if version == "v1" {
    symbol_to_id_v1().contains_key(ph)
  } else {
    symbol_to_id_v2().contains_key(ph)
  }
}

/// The result of `clean_text` and `clean_special`: `(phones, word2ph,
/// norm_text)`. `word2ph` is `None` for `en` and `ja`.
pub type CleanedText = (Vec<String>, Option<Vec<usize>>, String);

/// `(norm_text, phones, word2ph)` from `text_normalize` and `g2p`.
type NormalizedG2p = (String, Vec<String>, Option<Vec<usize>>);

/// The language modules that `cleaner.py` imports (see the module docs).
///
/// Only `chinese2` needs a constructor argument (the G2PW model directory),
/// so it is the only field; `english`, `japanese` and `cantonese` are used
/// through their module-level functions.
#[derive(Debug)]
pub struct Cleaner {
  /// `text.chinese2` as GPT-SoVITS imports it.
  pub chinese2: Chinese2,
}

impl Cleaner {
  /// The modules as GPT-SoVITS imports them: `chinese2` with G2PW from
  /// `g2pw_model_dir` (the extracted `G2PWModel` directory, upstream
  /// `GPT_SoVITS/text/G2PWModel`).
  pub fn new(g2pw_model_dir: impl AsRef<Path>) -> Result<Self, chinese2::Error> {
    Ok(Self {
      chinese2: Chinese2::new(g2pw_model_dir)?,
    })
  }

  /// A cleaner with a given `chinese2` module (for example
  /// [`Chinese2::without_g2pw`]).
  pub fn from_chinese2(chinese2: Chinese2) -> Self {
    Self { chinese2 }
  }

  /// `language_module.text_normalize(text)` followed by
  /// `language_module.g2p(norm_text)` for the four ported modules, returning
  /// `(norm_text, phones, word2ph)`; `word2ph` is `Some` for the modules
  /// whose `g2p` returns `(phones, word2ph)`.
  fn normalize_and_g2p(
    &self,
    module: LanguageModule,
    text: &str,
    language: &str,
    version: &str,
  ) -> Result<NormalizedG2p, Error> {
    match module {
      LanguageModule::Chinese2 => {
        let norm_text = chinese2::text_normalize(text)?;
        let (phones, word2ph) = self.chinese2.g2p(&norm_text)?;
        Ok((norm_text, phones, Some(word2ph)))
      }
      LanguageModule::Cantonese => {
        let norm_text = cantonese::text_normalize(text)?;
        let (phones, word2ph) = cantonese::g2p(&norm_text)?;
        Ok((norm_text, phones, Some(word2ph)))
      }
      LanguageModule::English => {
        let norm_text = english::text_normalize(text)?;
        let phones = english::g2p(&norm_text)?;
        Ok((norm_text, phones, None))
      }
      LanguageModule::Japanese => {
        let norm_text = japanese::text_normalize(text);
        // `g2p(norm_text, with_prosody=True)`.
        let phones = japanese::g2p(&norm_text, true).map_err(Error::Japanese)?;
        Ok((norm_text, phones, None))
      }
      LanguageModule::Chinese | LanguageModule::Korean => Err(Error::NotPorted {
        module: module.name(),
        language: language.to_string(),
        version: version.to_string(),
      }),
    }
  }

  /// Port of `clean_text(text, language, version=None)`.
  ///
  /// `version = None` reads the `version` environment variable (default
  /// `"v2"`). `"v1"` selects `symbols_v1` and `{"zh": "chinese", "ja":
  /// "japanese", "en": "english"}`; any other value selects `symbols_v2`
  /// and `{"zh": "chinese2", "ja": "japanese", "en": "english", "ko":
  /// "korean", "yue": "cantonese"}`. A language that is not in the map
  /// becomes `"en"` with the text `" "`. Returns `(phones, word2ph,
  /// norm_text)`; phones that are not in `symbols` become `"UNK"`.
  pub fn clean_text(
    &self,
    text: &str,
    language: &str,
    version: Option<&str>,
  ) -> Result<CleanedText, Error> {
    let version = resolve_version(version);
    let (module, language, text) = match language_module(&version, language) {
      Some(module) => (module, language, text),
      None => (LanguageModule::English, "en", " "),
    };
    for &(special_s, special_l, target_symbol) in SPECIAL {
      if text.contains(special_s) && language == special_l {
        return self.clean_special(text, language, special_s, target_symbol, Some(&version));
      }
    }
    let (norm_text, phones, word2ph) = self.normalize_and_g2p(module, text, language, &version)?;
    let phones = match (language, word2ph) {
      ("zh" | "yue", Some(word2ph)) => {
        if phones.len() != word2ph.iter().sum::<usize>() {
          return Err(Error::AssertionError("len(phones) == sum(word2ph)"));
        }
        if norm_text.chars().count() != word2ph.len() {
          return Err(Error::AssertionError("len(norm_text) == len(word2ph)"));
        }
        return Ok((unk_phones(&version, phones), Some(word2ph), norm_text));
      }
      ("en", _) => {
        if phones.len() < 4 {
          let mut p = vec![",".to_string()];
          p.extend(phones);
          p
        } else {
          phones
        }
      }
      _ => phones,
    };
    Ok((unk_phones(&version, phones), None, norm_text))
  }

  /// Port of `clean_special(text, language, special_s, target_symbol,
  /// version=None)` (特殊静音段sp符号处理): replaces `special_s` with `","`,
  /// normalizes, runs `g2p`, and turns every `","` phone into
  /// `target_symbol`. A phone that is not in `symbols` is an
  /// `AssertionError`.
  ///
  /// Only `zh` and `yue` are supported (see the module docs); `en` and
  /// `ja` give [`Error::NotPorted`].
  pub fn clean_special(
    &self,
    text: &str,
    language: &str,
    special_s: &str,
    target_symbol: &str,
    version: Option<&str>,
  ) -> Result<CleanedText, Error> {
    let version = resolve_version(version);
    let text = text.replace(special_s, ",");
    let Some(module) = language_module(&version, language) else {
      return Err(Error::KeyError(language.to_string()));
    };
    if matches!(module, LanguageModule::English | LanguageModule::Japanese) {
      return Err(Error::NotPorted {
        module: module.name(),
        language: language.to_string(),
        version,
      });
    }
    let (norm_text, phones, word2ph) = self.normalize_and_g2p(module, &text, language, &version)?;
    let mut new_ph = Vec::with_capacity(phones.len());
    for ph in phones {
      if !in_symbols(&version, &ph) {
        return Err(Error::AssertionError("ph in symbols"));
      }
      if ph == "," {
        new_ph.push(target_symbol.to_string());
      } else {
        new_ph.push(ph);
      }
    }
    Ok((new_ph, word2ph, norm_text))
  }

  /// Port of `text_to_sequence(text, language, version=None)`.
  ///
  /// Upstream calls `clean_text(text)` without the `language` argument, so
  /// the function always raises `TypeError` before it reaches
  /// `cleaned_text_to_sequence`. The port returns that error. (It resolves
  /// `version` first, as Python does, but the value is never used.)
  pub fn text_to_sequence(
    &self,
    _text: &str,
    _language: &str,
    version: Option<&str>,
  ) -> Result<Vec<usize>, Error> {
    // version = os.environ.get("version", version); default "v2".
    let _version = std::env::var("version")
      .ok()
      .or_else(|| version.map(str::to_string))
      .unwrap_or_else(|| "v2".to_string());
    Err(Error::TypeError(
      "clean_text() missing 1 required positional argument: 'language'".to_string(),
    ))
  }
}

/// `["UNK" if ph not in symbols else ph for ph in phones]`.
fn unk_phones(version: &str, phones: Vec<String>) -> Vec<String> {
  phones
    .into_iter()
    .map(|ph| {
      if in_symbols(version, &ph) {
        ph
      } else {
        "UNK".to_string()
      }
    })
    .collect()
}

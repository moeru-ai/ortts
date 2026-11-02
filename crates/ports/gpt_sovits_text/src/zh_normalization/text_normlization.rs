//! Port of `GPT_SoVITS/text/zh_normalization/text_normlization.py` (GPT-SoVITS @ 04a44eb).

use std::sync::LazyLock;

use super::Error;
use super::char_convert::tranditional_to_simplified;
use super::chronology::{
  RE_DATE, RE_DATE2, RE_TIME, RE_TIME_RANGE, replace_date, replace_date2, replace_time,
};
use super::constants::{F2H_ASCII_LETTERS, F2H_DIGITS, F2H_SPACE, translate, translate_str_keyed};
use super::num::{
  RE_ASMD, RE_DECIMAL_NUM, RE_DEFAULT_NUM, RE_FRAC, RE_INTEGER, RE_NUMBER, RE_PERCENTAGE,
  RE_POSITIVE_QUANTIFIERS, RE_POWER, RE_RANGE, RE_TO_RANGE, RE_VERSION_NUM, replace_asmd,
  replace_default_num, replace_frac, replace_negative_num, replace_number, replace_percentage,
  replace_positive_quantifier, replace_power, replace_range, replace_to_range, replace_vrsion_num,
};
use super::phonecode::{
  RE_MOBILE_PHONE, RE_NATIONAL_UNIFORM_NUMBER, RE_TELEPHONE, replace_mobile, replace_phone,
};
use super::pyre::{PyRegex, py_strip};
use super::quantifier::{RE_TEMPERATURE, replace_measure, replace_temperature};

/// `re.sub(r"[——《》【】<>{}()（）#&@“”^_|\\]", "", text)` in `_split`.
static RE_SPLIT_FILTER: LazyLock<PyRegex> =
  LazyLock::new(|| PyRegex::compile(r"[——《》【】<>{}()（）#&@“”^_|\\]"));
/// `re.split(r"\n+", text)` in `_split`.
static RE_NEWLINES: LazyLock<PyRegex> = LazyLock::new(|| PyRegex::compile(r"\n+"));
/// `re.sub(r"[-——《》【】<=>{}()（）#&@“”^_|\\]", "", sentence)` in `_post_replace`.
static RE_POST_FILTER: LazyLock<PyRegex> =
  LazyLock::new(|| PyRegex::compile(r"[-——《》【】<=>{}()（）#&@“”^_|\\]"));

/// Port of `TextNormalizer`.
#[allow(non_snake_case)]
pub struct TextNormalizer {
  /// Port of `self.SENTENCE_SPLITOR`.
  pub SENTENCE_SPLITOR: PyRegex,
}

impl Default for TextNormalizer {
  fn default() -> Self {
    Self::new()
  }
}

impl TextNormalizer {
  /// Port of `TextNormalizer.__init__`.
  pub fn new() -> Self {
    Self {
      SENTENCE_SPLITOR: PyRegex::compile(r"([：、，；。？！,;?!][”’]?)"),
    }
  }

  /// Port of `TextNormalizer._split`: split long text into sentences with
  /// sentence-splitting punctuations.
  pub fn _split(&self, text: &str, lang: &str) -> Vec<String> {
    let mut text = text.to_string();
    // Only for pure Chinese here
    if lang == "zh" {
      text = text.replace(' ', "");
      // 过滤掉特殊字符
      text = RE_SPLIT_FILTER.sub_infallible(|_| String::new(), &text);
    }
    text = self
      .SENTENCE_SPLITOR
      .sub_infallible(|m| format!("{}\n", m.group(1).unwrap_or_default()), &text);
    let text = py_strip(&text);
    RE_NEWLINES
      .split(text)
      .into_iter()
      .map(|sentence| py_strip(sentence).to_string())
      .collect()
  }

  /// Port of `TextNormalizer._post_replace`.
  pub fn _post_replace(&self, sentence: &str) -> String {
    let mut sentence = sentence.replace('/', "每");
    // sentence = sentence.replace('~', '至')
    // sentence = sentence.replace('～', '至')
    const REPLACEMENTS: &[(&str, &str)] = &[
      ("①", "一"),
      ("②", "二"),
      ("③", "三"),
      ("④", "四"),
      ("⑤", "五"),
      ("⑥", "六"),
      ("⑦", "七"),
      ("⑧", "八"),
      ("⑨", "九"),
      ("⑩", "十"),
      ("α", "阿尔法"),
      ("β", "贝塔"),
      ("γ", "伽玛"),
      ("Γ", "伽玛"),
      ("δ", "德尔塔"),
      ("Δ", "德尔塔"),
      ("ε", "艾普西龙"),
      ("ζ", "捷塔"),
      ("η", "依塔"),
      ("θ", "西塔"),
      ("Θ", "西塔"),
      ("ι", "艾欧塔"),
      ("κ", "喀帕"),
      ("λ", "拉姆达"),
      ("Λ", "拉姆达"),
      ("μ", "缪"),
      ("ν", "拗"),
      ("ξ", "克西"),
      ("Ξ", "克西"),
      ("ο", "欧米克伦"),
      ("π", "派"),
      ("Π", "派"),
      ("ρ", "肉"),
      ("ς", "西格玛"),
      ("Σ", "西格玛"),
      ("σ", "西格玛"),
      ("τ", "套"),
      ("υ", "宇普西龙"),
      ("φ", "服艾"),
      ("Φ", "服艾"),
      ("χ", "器"),
      ("ψ", "普赛"),
      ("Ψ", "普赛"),
      ("ω", "欧米伽"),
      ("Ω", "欧米伽"),
      // 兜底数学运算，顺便兼容懒人用语
      ("+", "加"),
      ("-", "减"),
      ("×", "乘"),
      ("÷", "除"),
      ("=", "等"),
    ];
    for (from, to) in REPLACEMENTS {
      sentence = sentence.replace(from, to);
    }
    // re filter special characters, have one more character "-" than line 68
    RE_POST_FILTER.sub_infallible(|_| String::new(), &sentence)
  }

  /// Port of `TextNormalizer.normalize_sentence`.
  pub fn normalize_sentence(&self, sentence: &str) -> Result<String, Error> {
    // basic character conversions
    let sentence = tranditional_to_simplified(sentence);
    let sentence = translate(&sentence, &F2H_ASCII_LETTERS);
    let sentence = translate(&sentence, &F2H_DIGITS);
    let sentence = translate_str_keyed(&sentence, &F2H_SPACE);

    // number related NSW verbalization
    let sentence = RE_DATE.sub(replace_date, &sentence)?;
    let sentence = RE_DATE2.sub(replace_date2, &sentence)?;

    // range first
    let sentence = RE_TIME_RANGE.sub(replace_time, &sentence)?;
    let sentence = RE_TIME.sub(replace_time, &sentence)?;

    // 处理~波浪号作为至的替换
    let sentence = RE_TO_RANGE.sub(replace_to_range, &sentence)?;
    let sentence = RE_TEMPERATURE.sub(replace_temperature, &sentence)?;
    let mut sentence = replace_measure(&sentence);

    // 处理数学运算
    while RE_ASMD.search(&sentence) {
      sentence = RE_ASMD.sub(replace_asmd, &sentence)?;
    }
    let sentence = RE_POWER.sub(replace_power, &sentence)?;

    let sentence = RE_FRAC.sub(replace_frac, &sentence)?;
    let sentence = RE_PERCENTAGE.sub(replace_percentage, &sentence)?;
    let sentence = RE_MOBILE_PHONE.sub(replace_mobile, &sentence)?;

    let sentence = RE_TELEPHONE.sub(replace_phone, &sentence)?;
    let sentence = RE_NATIONAL_UNIFORM_NUMBER.sub(replace_phone, &sentence)?;

    let sentence = RE_RANGE.sub(replace_range, &sentence)?;

    let sentence = RE_INTEGER.sub(replace_negative_num, &sentence)?;
    let sentence = RE_VERSION_NUM.sub(replace_vrsion_num, &sentence)?;
    let sentence = RE_DECIMAL_NUM.sub(replace_number, &sentence)?;
    let sentence = RE_POSITIVE_QUANTIFIERS.sub(replace_positive_quantifier, &sentence)?;
    let sentence = RE_DEFAULT_NUM.sub(replace_default_num, &sentence)?;
    let sentence = RE_NUMBER.sub(replace_number, &sentence)?;
    let sentence = self._post_replace(&sentence);

    Ok(sentence)
  }

  /// Port of `TextNormalizer.normalize`.
  pub fn normalize(&self, text: &str) -> Result<Vec<String>, Error> {
    let sentences = self._split(text, "zh");
    sentences
      .iter()
      .map(|sent| self.normalize_sentence(sent))
      .collect()
  }
}

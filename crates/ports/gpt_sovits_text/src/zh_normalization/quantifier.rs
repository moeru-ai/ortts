//! Port of `GPT_SoVITS/text/zh_normalization/quantifier.py` (GPT-SoVITS @ 04a44eb).

use std::sync::LazyLock;

use super::Error;
use super::num::num2str;
use super::pyre::{PyMatch, PyRegex};

/// Port of `RE_TEMPERATURE` (温度表达式，温度会影响负号的读法; -3°C 零下三度).
pub static RE_TEMPERATURE: LazyLock<PyRegex> =
  LazyLock::new(|| PyRegex::compile(r"(-?)(\d+(\.\d+)?)(°C|℃|度|摄氏度)"));

/// Port of `measure_dict` (insertion order matters: `replace_measure` walks it in order).
pub const MEASURE_DICT: [(&str, &str); 17] = [
  ("cm2", "平方厘米"),
  ("cm²", "平方厘米"),
  ("cm3", "立方厘米"),
  ("cm³", "立方厘米"),
  ("cm", "厘米"),
  ("db", "分贝"),
  ("ds", "毫秒"),
  ("kg", "千克"),
  ("km", "千米"),
  ("m2", "平方米"),
  ("m²", "平方米"),
  ("m³", "立方米"),
  ("m3", "立方米"),
  ("ml", "毫升"),
  ("m", "米"),
  ("mm", "毫米"),
  ("s", "秒"),
];

/// Port of `replace_temperature`.
pub fn replace_temperature(m: &PyMatch) -> Result<String, Error> {
  let sign = m.group(1);
  let temperature = m.group(2).unwrap_or_default();
  // Upstream reads group 3, the optional decimal part, not the unit (group 4),
  // so the unit is always verbalized as "度".
  let unit = m.group(3);
  let sign = if sign.is_some_and(|s| !s.is_empty()) {
    "零下"
  } else {
    ""
  };
  let temperature = num2str(temperature)?;
  let unit = if unit == Some("摄氏度") {
    "摄氏度"
  } else {
    "度"
  };
  Ok(format!("{sign}{temperature}{unit}"))
}

/// Port of `replace_measure`.
pub fn replace_measure(sentence: &str) -> String {
  let mut sentence = sentence.to_string();
  for (q_notation, verbalized) in MEASURE_DICT {
    if sentence.contains(q_notation) {
      sentence = sentence.replace(q_notation, verbalized);
    }
  }
  sentence
}

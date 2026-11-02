//! Port of `proces/preprocess.py` (`traditional_to_simplified`,
//! `full_angle_to_half_angle` and the `preprocess` driver for them).

use std::collections::HashMap;
use std::sync::OnceLock;

use crate::{Error, Result};

/// `proces.conf.T2S_DICT`, exported to `data/t2s.tsv` by
/// `parity/export_data.py`.
const T2S_TSV: &str = include_str!("../../data/t2s.tsv");

fn t2s_dict() -> &'static HashMap<char, char> {
  static T2S: OnceLock<HashMap<char, char>> = OnceLock::new();
  T2S.get_or_init(|| {
    T2S_TSV
      .lines()
      .filter(|line| !line.is_empty())
      .map(|line| {
        let mut chars = line.chars();
        let k = chars.next().expect("t2s key");
        assert_eq!(chars.next(), Some('\t'), "t2s.tsv format");
        let v = chars.next().expect("t2s value");
        assert_eq!(chars.next(), None, "t2s.tsv format");
        (k, v)
      })
      .collect()
  })
}

/// Port of `proces.preprocess.traditional_to_simplified`.
pub fn traditional_to_simplified(text: &str) -> String {
  let dict = t2s_dict();
  text.chars().map(|t| *dict.get(&t).unwrap_or(&t)).collect()
}

/// Port of `proces.preprocess.full_angle_to_half_angle`.
pub fn full_angle_to_half_angle(text: &str) -> String {
  text
    .chars()
    .map(|uchar| {
      let mut inside_code = uchar as u32;
      if inside_code == 12288 {
        inside_code = 32;
      } else if (65281..=65374).contains(&inside_code) {
        inside_code -= 65248;
      }
      char::from_u32(inside_code).expect("half angle char")
    })
    .collect()
}

/// Port of `proces.preprocess.preprocess` for a single `str` input with
/// `params=None`.
///
/// Only the `traditional_to_simplified` and `full_angle_to_half_angle`
/// pipelines are ported. The other names in `ALL_PIPELINES` return
/// [`Error::NotPorted`]; unknown names return [`Error::ValueError`] as in
/// Python.
pub fn preprocess(data: &str, pipelines: &[&str]) -> Result<String> {
  let mut text = data.to_owned();
  for func in pipelines {
    text = match *func {
      "traditional_to_simplified" => traditional_to_simplified(&text),
      "full_angle_to_half_angle" => full_angle_to_half_angle(&text),
      "filter_unusual_characters"
      | "handle_blank_character"
      | "uppercase_to_lowercase"
      | "handle_substitute" => return Err(Error::NotPorted(format!("pipeline: {func}"))),
      _ => return Err(Error::ValueError(format!("pipeline: {func} not support!"))),
    };
  }
  Ok(text)
}

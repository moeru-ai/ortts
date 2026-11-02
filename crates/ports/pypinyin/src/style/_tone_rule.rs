//! Port of `pypinyin/style/_tone_rule.py`: where the tone mark goes.

/// Python `str.index(sub)` as a code point index.
fn py_index(hay: &str, needle: &str) -> Option<usize> {
  hay.find(needle).map(|b| hay[..b].chars().count())
}

/// Port of `pypinyin.style._tone_rule.right_mark_index`: the code point
/// index of the character that carries the tone mark, or `None`.
pub fn right_mark_index(pinyin_no_tone: &str) -> Option<usize> {
  // 'iou', 'uei', 'uen': 还原原始韵母后，还需要按照还原前的声调进行标记
  if pinyin_no_tone.contains("iou") {
    return py_index(pinyin_no_tone, "u");
  }
  if pinyin_no_tone.contains("uei") {
    return py_index(pinyin_no_tone, "i");
  }
  if pinyin_no_tone.contains("uen") {
    return py_index(pinyin_no_tone, "u");
  }

  // 有 ɑ 不放过, 没 ɑ 找 o、e
  for c in ["a", "o", "e"] {
    if let Some(i) = py_index(pinyin_no_tone, c) {
      return Some(i + c.chars().count() - 1);
    }
  }

  // i、u 若是连在一起，谁在后面就标谁
  for c in ["iu", "ui"] {
    if let Some(i) = py_index(pinyin_no_tone, c) {
      return Some(i + c.chars().count() - 1);
    }
  }

  // ɑ、o、e、i、u、ü
  for c in ["i", "u", "v", "ü"] {
    if let Some(i) = py_index(pinyin_no_tone, c) {
      return Some(i + c.chars().count() - 1);
    }
  }

  // n, m, ê
  for c in ["n", "m", "ê"] {
    if let Some(i) = py_index(pinyin_no_tone, c) {
      return Some(i + c.chars().count() - 1);
    }
  }
  None
}

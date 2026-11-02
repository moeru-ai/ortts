//! Port of `pypinyin/seg/mmseg.py`: forward maximum matching segmentation
//! (最大正向匹配分词) over the phrase dictionary.

use std::collections::HashSet;
use std::sync::LazyLock;

use crate::constants::PHRASES_DICT;

/// Port of `pypinyin.seg.mmseg.Seg` (正向最大匹配分词).
#[derive(Clone, Debug)]
pub struct Seg {
  /// `self._prefix_set`.
  pub _prefix_set: PrefixSet,
  /// 是否严格按照词语分词，不允许把非词语的词当做词语进行分词
  pub _no_non_phrases: bool,
}

impl Seg {
  /// Port of `Seg.__init__(prefix_set, no_non_phrases=False)`.
  pub fn new(prefix_set: PrefixSet, no_non_phrases: bool) -> Self {
    Self {
      _prefix_set: prefix_set,
      _no_non_phrases: no_non_phrases,
    }
  }

  /// Port of `Seg.cut` (a generator in Python): split `text` into words.
  pub fn cut(&self, text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let chars: Vec<char> = text.chars().collect();
    let mut remain: &[char] = &chars;
    let in_phrases = |w: &[char]| PHRASES_DICT.contains_key(&w.iter().collect::<String>());
    let to_string = |w: &[char]| w.iter().collect::<String>();

    while !remain.is_empty() {
      let mut matched: &[char] = &[];
      let mut last_valid_word: &[char] = &[];
      let mut last_valid_index = 0;
      let mut broke = false;

      // 一次加一个字的匹配
      for index in 0..remain.len() {
        let word = &remain[..index + 1];
        if self._prefix_set.contains(&to_string(word)) {
          matched = word;
          // 检查当前匹配的词是否为有效词语
          if !self._no_non_phrases || in_phrases(word) {
            last_valid_word = word;
            last_valid_index = index + 1;
          }
        } else {
          // 前缀匹配失败，需要处理之前的匹配结果
          if !last_valid_word.is_empty() {
            // 有有效词语，输出最后一个有效词语
            out.push(to_string(last_valid_word));
            remain = &remain[last_valid_index..];
          } else if self._no_non_phrases {
            // 严格模式：输出第一个字符
            out.push(remain[0].to_string());
            remain = &remain[1..];
          } else if !matched.is_empty() {
            // 非严格模式：输出匹配到的前缀（如果有）或第一个字符
            out.push(to_string(matched));
            remain = &remain[matched.len()..];
          } else {
            out.push(remain[0].to_string());
            remain = &remain[1..];
          }
          broke = true;
          break;
        }
      }
      if !broke {
        // 整个剩余文本都能匹配前缀
        if !last_valid_word.is_empty() {
          // 有有效词语，输出最后一个有效词语
          out.push(to_string(last_valid_word));
          remain = &remain[last_valid_index..];
        } else {
          // 没有有效词语，处理剩余文本
          if self._no_non_phrases && !in_phrases(remain) {
            // 严格模式且不在词典中：拆分为单字符
            out.extend(remain.iter().map(|c| c.to_string()));
          } else {
            // 非严格模式或在词典中：输出整个剩余文本
            out.push(to_string(remain));
          }
          break;
        }
      }
    }
    out
  }

  /// Port of `Seg.train`: 训练分词器.
  pub fn train<I, S>(&mut self, words: I)
  where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
  {
    self._prefix_set.train(words);
  }
}

/// Port of `pypinyin.seg.mmseg.PrefixSet`.
#[derive(Clone, Debug, Default)]
pub struct PrefixSet {
  /// `self._set`.
  pub _set: HashSet<String>,
}

impl PrefixSet {
  /// Port of `PrefixSet.__init__`.
  pub fn new() -> Self {
    Self::default()
  }

  /// Port of `PrefixSet.train`: add every prefix of every word.
  pub fn train<I, S>(&mut self, word_s: I)
  where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
  {
    for word in word_s {
      // 把词语的每个前缀更新到 prefix_set 中
      let word = word.as_ref();
      for (i, c) in word.char_indices() {
        self._set.insert(word[..i + c.len_utf8()].to_string());
      }
    }
  }

  /// Port of `PrefixSet.__contains__`.
  pub fn contains(&self, key: &str) -> bool {
    self._set.contains(key)
  }
}

/// Port of `pypinyin.seg.mmseg.seg`: the built-in strict
/// (`no_non_phrases=True`) segmenter trained on `PHRASES_DICT`.
#[allow(non_upper_case_globals)]
pub static seg: LazyLock<Seg> = LazyLock::new(|| {
  let mut prefix_set = PrefixSet::new();
  prefix_set.train(PHRASES_DICT.keys());
  Seg::new(prefix_set, true)
});

/// Port of `pypinyin.seg.mmseg.p_set`: the prefix set of [`seg`].
pub fn p_set() -> &'static PrefixSet {
  &seg._prefix_set
}

/// Port of `pypinyin.seg.mmseg.retrain`: train `seg_instance` on
/// `PHRASES_DICT` again.
pub fn retrain(seg_instance: &mut Seg) {
  seg_instance.train(PHRASES_DICT.keys());
}

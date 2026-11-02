//! Port of `nltk.tag.perceptron` (greedy averaged perceptron tagger, as
//! implemented by Matthew Honnibal).
//!
//! Only tagging with a pretrained model is ported; training, saving and
//! confidence scores are left out.

use std::collections::HashMap;
use std::sync::OnceLock;

use crate::python;

/// Port of `nltk.tag.perceptron.AveragedPerceptron` (prediction only).
#[derive(Debug, Clone)]
pub struct AveragedPerceptron {
  /// `self.weights`: feature -> [(class index, weight)], with class indices
  /// into [`AveragedPerceptron::classes`].
  pub weights: HashMap<String, Vec<(usize, f64)>>,
  /// `self.classes`, sorted. Python keeps a `set`; the order does not
  /// matter because `predict` breaks ties by label.
  pub classes: Vec<String>,
}

impl AveragedPerceptron {
  /// Builds the model from the decoded `weights` json object and the class
  /// list (`PerceptronTagger.decode_json_params`).
  pub fn new(weights: HashMap<String, HashMap<String, f64>>, class_list: Vec<String>) -> Self {
    let mut classes = class_list;
    classes.sort();
    classes.dedup();
    let index: HashMap<&str, usize> = classes
      .iter()
      .enumerate()
      .map(|(i, c)| (c.as_str(), i))
      .collect();
    let weights = weights
      .into_iter()
      .map(|(feat, w)| {
        let w = w
          .into_iter()
          // A label outside `classes` would get a score, but `max`
          // only looks at `classes`, so it can be dropped.
          .filter_map(|(label, weight)| index.get(label.as_str()).map(|&i| (i, weight)))
          .collect();
        (feat, w)
      })
      .collect();
    Self { weights, classes }
  }

  /// Port of `AveragedPerceptron.predict` with `return_conf=False`.
  ///
  /// Dot-products the features and current weights and returns the best
  /// label. Scores are summed in feature order, as in Python, and ties are
  /// broken like `max(self.classes, key=lambda label: (scores[label], label))`.
  pub fn predict(&self, features: &[(String, i64)]) -> &str {
    let mut scores = vec![0.0f64; self.classes.len()];
    for (feat, value) in features {
      if *value == 0 {
        continue;
      }
      let Some(weights) = self.weights.get(feat) else {
        continue;
      };
      for &(label, weight) in weights {
        scores[label] += *value as f64 * weight;
      }
    }
    let mut best = 0;
    for i in 1..self.classes.len() {
      if scores[i] > scores[best]
        || (scores[i] == scores[best] && self.classes[i] > self.classes[best])
      {
        best = i;
      }
    }
    &self.classes[best]
  }
}

/// Port of `nltk.tag.perceptron.PerceptronTagger` (tagging only).
#[derive(Debug, Clone)]
pub struct PerceptronTagger {
  /// `self.model`
  pub model: AveragedPerceptron,
  /// `self.tagdict`: tags for frequent, unambiguous words.
  pub tagdict: HashMap<String, String>,
}

/// Options of [`PerceptronTagger::tag`].
#[derive(Debug, Clone, Copy)]
pub struct TagOptions {
  /// `use_tagdict`, defaults to true.
  pub use_tagdict: bool,
}

impl Default for TagOptions {
  fn default() -> Self {
    Self { use_tagdict: true }
  }
}

impl PerceptronTagger {
  /// Port of `PerceptronTagger.START`.
  pub const START: [&'static str; 2] = ["-START-", "-START2-"];
  /// Port of `PerceptronTagger.END`.
  pub const END: [&'static str; 2] = ["-END-", "-END2-"];

  /// Port of `PerceptronTagger.decode_json_params`: builds a tagger from the
  /// three json model files (weights, tagdict, classes).
  pub fn from_json(weights: &str, tagdict: &str, classes: &str) -> serde_json::Result<Self> {
    let weights: HashMap<String, HashMap<String, f64>> = serde_json::from_str(weights)?;
    let tagdict: HashMap<String, String> = serde_json::from_str(tagdict)?;
    let classes: Vec<String> = serde_json::from_str(classes)?;
    Ok(Self {
      model: AveragedPerceptron::new(weights, classes),
      tagdict,
    })
  }

  /// Port of `PerceptronTagger()` (`load=True, lang="eng"`).
  ///
  /// The model is nltk_data `averaged_perceptron_tagger_eng`, embedded in the
  /// crate. It is parsed once and shared.
  pub fn eng() -> &'static Self {
    static TAGGER: OnceLock<PerceptronTagger> = OnceLock::new();
    TAGGER.get_or_init(|| {
      Self::from_json(
        include_str!("../../data/averaged_perceptron_tagger_eng.weights.json"),
        include_str!("../../data/averaged_perceptron_tagger_eng.tagdict.json"),
        include_str!("../../data/averaged_perceptron_tagger_eng.classes.json"),
      )
      .expect("embedded averaged_perceptron_tagger_eng model is valid")
    })
  }

  /// Port of `PerceptronTagger.classes`.
  pub fn classes(&self) -> &[String] {
    &self.model.classes
  }

  /// Port of `PerceptronTagger.tag` with `return_conf=False`.
  pub fn tag<S: AsRef<str>>(&self, tokens: &[S], options: TagOptions) -> Vec<(String, String)> {
    let [mut prev, mut prev2] = Self::START.map(str::to_owned);
    let mut output = Vec::with_capacity(tokens.len());

    let context: Vec<String> = Self::START
      .iter()
      .map(|s| (*s).to_owned())
      .chain(tokens.iter().map(|w| self.normalize(w.as_ref())))
      .chain(Self::END.iter().map(|s| (*s).to_owned()))
      .collect();
    for (i, word) in tokens.iter().enumerate() {
      let word = word.as_ref();
      let tag = if options.use_tagdict {
        self.tagdict.get(word).filter(|t| !t.is_empty()).cloned()
      } else {
        None
      };
      let tag = match tag {
        Some(tag) => tag,
        None => {
          let features = self._get_features(i, word, &context, &prev, &prev2);
          self.model.predict(&features).to_owned()
        }
      };
      output.push((word.to_owned(), tag.clone()));

      prev2 = std::mem::replace(&mut prev, tag);
    }

    output
  }

  /// Port of `PerceptronTagger.normalize`.
  ///
  /// - All words are lower cased
  /// - Groups of digits of length 4 are represented as `!YEAR`
  /// - Other digits are represented as `!DIGITS`
  pub fn normalize(&self, word: &str) -> String {
    if word.contains('-') && !word.starts_with('-') {
      return "!HYPHEN".to_owned();
    }
    if python::isdigit(word) && word.chars().count() == 4 {
      return "!YEAR".to_owned();
    }
    if word.chars().next().is_some_and(python::char_isdigit) {
      return "!DIGITS".to_owned();
    }
    python::lower(word)
  }

  /// Port of `PerceptronTagger._get_features`.
  ///
  /// Maps tokens into a feature representation, a `{feature: count}` dict
  /// kept in insertion order (the order in which Python sums the scores).
  pub fn _get_features(
    &self,
    i: usize,
    word: &str,
    context: &[String],
    prev: &str,
    prev2: &str,
  ) -> Vec<(String, i64)> {
    let mut features: Vec<(String, i64)> = Vec::with_capacity(14);
    let mut add = |name: &str, args: &[&str]| {
      let mut key = name.to_owned();
      for a in args {
        key.push(' ');
        key.push_str(a);
      }
      match features.iter_mut().find(|(k, _)| *k == key) {
        Some((_, v)) => *v += 1,
        None => features.push((key, 1)),
      }
    };

    let i = i + Self::START.len();
    let first = word.chars().next().map(String::from).unwrap_or_default();
    add("bias", &[]);
    add("i suffix", &[suffix3(word)]);
    add("i pref1", &[&first]);
    add("i-1 tag", &[prev]);
    add("i-2 tag", &[prev2]);
    add("i tag+i-2 tag", &[prev, prev2]);
    add("i word", &[&context[i]]);
    add("i-1 tag+i word", &[prev, &context[i]]);
    add("i-1 word", &[&context[i - 1]]);
    add("i-1 suffix", &[suffix3(&context[i - 1])]);
    add("i-2 word", &[&context[i - 2]]);
    add("i+1 word", &[&context[i + 1]]);
    add("i+1 suffix", &[suffix3(&context[i + 1])]);
    add("i+2 word", &[&context[i + 2]]);
    features
  }
}

/// Python `s[-3:]` (the last three code points).
fn suffix3(s: &str) -> &str {
  match s.char_indices().rev().nth(2) {
    Some((i, _)) => &s[i..],
    None => s,
  }
}

//! Port of `nltk.tag` (`pos_tag` for English).

pub mod perceptron;

pub use perceptron::{AveragedPerceptron, PerceptronTagger, TagOptions};

/// Port of `nltk.tag.pos_tag(tokens)` with the defaults `tagset=None` and
/// `lang="eng"`.
///
/// Tags the tokens with the pretrained English [`PerceptronTagger`]. Mapping
/// to another tagset and the Russian model are not ported.
pub fn pos_tag<S: AsRef<str>>(tokens: &[S]) -> Vec<(String, String)> {
  PerceptronTagger::eng().tag(tokens, TagOptions::default())
}

/// Port of `nltk.tag.pos_tag_sents(sentences)` with the defaults
/// `tagset=None` and `lang="eng"`.
pub fn pos_tag_sents<S: AsRef<str>>(sentences: &[Vec<S>]) -> Vec<Vec<(String, String)>> {
  sentences.iter().map(|s| pos_tag(s)).collect()
}

//! Subset port of the Python nltk 3.10.4 package (TweetTokenizer, pos_tag) used by GPT-SoVITS.
//!
//! GPT-SoVITS `text/english.py` calls
//! `nltk.tokenize.TweetTokenizer().tokenize(text)` and then
//! `nltk.pos_tag(words)`:
//!
//! ```
//! use nltk::{TweetTokenizer, pos_tag};
//!
//! let words = TweetTokenizer::new().tokenize("I can't wait :-)").unwrap();
//! assert_eq!(words, ["I", "can't", "wait", ":-)"]);
//! let tags = pos_tag(&words);
//! assert_eq!(tags[0], ("I".to_owned(), "PRP".to_owned()));
//! ```
//!
//! See the crate README for the ported scope and `parity/README.md` for the
//! known differences.

pub mod python;
pub mod tag;
pub mod tokenize;

pub use tag::{pos_tag, pos_tag_sents};
pub use tokenize::TweetTokenizer;

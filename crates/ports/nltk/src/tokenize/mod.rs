//! Port of `nltk.tokenize` (`TweetTokenizer` only).

pub mod casual;

pub use casual::{TweetTokenizer, casual_tokenize};

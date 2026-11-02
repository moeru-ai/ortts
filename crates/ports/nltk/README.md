# nltk

Rust port of [NLTK](https://www.nltk.org/) 3.10.4, limited to the two calls
that GPT-SoVITS `text/english.py` makes: `TweetTokenizer().tokenize` and
`pos_tag`.

```rust
use nltk::{TweetTokenizer, pos_tag};

let words = TweetTokenizer::new().tokenize("I can't wait :-)")?;
let tags = pos_tag(&words); // [("I", "PRP"), ...]
```

## Ported

- `nltk.tokenize.casual`: `TweetTokenizer`, `casual_tokenize`,
  `reduce_lengthening`, `remove_handles` and the regex constants
- `nltk.tag.perceptron`: `PerceptronTagger::eng()` (embedded English model),
  `AveragedPerceptron::predict`
- `nltk.tag`: `pos_tag`, `pos_tag_sents`
- `python`: the CPython `str` methods and builtins the above rely on

## Not ported

- Training and saving the tagger
- `return_conf=True`, `pos_tag` with a `tagset`, the Russian model
- The rest of NLTK

## Parity

Fixtures are generated from the pinned Python package with
`pixi run --manifest-path crates/ports/nltk/pixi.toml fixtures`; the Rust
tests need no downloads. All 7 fixture sets match exactly (for example 7681 of
7681 `tokenize.jsonl` cases, 1220 of 1220 `pos_tag.jsonl` cases).
Known differences:

- Lone surrogates from HTML entities become U+FFFD.
- The regex time limit counts backtracking steps, not seconds.
- `int()` values that do not fit in an `i64` saturate (same output).

See [parity/README.md](parity/README.md) for the regex translation, data
provenance, every fixture set and the details of each difference.

## License

Apache-2.0, Copyright (C) 2001-2026 NLTK Project. See `LICENSE` and `NOTICE`.

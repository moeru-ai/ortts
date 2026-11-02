# nltk

Subset port of [NLTK](https://www.nltk.org/) 3.10.4 (Apache-2.0) to Rust: the
two calls that GPT-SoVITS `text/english.py` makes.

```python
word_tokenize = TweetTokenizer().tokenize
tokens = pos_tag(word_tokenize(text))
```

```rust
let words = nltk::TweetTokenizer::new().tokenize(text)?;
let tokens = nltk::pos_tag(&words);
```

## Ported scope

| Python | Rust |
|---|---|
| `nltk.tokenize.casual`: `EMOTICONS`, `URLS`, `FLAGS`, `PHONE_REGEX`, `REGEXPS`, `REGEXPS_PHONE` | `tokenize::casual` constants, sources copied verbatim |
| `HANG_RE`, `EMOTICON_RE`, `ENT_RE`, `HANDLES_RE` | `hang_re()`, `emoticon_re()`, `ent_re()`, `handles_re()` (sources in `*_SRC`) |
| `_replace_html_entities(text, keep, remove_illegal)` | `_replace_html_entities` |
| `TweetTokenizer(preserve_case, reduce_len, strip_handles, match_phone_numbers)` | `TweetTokenizer` (the fields are the keyword arguments), `WORD_RE()`, `PHONE_WORD_RE()`, `tokenize` |
| `reduce_lengthening`, `remove_handles`, `casual_tokenize` | same names |
| `nltk.redos.compile` | `casual::compile` (see below) |
| `nltk.tag.perceptron.AveragedPerceptron.predict` (`return_conf=False`) | `AveragedPerceptron::predict` |
| `PerceptronTagger()` with `lang="eng"` | `PerceptronTagger::eng()` (embedded model, parsed once) |
| `PerceptronTagger.tag(tokens, use_tagdict=...)`, `normalize`, `_get_features`, `START`, `END`, `decode_json_params` | `tag`, `normalize`, `_get_features`, `START`, `END`, `from_json` |
| `nltk.tag.pos_tag(tokens)`, `pos_tag_sents(sentences)` (`tagset=None`, `lang="eng"`) | `pos_tag`, `pos_tag_sents` |
| CPython builtins used by the above: `str.lower`, `str.isdigit`, `str.isspace`, `int(s, 10/16)`, `bytes.decode("cp1252")`, `html.entities.name2codepoint` | `python` module |

`tokenize` and the other regex helpers return `Result<_, TimeoutError>`, like
Python raising `TimeoutError` (see "Differences").

## Left out

- Training and saving (`train`, `update`, `average_weights`, `save`, `load`,
  `save_to_json`, `_make_tagdict`) and the path sandboxing around them.
- `return_conf=True` (the numpy softmax confidence).
- `pos_tag` with a `tagset` (`map_tag`) and the Russian model (`lang="rus"`).
- `_str_to_unicode`: Rust input is always `&str`, so there are no bytes to
  decode.
- The rest of NLTK.

## How the regexes are ported

NLTK compiles the tokenizer regexes with the third-party `regex` module
(2026.9.29 in the parity environment), flags `VERBOSE | I | UNICODE` (V0
semantics). The port uses `fancy-regex`, a backtracking engine (needed for
`(?<!@)`, `(?!@)` and the `\1` backreferences), with the same leftmost-first
alternation order. Two translation steps keep the semantics exact:

1. `strip_verbose` applies `regex.VERBOSE` the way the `regex` parser does:
   it drops whitespace (`str.isspace`) and `#` comments outside sets, and
   keeps sets verbatim. Rust verbose mode would also drop the space in
   `PHONE_REGEX`'s `[ *\-.\)]`, so the pattern is compiled non-verbose.
2. Every atom whose meaning depends on the Unicode database or on case
   folding (sets, `\w \W \d \D \s \S`, `\p{..}`, `.`, and under `I` literal
   letters, non-ASCII literals and `\U \u \x` escapes) is replaced by the
   explicit set of code points that the Python `regex` module matches for it
   with the same flags (`data/regex_sets.json`). `\b` becomes a lookaround on
   that `\w` set. No `(?i)` is passed to Rust, so Rust's Unicode 16 tables and
   simple case folding are never used. Without this, `\w` differs in 17559
   code points (`regex` uses a newer Unicode version) and `[a-z]` under `I`
   would miss U+0130 (Python matches it).

The Python generator and the Rust compiler share the atom scanner; a test
checks that both find the same atoms in every pattern.

CPython string methods use the Unicode 14.0.0 database of Python 3.11, not
the one of the `regex` module, so `python::lower`, `isdigit`, `isspace` and
the decimal digits for `int()` use tables measured from CPython
(`data/python_str.json`), including the Case_Ignorable and Cased properties
that the Final_Sigma rule of `str.lower` reads.

## Data

| File | Source | sha256 |
|---|---|---|
| `data/averaged_perceptron_tagger_eng.weights.json` | nltk_data `taggers/averaged_perceptron_tagger_eng`, unchanged | `789e8e35f6fac656d9b7eccea29ba4e8a137cbb5b525dd31f39e4291a9e80832` |
| `data/averaged_perceptron_tagger_eng.tagdict.json` | same | `4c713731cb06727736962bc8cd94ad919217a040e76691059af99bc0c2c9c246` |
| `data/averaged_perceptron_tagger_eng.classes.json` | same | `d152ffa69d45b8357276341a3d592c6c025911d52ec0d9068d4f382560dd48b6` |
| `data/html_name2codepoint.json` | Python 3.11 `html.entities.name2codepoint` | generated |
| `data/python_str.json` | CPython 3.11 `str` methods (Unicode 14.0.0) | generated |
| `data/regex_sets.json` | Python `regex` 2026.9.29, one entry per regex atom | generated |

Regenerate the generated files (and check the model hashes) with:

```sh
pixi run --manifest-path crates/ports/nltk/pixi.toml data
```

The model weights are parsed with `serde_json`'s `float_roundtrip` feature,
so every weight is the correctly rounded `f64` that Python's `json` produces.
Scores are summed in the same feature order as Python, so the `f64` sums are
bit-identical; ties are broken by the label, as
`max(self.classes, key=lambda label: (scores[label], label))` does.

## Parity

Fixtures are generated by `parity/gen_fixtures.py` in the crate's own pixi
workspace (`pixi.toml`). The task first downloads the reference assets into
the gitignored `parity/.cache/` (see `parity/README.md`):

```sh
pixi run --manifest-path crates/ports/nltk/pixi.toml fixtures
```

| Fixture | Cases | Exact |
|---|---|---|
| `tokenize.jsonl`: `TweetTokenizer().tokenize` and `pos_tag` of the tokens (hand-written edge cases, every pool item alone and in context, 5000 generated sentences over the tagger vocabulary, 1500 random multi-script strings, random entity bodies and digit runs; seeds fixed) | 7681 | 7681 tokens, 7681 tags |
| `tokenize_options.jsonl`: `preserve_case=False`, `reduce_len`, `strip_handles`, `match_phone_numbers=False`, and all four | 2335 | 2335 |
| `pos_tag.jsonl`: `PerceptronTagger.tag` with and without the tagdict, on token lists that do not come from the tokenizer (empty strings, hyphens, years, Unicode digits, sigma) | 1220 | 1220 |
| `corpus.jsonl`: every text that `english.py` passes to `word_tokenize` while the GPT-SoVITS frontend processes `parity/corpus/gpt_sovits_zh_en_yue.json` | 122 | 122 tokens, 122 tags |
| `char_classes.json`: every regex atom (and `\w\b`) over all 1112064 code points | 56 probes | 56 |
| `python_lower.json`: `str.lower` of every code point, plus 3000 Final_Sigma strings | 1115064 | 1115064 |
| `regex_sources.json`: the regex sources, flags and atoms | 1 | 1 |

The exception lists in `tests/casual.rs` (`TOKENIZE_EXCEPTIONS`,
`CLASS_EXCEPTIONS`) are empty.

## Differences

- Lone surrogates: Python decodes `&#xD800;` to `&#xDFFF;` (and the decimal
  forms) to a lone surrogate, which a Rust `String` cannot hold. The port
  returns U+FFFD instead. JSON cannot carry lone surrogates either, so the
  generator writes U+FFFD in their place and marks the record
  `"surrogate": true` (85 records in `tokenize.jsonl`, 30 in
  `tokenize_options.jsonl`); the test checks that each such record contains
  a surrogate entity and that the output is otherwise exact. U+FFFD and a
  surrogate are both `\S`, not `\w`, not `\p{L}`/`\p{N}`, and uncased, so the
  token boundaries and tags are the same.
- Time limit: `nltk.redos` stops a match after 5 seconds of wall-clock time;
  `fancy-regex` stops after 100,000,000 backtracking steps. Both report
  `TimeoutError`, but an input near the limit can time out in one and not
  the other.
- `int()` results that do not fit in an `i64` saturate. They only reach
  `chr()`, which rejects them either way, so the output is the same.

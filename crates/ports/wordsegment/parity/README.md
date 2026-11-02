# wordsegment parity

Maintainer notes for the wordsegment 1.3.1 port: API mapping, behaviour
details, fixtures and the tooling that generates them.

GPT-SoVITS calls `wordsegment.segment(word.lower())` from `en_G2p.qryword`
(`GPT_SoVITS/text/english.py`) to split out-of-vocabulary English words.

## API mapping

All of `wordsegment/__init__.py` except the CLI:

| Python | Rust |
|---|---|
| `Segmenter` | `Segmenter` (`unigrams`, `bigrams`, `total`, `limit`, `words` fields) |
| `Segmenter.ALPHABET`, `TOTAL`, `LIMIT` | associated consts |
| `Segmenter.UNIGRAMS_FILENAME`, `BIGRAMS_FILENAME`, `WORDS_FILENAME` | `UNIGRAMS_DATA`, `BIGRAMS_DATA`, `WORDS_DATA` (embedded file contents) |
| `Segmenter.load`, `parse`, `score`, `isegment`, `segment`, `divide`, `clean` | methods with the same names |
| module-level `load`, `clean`, `isegment`, `segment` | free functions that use a global `RwLock<Segmenter>` |
| module-level `UNIGRAMS`, `BIGRAMS`, `WORDS` | functions that return a read guard to the global segmenter's field |

Data files `data/unigrams.txt`, `data/bigrams.txt` and `data/words.txt` are
copied unmodified from the 1.3.1 wheel and embedded with `include_str!`.

Left out:

- `main` (the command-line interface) and `__main__.py`.
- `parse` takes file contents instead of a file name; `load` reads the
  embedded data instead of files next to the module.
- `isegment` computes the whole result before it returns the iterator, instead
  of yielding words lazily per chunk. The words are the same.

## Behaviour kept from Python

- `f64` math in the same order as Python: `count / total`,
  `10.0 / (total * float(10 ** len(word)))` (the power of ten is correctly
  rounded, like Python's int-to-float conversion), and
  `bigram / total / score(previous)` for the stupid-backoff path.
- `max(candidates())` tie-breaking: a candidate replaces the current best only
  when the `(score, words)` tuple is strictly greater, so equal scores fall back
  to comparing the word lists lexicographically.
- `isegment` chunks the cleaned text in 250-character chunks and carries the
  last five words of each chunk into the next.
- `bigrams.txt` contains duplicate keys; the last value wins, like `dict()`.
- `load()` called twice extends `words` again, like Python.
- Python exceptions become panics with the Python exception name in the
  message: `ZeroDivisionError` when `score` runs on an unloaded segmenter,
  `OverflowError` for an OOV word longer than 308 characters, `ValueError`
  from `max()` when `limit` is 0, and `ValueError` for malformed data lines.

## Implementation differences that do not change results

- `unigrams` and `bigrams` are `Counts` (a `HashMap` with an in-crate
  `FxHasher`) instead of the default SipHash map, for lookup speed.
- The `search` memo is keyed by byte offsets into the searched text instead of
  `(suffix, prefix)` strings, and candidate word lists share their memoized
  suffix as a linked list instead of copying it.
- `score` checks `previous in unigrams` before building the bigram key. Python
  checks `bigram in bigrams` first; both are side-effect-free lookups.

## Fixtures

`gen_fixtures.py` writes the fixtures in `tests/fixtures/`; `tests/parity.rs`
checks them exactly (scores compare the f64 bit pattern). Inputs are
hand-written edge cases (empty, punctuation, digits, apostrophes, non-ASCII
including characters that lowercase to ASCII, inputs longer than one
250-character chunk) and seeded bulk inputs (2-5 word concatenations from
common, rare and `words.txt` entries, camelCase product names, synthetic OOV
names, mixed digits and apostrophes, very short and 40+ character strings).
There are no known mismatches, so the exception list is empty.

| Fixture | Cases | Exact |
|---|---|---|
| `segment.jsonl` (`clean` and `segment`) | 6050 | 6050 |
| `score.jsonl` (f64 bits) | 4491 | 4491 |
| `divide.jsonl` | 40 | 40 |
| `meta.json` (table sizes, `WORDS` samples, `TOTAL`, `LIMIT`) | 1 | 1 |

The segment fixture runs about 93 million candidate evaluations, so
`cargo test -p wordsegment` takes about 80 s in a debug build (about 12 s with
`--release`). Generating the fixtures with Python takes about 9 minutes.

The fixtures use fixed seeds, so the same environment gives byte-identical
output.

## Tooling

These scripts generate the golden fixtures, the data files and `LICENSE` of
this crate from the Python package wordsegment 1.3.1. They run in the small
pixi workspace of this crate (`pixi.toml`, `pixi.lock`). Run the tasks from
the repository root:

```sh
pixi run --manifest-path crates/ports/wordsegment/pixi.toml fixtures
pixi run --manifest-path crates/ports/wordsegment/pixi.toml data
pixi run --manifest-path crates/ports/wordsegment/pixi.toml license
```

Or run `pixi run <task>` in `crates/ports/wordsegment/`.

| Task | Script | Writes |
|---|---|---|
| `fixtures` | `gen_fixtures.py` | `tests/fixtures/{segment,score,divide}.jsonl`, `tests/fixtures/meta.json` (about 9 minutes) |
| `data` | `copy_data.py` | `data/{unigrams,bigrams,words}.txt`, copied unmodified from the wordsegment 1.3.1 package |
| `license` | `make_license.py` | `LICENSE` (upstream copyright and the Apache-2.0 text) |
| `apache-license` | `fetch.py` | `parity/.cache/LICENSE-2.0.txt`, downloaded from apache.org; `license` runs it first |

`fetch.py` skips the download when the file is present with the expected
SHA-256. The cache directory is `parity/.cache/` (gitignored). Set
`WORDSEGMENT_PARITY_CACHE` to use another directory.

# `gpt_sovits_text::en_normalization` parity

Port of `GPT_SoVITS/text/en_normalization/expend.py` (GPT-SoVITS @ `04a44eb`)
to `crates/ports/gpt_sovits_text/src/en_normalization/`.

## Ported scope

All of `expend.py`, under the Python names:

- `normalize(text)`, which returns `Result<String, Error>`. `Error` is the
  Python exception that `normalize` raises: `ValueError` from `int()` (for
  example `"$,5"` gives `int(",5")`, `"(5$"` gives `int("(5")`), or an
  `inflect` exception (`NumOutOfRangeError` for numbers of 37 or more digits).
- The regexes `_ordinal_number_re`, `_comma_number_re`, `_time_re`,
  `_measurement_re`, `_pounds_re_start`, `_pounds_re_end`,
  `_dollars_re_start`, `_dollars_re_end`, `_decimal_number_re`,
  `_fraction_re`, `_ordinal_re`, `_number_re`, `RE_ASMD`, `RE_INTEGER`
  (upper-case statics with the Python sources unchanged), and the inline
  patterns of `normalize`.
- The replacement functions `_convert_ordinal`, `_remove_commas`,
  `_expand_time`, `_expand_measurement`, `_expand_pounds`, `_expand_dollars`,
  `_expand_decimal_number`, `_expend_fraction`, `_expand_ordinal`,
  `_expand_number`, `replace_asmd`, `replace_negative_num`.
- The data `measurement_map` (`MEASUREMENT_MAP`) and `asmd_map`
  (`ASMD_MAP`).

The quirks of the Python code are kept, for example: `RE_INTEGER` drops the
whitespace before a negative number (`"it is -5"` gives `"it isnegative
five"`); `5km/h` gives `"five kilometersh"` because `km` comes before `km/h`
in the alternation and the `/` is filtered out later; `"$1.2.3"` gives
`"one point two.three dollars"`; the `(` in `_dollars_re_end` makes `"(5$"`
raise `ValueError`.

Left out: the `if __name__ == "__main__"` demo (its inputs are in the
fixtures).

## Python semantics

`pyre.rs` (no Python counterpart) reproduces what the Rust regex crates do
differently from Python 3.11 `re`:

- `\d`, `\s` and `\w` are Unicode classes in Python (`str.isdecimal`,
  `str.isspace`, `str.isalnum` or `_`). Rust's `\s` lacks U+001C..U+001F, and
  its `\w` includes combining marks and lacks other numbers such as `²`. The
  patterns use explicit classes from `py_tables.rs`, which `gen_tables.py`
  generates from Python (Unicode 14.0.0).
- `\b` is built from that `\w` class with lookarounds (`fancy-regex`), so
  `"²3:30"` has no boundary before `3`, as in Python.
- `\×` (an escaped non-ASCII char) is the literal char.
- `int()` is `py_int`: Python `int` values are kept as digit strings without
  leading zeros, so numbers beyond `i128` behave as in Python. It reproduces
  the CPython 3.11 errors, including the 4300 digit limit and the `%.200R`
  truncation of the `invalid literal` message.
- "Strip accents" is NFD (`unicode-normalization`) and the removal of
  category `Mn` (table generated from Python). `unicode-normalization` may
  use a newer Unicode version than Python's 14.0.0; characters that it
  decomposes and Python does not are newer than Unicode 14 and do not
  decompose to ASCII, so the next step (`[^ A-Za-z'.,?!\-]`) removes them
  either way.
- `(?i)i\.e\.` and `(?i)e\.g\.` and the capital split
  `(?<!^)(?<![\s])([A-Z])` only see ASCII (they run after the filter). The
  capital split is a char scan, which is equivalent because both lookbehinds
  test the original string.

## Fixtures

`gen_fixtures.py` writes `crates/ports/gpt_sovits_text/tests/fixtures/en_normalization/`:

| Fixture | Cases | Exact matches |
|---|---|---|
| `hand.jsonl`: hand-written cases for every regex and branch (upstream `__main__` examples, ordinals, math operators, negatives, commas, times, units, pounds, dollars, decimals, fractions, years, percentages, accents, Unicode digits, spaces and boundaries, abbreviations, acronyms, ranges, error inputs) | 345 | 345 |
| `corpus.jsonl`: every text of `parity/corpus/gpt_sovits_zh_en_yue.json` and every non-CJK span of it (as is and stripped) | 220 | 220 |
| `bulk.jsonl`: 5200 random sentences and 1500 single tokens (seed `20240501`) | 6700 | 6700 |
| `py_int.jsonl`: `str(int(s))` for the strings `expend.py` can pass to `int()` | 32 | 32 |

Each line is `{"i": input, "o": output}` or `{"i": input, "e": [type,
message]}`. 94 normalize cases expect an exception. There are no known
mismatches; `tests/en_normalization.rs` still has an explicit (empty)
`KNOWN_MISMATCHES` list.

The 4300 digit limit of `int()` is tested through `py_int` only:
`normalize` on a run of more than 4300 digits does not finish in Python in
reasonable time, because `_pounds_re_end` (`[0-9\.\,]*[0-9]+£`) backtracks in
cubic time.

Regenerate:

```sh
pixi run --manifest-path crates/ports/gpt_sovits_text/pixi.toml data en_normalization
pixi run --manifest-path crates/ports/gpt_sovits_text/pixi.toml fixtures en_normalization
```

# inflect

A Rust port of a subset of the Python
[inflect](https://github.com/jaraco/inflect) package, version 7.6.0
(MIT). GPT-SoVITS uses it in
`GPT_SoVITS/text/en_normalization/expend.py` to read numbers as English
words.

```rust
use inflect::{NumberToWordsOptions, engine};

let mut p = engine::new();
assert_eq!(p.number_to_words(101, &NumberToWordsOptions::default())?, "one hundred and one");
let opts = NumberToWordsOptions { andword: "", zero: "oh", group: 2, ..Default::default() };
assert_eq!(p.number_to_words(1905, &opts)?, "nineteen, oh five");
assert_eq!(p.ordinal("twenty-one")?, "twenty-first");
assert_eq!(p.ordinal("3")?, "3rd");
# Ok::<(), inflect::Error>(())
```

## How GPT-SoVITS calls it

`expend.py` makes these calls:

| Call | Argument type |
|---|---|
| `number_to_words(hours)`, `number_to_words(minutes)` | `int` |
| `number_to_words(numerator)`, `number_to_words(denominator)` | `int` |
| `ordinal(number_to_words(denominator))` | `str` (number words) |
| `ordinal(m.group(1))` | `str` of ASCII digits |
| `number_to_words(m.group(0))` | `str` such as `"21st"` |
| `number_to_words(num % 100)`, `number_to_words(num // 100)` | `int` |
| `number_to_words(num, andword="", zero="oh", group=2)` | `int` |
| `number_to_words(num, andword="")` | `int` |

`NumberOrWord` holds both argument types: `Number(i128)` for a Python `int`
and `Word(&str)` for a Python `str`. Integer types and `&str` convert into it.

## Ported

From `inflect/__init__.py`:

- `engine.number_to_words` with the keyword arguments `group`, `comma`,
  `andword`, `zero`, `one` and `decimal` (`NumberToWordsOptions`).
- `engine.ordinal`.
- The call graph of these two methods: `enword`, `_render_group`,
  `group1sub`, `group1bsub`, `group2sub`, `group3sub`, `hundsub`, `tensub`,
  `unitsub`, `hundfn`, `tenfn`, `unitfn`, `millfn`, `_sub_ord`, `_chunk_num`,
  `_remove_last_blank`, `_get_sign`, `_render`, `_grouped_from_right`.
- The data: `nth`, `nth_suff`, `ordinal`, `unit`, `teen`, `ten`, `mill`.
- The regular expressions on that path (`DIGIT`, `DIGIT_GROUP`, `TWO_DIGITS`,
  `THREE_DIGITS`, `THREE_DIGITS_WORD`, `TWO_DIGITS_WORD`, `ONE_DIGIT_WORD`,
  `NON_DIGIT`, `WHITESPACES_COMMA`, `COMMA_WORD`, `WHITESPACES`,
  `ordinal_suff`) as hand-written matchers with the same results.
- The exceptions on that path: `BadNumValueError`, `BadChunkingOptionError`,
  `NumOutOfRangeError`, and the Python `IndexError` and
  `typeguard.TypeCheckError` (empty string) that the methods raise for some
  inputs. They are variants of `Error`.

Python `re` matches `\d` against all Unicode decimal digits and `\s` against
all Unicode whitespace, and `int()` accepts Unicode digits too. The port does
the same. `data/decimal_digits.tsv` holds the decimal digit ranges of the
pinned Python 3.11 (Unicode 14.0.0). `parity/gen_fixtures.py` generates it
(`pixi run --manifest-path crates/ports/inflect/pixi.toml data`).

## Not ported

- All other parts of inflect: plurals, singulars, articles (`a`, `an`),
  comparisons, `present_participle`, `number_to_currency`, `join`, the
  user-defined words (`defnoun`, ...) and classical modes.
- `number_to_words(wantlist=True)` and the `threshold` argument. GPT-SoVITS
  uses the defaults (`wantlist=False`, `threshold=None`).
- `float` and `Decimal` inputs (including the exponent formatting of
  floats). GPT-SoVITS passes only `int` and `str`.
- Python `int` values outside the `i128` range. Pass them as a string: any
  number with more than 36 digits raises `NumOutOfRangeError` anyway, except
  numbers whose high groups are all zero.
- Non-string falsy values for `comma` and `decimal`. An empty `comma` stands
  for every falsy value. `decimal` is `Option<&str>`: `None` is Python
  `None` and `Some("")` is the empty string. Python `False` for `decimal` is
  not supported.
- An `andword` with a backslash. Python uses `andword` in a `re`
  replacement template in one place (`COMMA_WORD.sub`), where backslash
  escapes are expanded. The port inserts `andword` literally.

## Parity

`parity/gen_fixtures.py` runs the pinned inflect and writes:

- `tests/fixtures/range.json`: `number_to_words(n)` and
  `number_to_words(n, andword="")` for every `n` in `0..=20000`
  (2 x 20001 cases).
- `tests/fixtures/cases.jsonl`: 19702 cases in blocks that share keyword
  arguments:
  - number_to_words, 13430 cases: hand-written edge cases (padding, signs,
    decimal strings, ordinal suffixes such as `"21st"`, separators, whitespace
    only input, Unicode digits and whitespace, numbers at and above the
    decillion limit, the empty string) for each of 23 keyword sets (the
    three sets that `expend.py` uses, plus variants of `group`, `zero`,
    `one`, `andword`, `comma` and `decimal`); `group` -1 and 4; every int in
    `1000..=3000` with `andword="", zero="oh", group=2`; `"<n>st"`,
    `"<n>nd"`, `"<n>rd"` and `"<n>th"` for `n` in `0..1000`; seeded random
    ints of up to 38 digits (some negative), random ints with many zero
    groups, and random strings of digits, points, signs, commas, spaces,
    underscores and letters.
  - ordinal, 6272 cases: hand-written edge cases (decimal points, signs,
    underscores, the whitespace that `int()` accepts or rejects, Unicode
    digits, number words); every int in `-120..1200` as `int` and as `str`;
    `number_to_words(n)` for `n` in `0..=2000`; seeded random ints of up to
    36 digits and their words; random short strings.
  - 438 cases expect a Python exception, which the test checks by class
    name.

`tests/parity.rs` asserts that every case is exactly equal, outputs and
errors. All 59704 cases (19702 + 2 x 20001) match. There are no documented
exceptions.

The crate has its own pixi workspace (`pixi.toml`, `pixi.lock`) with the
pinned Python 3.11 and inflect 7.6.0. Regenerate the fixtures and the data
table with:

```sh
pixi run --manifest-path crates/ports/inflect/pixi.toml fixtures
pixi run --manifest-path crates/ports/inflect/pixi.toml data
```

See `parity/README.md` for details.

# split_lang parity and port notes

Python reference scripts for this crate, plus the maintainer notes on how the
port maps to Python. They run in the crate's own pixi
workspace (`../pixi.toml`, locked by `../pixi.lock`), which pins Python 3.11,
split-lang 2.1.1, budoux 0.10.2, fast-langdetect 1.1.0 and fasttext-predict
0.9.2.4. Nothing outside this crate is needed.

## Tasks

Run from the repository root (or drop `--manifest-path` and run from the
crate directory):

```sh
pixi run --manifest-path crates/ports/split_lang/pixi.toml lid-model
pixi run --manifest-path crates/ports/split_lang/pixi.toml fixtures
pixi run --manifest-path crates/ports/split_lang/pixi.toml data
```

| Task        | Script               | Output                                                     |
| ----------- | -------------------- | ---------------------------------------------------------- |
| `lid-model` | `fetch_lid_model.py` | `parity/.cache/fast_langdetect/lid.176.bin` (131 MB)       |
| `fixtures`  | `gen_fixtures.py`    | `tests/fixtures/split_by_lang.jsonl`, `tests/fixtures/budoux.jsonl` |
| `data`      | `gen_data.py`        | `data/python_isdigit.txt`, `data/budoux/{ja,zh-hans}.json` |

`fixtures` runs `lid-model` first. `lid-model` skips the download when the
file is present with the expected MD5. `fixtures` takes about 15 seconds,
`data` under a second.

## Files

- `gen_fixtures.py`: runs split-lang with the GPT-SoVITS `LangSegmenter`
  configuration (its `DEFAULT_LANG_MAP` is copied into the script) and the
  full fasttext model, and budoux, over the inputs of `lang_inputs.py`.
- `gen_data.py`: writes the code point ranges for which Python 3.11
  `str.isdigit()` is true, and copies the budoux default models.
- `lang_inputs.py`: the fixture inputs (hand-written cases, pool fragments,
  seeded random concatenations, and the corpus texts).
- `corpus/*.json`: copies of the GPT-SoVITS end-to-end sentence corpora.
- `fetch_lid_model.py`: downloads `lid.176.bin`.
- `.cache/` (gitignored): downloaded files.

## Environment variables

- `FAST_LANGDETECT_MODEL`: path of `lid.176.bin`. Default
  `parity/.cache/fast_langdetect/lid.176.bin`. `lid-model` downloads to this
  path, `gen_fixtures.py` reads it, and the Rust `split_by_lang` test reads
  it (the test is skipped when it is unset).

## Fixtures

`gen_data.py` (task `data`) writes `data/python_isdigit.txt` and copies the
budoux models. `gen_fixtures.py` (task `fixtures`) writes `tests/fixtures/`
(1.3 MB):

- `split_by_lang.jsonl`: `split_by_lang` for 3571 inputs in three
  configurations: `gsv` (`LangSplitter(lang_map=LangSegmenter.DEFAULT_LANG_MAP)`),
  `gsv_no_digit` (the same with `merge_across_digit = False`, as `getTexts`
  runs it) and `default` (`LangSplitter()`, every 5th input). Results are
  `[lang, text, index, length]` lists or the exception class name.
- `budoux.jsonl`: Japanese and Simplified Chinese `parse` of 4104 texts
  (every CJK input and every ZH_JA section from `pre_split`).

The inputs (`lang_inputs.py`, the same inputs as the
`gpt_sovits_text::lang_segmenter` fixtures) are hand-written edge cases
(empty and whitespace-only strings, `\r\n`, apostrophes, digits including
full-width and superscript ones, punctuation runs, emoji, kana-only and
kanji-only strings, Hangul and compatibility jamo, half-width kana,
full-width Latin, Cantonese, Traditional Chinese, short Latin words, names
and acronyms), every pool fragment on its own, 2300 random concatenations
(seed 20240611) and every text of `corpus/*.json` (copies of the
GPT-SoVITS end-to-end corpora).

## Results

Run the tests with the model:

```sh
FAST_LANGDETECT_MODEL=crates/ports/split_lang/parity/.cache/fast_langdetect/lid.176.bin cargo test -p split_lang
```

Without `FAST_LANGDETECT_MODEL` the `split_by_lang` test is skipped (it
prints a message); the budoux test always runs.

Result (macOS arm64): `gsv` 3571/3571, `gsv_no_digit` 3571/3571, `default`
715/715, budoux 4104/4104 texts exact. The documented exception list
(`KNOWN_MISMATCHES` in `tests/parity.rs`) is empty.

Notes:

- Line coverage of `splitter.py` over the inputs (checked with
  `sys.settrace`) reaches every reachable line. Not reached:
  `_get_merge_direction`'s first/last-index branches (its caller handles
  those indices), `merge_across_punctuation`'s "keep separate" branch (needs
  a non-empty `not_merge_punctuation`), and the merge of consecutive
  same-language substrings in `_merge_substrings_across_digit` /
  `_merge_substrings_across_newline` and of a non-leading punctuation
  substring in `_merge_substrings_across_punctuation` (earlier passes already
  merge such substrings; a 30k random string search found no input).
- Parity of detection results relies on `fast_langdetect`, whose own notes
  apply (platform libm, libc++ tie order).

## Language detection model

GPT-SoVITS (`text/LangSegmenter/langsegmenter.py`) runs
`LangSplitter(lang_map=LangSegmenter.DEFAULT_LANG_MAP)` with
`merge_across_digit = False`. split-lang calls the module-level
`fast_langdetect.detect`, which uses fast-langdetect's `_default_detector`.
GPT-SoVITS replaces that detector at import time so the full `lid.176.bin`
from `pretrained_models/fast_langdetect` is used. The port keeps this design
and adds an optional handle:

- `LangSplitter::detector` is `None` by default. The detector functions then
  call `fast_langdetect::detect(.., None)`, which uses the global default
  detector. Replace it with `fast_langdetect::set_default_detector` (or
  `gpt_sovits_text::lang_segmenter::set_fast_langdetect_cache_dir`), as
  GPT-SoVITS does.
- `LangSplitter::detector = Some(Arc<LangDetector>)` uses that detector
  instead, without global state. The functions in `detect_lang::detector` take
  the same `Option<&LangDetector>`.

The model file is not committed and never downloaded by the crate. A missing
model gives `SplitLangError::FastLangdetect(FileNotFound)`.

## Module mapping

`split_lang/config.py` (`split_lang::config`): `DEFAULT_LANG_MAP`,
`ZH_JA_LANG_MAP`, `NO_ZH_JA_LANG_MAP`, `DEFAULT_LANG`. Dicts are
`(key, value)` slices in insertion order; `lang_map.get(k, default)` is
`config::lang_map_get`.

`split_lang/model.py` (`split_lang::model`): `LangSectionType`, `SubString`,
`SubStringSection`. `index` and `length` count code points, like Python
`len()`.

`split_lang/detect_lang/detector.py` (`split_lang::detect_lang::detector`):
`fast_lang_detect`, `detect_lang_combined`, `possible_detection_list`.

`split_lang/split/utils.py` (`split_lang::split::utils`): `PUNCTUATION`,
`contains_chinese_char`, `contains_hangul`, `contains_ja_kana`,
`contains_zh_ja`, `contains_only_kana`.

`split_lang/split/splitter.py` (`split_lang::split::splitter`): `LangSplitter`
with every method that `split_by_lang` reaches: `pre_split`, `_split`,
`_smart_merge_all`, `_parse_without_zh_ja`, `_parse_zh_ja`, `_smart_merge`,
`_is_merge_middle_to_two_side`, `_is_cur_short_and_near_long`,
`_merge_middle_substr_to_two_side`, `_merge_side_substr_to_near`,
`_fill_unknown_language`, `_get_nearest_lang_with_direction`,
`_get_merge_direction`, `_merge_substrings`, `_special_merge_for_zh_ja`, and
the `_merge_substrings_across_{digit,punctuation,newline}` methods with their
`_based_on_sections` drivers. All `__init__` options are supported.

budoux 0.10.2 `budoux/parser.py` (`split_lang::budoux`): `Parser`,
`Parser.parse`, `load_default_japanese_parser`,
`load_default_simplified_chinese_parser`, with the models embedded from
`data/budoux/`. Scores use the same `f64` arithmetic as Python (integer
weights plus the `-sum * 0.5` base score, all exact).

## Kept quirks and Python semantics

Quirks are kept, for example:

- `_get_languages` computes a language and discards it (the detector is still
  called, so its errors propagate).
- A KO section's substring gets `section_index` added twice in `_split`.
- `_special_merge_for_zh_ja` skips the length count of a block it merges
  ahead, and `max(dict, key=dict.get)` picks the first key with the maximum.
- Empty or whitespace-only input raises `IndexError` (`sections[0]`).

Python `str` semantics are reproduced where they matter: `str.isdigit()`
(table `data/python_isdigit.txt`, generated from Python 3.11 / Unicode 14.0),
`str.isspace()` (Rust `White_Space` plus U+001C..U+001F) and `str.strip()`.

Python exceptions are returned as `SplitLangError` (`IndexError`, `KeyError`,
or the fast-langdetect error). `SplitLangError::exception_name` gives the
Python class name.

## Not ported

- Logging (`debug`, `log_level`, `logging.basicConfig`): there is no output
  to mirror.
- pydantic validation of the model classes (all values are built by the
  splitter itself, so validation never fails). The extra `is_digit` /
  `is_punctuation` keyword arguments that `_special_merge_for_zh_ja` passes
  to `SubString` are ignored by pydantic and have no counterpart.
- budoux HTML processing, the Thai and Traditional Chinese parsers, and the
  budoux CLI and training scripts.
- split-lang does not use jieba; nothing else is imported.

# `gpt_sovits_text::chinese2` parity

Port of `GPT_SoVITS/text/chinese2.py` (GPT-SoVITS `04a44eb`) in
`crates/ports/gpt_sovits_text/src/chinese2.rs`, built on the `g2pw`,
`tone_sandhi`, `zh_normalization` and `symbols` modules of the crate and on
the `jieba_fast` (`posseg.lcut`), `pypinyin` and `cn2an` crates.

```sh
pixi run --manifest-path crates/ports/gpt_sovits_text/pixi.toml data chinese2
pixi run --manifest-path crates/ports/gpt_sovits_text/pixi.toml fixtures chinese2
G2PW_MODEL_DIR=$PWD/crates/ports/gpt_sovits_text/parity/.cache/G2PWModel \
  cargo test -p gpt_sovits_text --test chinese2 --test chinese2_g2pw
```

## Ported scope

| Python | Rust |
|---|---|
| `normalizer = lambda x: cn2an.transform(x, "an2cn")` | `normalizer` (nothing calls it upstream) |
| `pinyin_to_symbol_map` (`opencpop-strict.txt`) | `PINYIN_TO_SYMBOL_MAP` (`data/chinese2/opencpop-strict.txt`, embedded) |
| `is_g2pw = True`, `g2pw = G2PWPinyin(model_dir="GPT_SoVITS/text/G2PWModel", model_source=bert_path, v_to_u=False, neutral_tone_with_five=True)` | `Chinese2::new(model_dir)`: `is_g2pw = true`, `g2pw = Some(G2PWPinyin::new(model_dir, &G2PWPinyinOptions::chinese2()))` |
| `is_g2pw = False` (the commented-out setting) | `Chinese2::without_g2pw()`: no model loaded, `_g2p` takes the pypinyin branch |
| `tone_modifier = ToneSandhi()` | `Chinese2::tone_modifier` |
| `rep_map`, `must_erhua`, `not_erhua` | `REP_MAP` (dict order), `MUST_ERHUA`, `NOT_ERHUA` |
| `replace_punctuation`, `replace_punctuation_with_en`, `replace_consecutive_punctuation`, `text_normalize` | same names (free functions) |
| `g2p`, `_g2p` | `Chinese2::g2p`, `Chinese2::_g2p` (they read the module singletons) |
| `_get_initials_finals`, `_merge_erhua` | same names |

`_g2p` is ported with both branches: the G2PW branch (one batched
`g2pw._g2pw(batch_inputs)` call per `g2p`, `correct_pronunciation`,
`to_initials` / `to_finals_tone3(neutral_tone_with_five=True)` or the raw
item when `pinyin[0]` is not alphabetic) and the pypinyin branch
(`_get_initials_finals`), each with `pre_merge_for_modify`, `modified_tone`
and `_merge_erhua`; then `v_rep_map`, `pinyin_rep_map`, `single_rep_map`,
`pinyin_to_symbol_map` and `word2ph`.

Left out: the `__main__` demo, `jieba_fast.setLogLevel` (the port does not
log).

### Differences in form (not in output)

- Python exceptions are `chinese2::Error` values with the exception name:
  `AssertionError` (with its args: empty for `assert c in punctuation` and
  `assert tone in "12345"`, `[pinyin, seg, raw_pinyin]` for the
  `pinyin_to_symbol_map` assert), `IndexError` (`word[i]`, `v[-1]`,
  `pinyin[0]`, `new_finals[-1][-1]`, `initials[i]`, and the `ToneSandhi`
  errors), `TypeError` (`pinyin[0]` on a `None` G2PW result), `KeyError` /
  `ValueError` from `TextNormalizer.normalize`, and `G2PW` for every
  exception of the G2PW converter.
- `model_dir` is an argument (Python hardcodes `GPT_SoVITS/text/G2PWModel`
  relative to the working directory). `model_source` (`bert_path`) is the
  tokenizer that the `g2pw` port embeds.
- G2PW results can hold `None`, but `g2pw::correct_pronunciation` takes
  `Vec<String>`. `correct_pronunciation_opt` calls it when there is no
  `None` and otherwise runs the same steps on `Option<String>`.
- `pinyin[0].isalpha()` uses a Python 3.11 table
  (`data/chinese2/python_isalpha.txt`); the `\s` of the `g2p` split pattern
  and `str.strip` use the Python whitespace set.
- The pypinyin branch prints `pypinyin结果 [...] [...]` like Python, with a
  simplified list repr (non-ASCII non-printable characters are not
  escaped). Stdout is not checked.

## Fixtures

`gen_fixtures.py` imports `text.chinese2` as GPT-SoVITS does (G2PW on), and
sets `chinese2.is_g2pw = False` around the pypinyin-branch calls. It writes
`crates/ports/gpt_sovits_text/tests/fixtures/chinese2/` (about 2.6 MB):

- `sentences.jsonl` (5607 texts): 268 distinct hand-written cases (every
  `rep_map` key, `...` runs of 2 to 6 dots, mixed punctuation and
  whitespace, Latin, digits in several scripts, dates, times, units, phone
  numbers, `一` / `不` cases, three-tone runs, every `must_erhua` word, other
  erhua words, pairs of readings for about 80 polyphonic characters,
  traditional characters, characters outside `一-龥`, interjections with no
  final (`噷` `呒` `嗯`), one Hangul syllable next to a polyphonic character
  for the `None` result of G2PW), the 147 zh items of
  `parity/corpus/gpt_sovits_zh_en_yue.json`, and 5200 generated texts
  (seed `20261102`): prose from jieba dictionary words, `pp_dict`-word and
  polyphonic-character text, numbers/dates/units, heavy punctuation, jieba
  words that end in `儿`, words with `一` / `不`, all-tone-3 pypinyin
  phrases, traditional text (`simplified_to_traditional`), Latin and
  digits mixed in, and random Han characters. Each record has
  `norm = text_normalize(t)`, `g2p = g2p(norm)` (what `cleaner.clean_text`
  computes for zh), `py = g2p(norm)` in the pypinyin branch, and for the
  hand-written inputs, the corpus and every third generated text
  `raw = g2p(t)` on the unnormalized text, which reaches the error branches.
- `punct.jsonl` (1561 texts): `replace_punctuation`,
  `replace_punctuation_with_en`, `replace_consecutive_punctuation` and
  `normalizer`.
- `units.jsonl`: `_merge_erhua` (3670 cases: every `must_erhua` /
  `not_erhua` word, their suffixes, 300 jieba words in `儿`, with changed
  finals `er1` / `er5`, an empty previous final, a short initial list, more
  finals than characters and fewer finals; `pos` in `n a j nr v x r eng` and
  empty) and `_get_initials_finals` (1175 words).

## Parity

| Fixture set | Cases | Exact matches |
|---|---|---|
| `sentences.jsonl` `text_normalize` | 5607 (1 `KeyError`) | 5607 |
| `sentences.jsonl` `g2p(norm)`, G2PW branch | 5606 (2 `AssertionError`) | 5606 |
| `sentences.jsonl` `g2p(norm)`, pypinyin branch | 5606 (2 `AssertionError`, 8 `IndexError`) | 5606 |
| `sentences.jsonl` `g2p(text)`, G2PW branch, raw text | 2143 (880 `AssertionError`, 620 `IndexError` from G2PW, 3 `TypeError`) | 2143 |
| `punct.jsonl` `replace_punctuation`, `replace_punctuation_with_en`, `replace_consecutive_punctuation`, `normalizer` | 1561 x 4 | 1561 x 4 |
| `units.jsonl` `_merge_erhua` | 3670 | 3670 |
| `units.jsonl` `_get_initials_finals` | 1175 | 1175 |

Results and exceptions are compared exactly (exception name and args). For
exceptions raised inside the G2PW converter only the name is compared; their
messages are checked by `tests/g2pw.rs`. There are no known mismatches, so
there is no exception list.

Rust tests: `tests/chinese2.rs` (no model: string functions, `normalizer`,
`_merge_erhua`, `_get_initials_finals`, `text_normalize`, the pypinyin
branch) and `tests/chinese2_g2pw.rs` (`G2PW_MODEL_DIR` must point at the
extracted `G2PWModel`; skipped otherwise, about 3 minutes in a debug build).

Branches that the fixtures do not reach: `correct_pronunciation` with a
`word_pinyins` shorter than the word (the G2PW result always has one item
per character of the segment, so the slices always fit), and
`initials[i]` past the end in `_merge_erhua` from `_g2p` (both lists have
the same length there; `units.jsonl` reaches it directly).

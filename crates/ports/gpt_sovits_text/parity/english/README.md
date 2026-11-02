# `gpt_sovits_text::english` parity

Port of `GPT_SoVITS/text/english.py` (GPT-SoVITS @ `04a44eb`) to
`crates/ports/gpt_sovits_text/src/english.rs`. It builds on the `nltk`
(`TweetTokenizer`, `pos_tag`), `g2p_en` (`G2p`, `predict`,
`homograph2features`), `wordsegment` and `inflect` ports, and on
`gpt_sovits_text::en_normalization`, `symbols` and `symbols2`.

## Ported scope

All of `english.py`, under the Python names:

- Path constants as embedded file contents: `CMU_DICT` (`cmudict.rep`),
  `CMU_DICT_FAST` (`cmudict-fast.rep`), `CMU_DICT_HOT` (`engdict-hot.rep`),
  `NAMECACHE` (`namedict_cache.pickle`, converted to text).
- `rep_map` (`REP_MAP`), `arpa` (`ARPA`, unused upstream).
- `replace_phs`, `replace_consecutive_punctuation`, `text_normalize`.
- `read_dict`, `read_dict_new`, `hot_reload_hot`, `get_dict`, `get_namedict`.
- `class en_G2p(G2p)` as the struct `en_G2p` (derefs to `g2p_en::G2p`):
  `__init__` (`new`: `wordsegment.load()`, the dictionaries, the deletion of
  `ae ai ar ios hud os`, the `read` / `complex` overrides), `__call__` and
  `qryword`.
- The module-level `_g2p` (`_g2p()`, built once) and `g2p`.

Left out:

- `cache_dict` (writes a pickle). `get_dict` loads `engdict_cache.pickle`
  upstream; that pickle equals `read_dict_new()` (`gen_data.py` asserts it),
  so the port builds the dict from the `.rep` files and then applies
  `hot_reload_hot`, which gives the same result.
- The `print("ph not in symbols: ", ph)` side effect of `replace_phs`: the
  phone is dropped without printing.
- The `__main__` demo (its inputs are in the fixtures).

Quirks kept from Python:

- `text_normalize` joins the `re.escape`d `rep_map` keys, so `[;:：，；]` and
  `["’]` replace only those literal strings. `;`, `:`, `"` and `’` on their
  own reach `normalize`, which removes them.
- `read_dict` raises `IndexError` (line 49 of `cmudict.rep` has no two-space
  separator). The port returns `Error::IndexError("list index out of
  range")`.
- `__call__` uses `self.cmu[word][0]` for single letters and `qryword` uses
  `self.cmu[w][0]` for each alphabetic character of a short OOV word, so a
  non-ASCII letter such as `é`, `ß`, `ñ` or `ﬁ` in unnormalized text gives
  `KeyError` (`Error::KeyError`).
- `g2p` drops `" "`, `<pad>`, `UW`, `</s>`, `<s>`, maps `<unk>` to `UNK`, and
  `replace_phs` drops every phone outside `symbols2.symbols` (for example
  the empty phones of `agronomy` and the `#`, `FOREIGN`, `FRENCH` markers of
  `danglars` in `cmudict-fast.rep`) and maps `'` to `-`.

Python semantics reproduced in the port:

- `str.lower` is `nltk::python::lower` (CPython 3.11 tables, including the
  final sigma rule), `str.strip()` uses `nltk::python::char_isspace`, and
  `str.split(sep)` is `str::split`.
- `str.isalpha` and `str.istitle` use the CPython 3.11 (Unicode 14.0.0)
  tables in `data/english/py_str.json`: `isalpha`, `upper_or_title`
  (`Py_UNICODE_ISUPPER or Py_UNICODE_ISTITLE`) and `lower`. `istitle`
  follows `unicode_istitle_impl`.
- `len(word)`, `word[:-2]` and `for w in word` count code points.
- `re.match(r"^([a-z]+)('s)$", word)` also matches before a trailing `\n`.
- The files are read as Python text mode does (universal newlines).

Possible exceptions are returned as `english::Error` (`KeyError`,
`IndexError`, `TimeoutError` from the tokenizer); `text_normalize` returns
the `en_normalization::Error` of `normalize`. The `IndexError` of
`phones[-1]` in the possessive branch needs a stem whose pronunciation is
empty; no fixture input reaches it.

## Data

`gen_data.py` writes `crates/ports/gpt_sovits_text/data/english/`:

| File | Source |
|---|---|
| `cmudict.rep`, `cmudict-fast.rep`, `engdict-hot.rep` | copied unchanged from `GPT_SoVITS/text` |
| `namedict.tsv` | `namedict_cache.pickle` (19929 entries), one line per entry: key, then one tab-separated field per pronunciation with the phones joined by spaces. The script reads it back and checks it against the pickle. |
| `py_str.json` | CPython 3.11 `str.isalpha`, `istitle` and `islower` code point ranges |

## Fixtures

`gen_fixtures.py` writes `crates/ports/gpt_sovits_text/tests/fixtures/english/`
(seed `20241102`). `tests/english.rs` checks:

| Fixture | Cases | Exact matches |
|---|---|---|
| `dict.json`: `read_dict` (raises), `read_dict_new`, `get_dict`, `en_G2p().cmu`, `get_namedict` (entry count, FNV-1a 64 digest of all entries, and a sample of about 500 entries each), `hot_reload_hot({})`, the `homograph2features` overrides | 5 dictionaries | all |
| `texts.jsonl` `hand`: hand-written cases (rep_map keys, punctuation runs, contractions, possessives for every final-phone class, homographs in several POS contexts, single letters, hot words, deleted abbreviations, names, compounds, predict words, acronyms, numbers, currency, dates, normalization errors, non-ASCII letters, tokenizer specifics, filtered phones) | 273 | 273 |
| `texts.jsonl` `corpus`: every text of `parity/corpus/gpt_sovits_zh_en_yue.json`, its non-CJK spans (as is and stripped), and the `en` segments of `LangSegmenter.getTexts(text, "zh")` and `getTexts(text)` | 210 | 210 |
| `texts.jsonl` `tokens`: seeded single tokens (dictionary words, names in lower, title and upper case, homographs, contractions, possessives, compounds, random letter strings, acronyms, numbers, punctuation) | 1949 | 1949 |
| `texts.jsonl` `sentences`: seeded random sentences of those tokens | 3720 | 3720 |
| `qryword.jsonl`: `_g2p.qryword(word)` | 2829 | 2829 |
| `funcs.jsonl`: `replace_consecutive_punctuation` (433) and `replace_phs` (212) | 645 | 645 |

Each `texts.jsonl` record has `norm` = `text_normalize(i)` and `g2p` =
`g2p(norm)` (what `cleaner.clean_text` does for `en`). The hand, corpus and
token records also have `call` = `_g2p(norm)` (the raw `__call__` output)
and `raw` = `g2p(i)` on the unnormalized text. Results are `{"o": output}` or
`{"e": [exception type, str(args[0])]}`.

A line trace of `english.py` over the generator inputs reaches every line
of `__call__` and `qryword` except the `IndexError` noted above.

There are no known mismatches. `tests/english.rs` still has an explicit
(empty) `KNOWN_MISMATCHES` list.

`predict` (the GRU fallback) can in theory differ from numpy in the last
float bits; see `crates/ports/g2p_en/README.md`. No fixture case differs.

Regenerate:

```sh
pixi run --manifest-path crates/ports/gpt_sovits_text/pixi.toml data english
pixi run --manifest-path crates/ports/gpt_sovits_text/pixi.toml fixtures english
```

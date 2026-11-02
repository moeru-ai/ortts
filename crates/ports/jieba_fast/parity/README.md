# jieba_fast parity and port notes

These scripts make the committed reference outputs of the jieba_fast port from
the Python package. They run in the crate's own pixi workspace
(`../pixi.toml`), which pins Python 3.11, `jieba_fast ==0.53` and
`opencc ==1.4.2`. Run the tasks from the repository root:

```sh
pixi run --manifest-path crates/ports/jieba_fast/pixi.toml fixtures
pixi run --manifest-path crates/ports/jieba_fast/pixi.toml data
```

Or run `pixi run <task>` in `crates/ports/jieba_fast`.

| Task | Script | Writes |
|---|---|---|
| `upstream` | `fetch_upstream.py` | `parity/.cache/GPT-SoVITS`: shallow, sparse checkout of RVC-Boss/GPT-SoVITS at `04a44eb` (`docs/cn`, `tools/i18n/locale`). Skipped when the checkout is already there. Needs `git` on `PATH`. |
| `fixtures` | `gen_fixtures.py` | `tests/fixtures/*.jsonl`. Runs `upstream` first. Takes about 1.5 minutes. |
| `data` | `export_data.py` | `data/dict.txt`, `data/finalseg/*.txt`, `data/posseg/*.txt` |

`gpt_sovits_zh_en_yue.json` is the input of the `corpus` fixture set. It is a
copy of the shared GPT-SoVITS text corpus, so this crate does not read files
outside its directory.

`parity/.cache/` is gitignored. Set `GPT_SOVITS_DIR` to use a GPT-SoVITS
checkout in a different location. The `upstream` task then checks it out
there, or uses it as is when it is already at `04a44eb`.

Python 3.11 is pinned on purpose: the `hand` set enumerates every
`str.isspace()` character, which depends on the Unicode version of Python.

`jieba_fast` is published as a source distribution only, so `pixi install`
compiles its C extension. This needs a C compiler (on Windows, the MSVC build
tools).

## Fixture sets

`tests/parity.rs` compares `lcut`, `lcut_for_search` and `posseg::lcut`
(word and flag), all with `HMM=True`, exactly:

| Set | Inputs | Cases | lcut | cut_for_search | posseg |
|---|---|---|---|---|---|
| `hand` | edge cases: whitespace, regex classes, Han range limits, non-BMP, DAG with more than 12 ends, HMM new words (`仳涏`, `忐睧儿`) | 348 | 348 | 348 | 348 |
| `corpus` | `parity/gpt_sovits_zh_en_yue.json` | 155 | 155 | 155 | 155 |
| `prose` | GPT-SoVITS Chinese docs and zh locale files, also split into sentences | 1276 | 1276 | 1276 | 1276 |
| `words` | random dictionary-word sentences, also in traditional characters (OpenCC) | 5400 | 5400 | 5400 | 5400 |
| `hmm` | runs of single characters, names, rare and out-of-table characters, CJK extensions | 3999 | 3999 | 3999 | 3999 |
| `mixed` | Chinese with English, digits, units, percentages, symbols | 2500 | 2500 | 2500 | 2500 |
| `long` | sentences of several hundred characters | 180 | 180 | 180 | 180 |

There are no mismatches, so the exception list in `tests/parity.rs` is empty.

## Data

`data/dict.txt` is the package dictionary, copied unmodified. The HMM tables
are Python dict literals (`finalseg/prob_*.py`, `posseg/*.py`), so
`export_data.py` converts them to text files under `data/finalseg/` and
`data/posseg/`. Floats are written with `repr()`, which round-trips exactly.
Dict order is kept. All data is embedded with `include_str!` and parsed on
first use.

## Port notes

jieba_fast is jieba with the hot loops moved to a C extension
(`_jieba_fast_functions_py3`). Where the C code and the pure Python code
behave differently, the port follows the code that jieba_fast actually runs.

`jieba_fast/__init__.py` (crate root):

- `Tokenizer` with `FREQ`, `total`, `gen_pfdict`, `get_DAG`, `calc`, the
  private `__cut_DAG` and `__cut_DAG_NO_HMM`, `cut` (`cut_all=False`, with and
  without HMM), `lcut`, `cut_for_search`, `lcut_for_search`.
- The default tokenizer `dt` and the module functions `cut`, `lcut`,
  `cut_for_search`, `lcut_for_search`, `get_FREQ`.
- The regexes `re_han_default`, `re_skip_default` and `re_eng` as character
  predicates. `\s` is Python's `str.isspace()` set (`is_python_space`).

`_jieba_fast_functions_py3` (`_jieba_fast_functions`), from
`source/jieba_fast_functions_wrap_py3.i`:

- `_get_DAG_and_calc`, used by `Tokenizer.cut`. It keeps at most 12 word ends
  per start position, and a route tie keeps the first (smallest) end. The pure
  Python `calc` that it replaces picks the largest end.
- `_viterbi`, used by `finalseg`. Ties go to the first previous state and to
  the final `E` state.

`jieba_fast/finalseg` (`finalseg`): `cut`, `viterbi` (through the C
`_viterbi`), `re_han`, `re_skip`, and the `start_P`, `trans_P`, `emit_P`
tables.

`jieba_fast/posseg` (`posseg`): `pair` (`Pair`), `POSTokenizer` with
`load_word_tag`, `__cut`, `__cut_detail`, `__cut_DAG`, `__cut_DAG_NO_HMM`,
`__cut_internal`, `cut`, `lcut`; the default `dt` and the module functions
`cut` and `lcut`. `posseg/viterbi.py` (`posseg::viterbi`) is the pure Python
viterbi, which posseg uses. posseg uses the pure Python `get_DAG` and `calc`,
not the C route, so ties in its route pick the largest end.

Left out because GPT-SoVITS calls none of them: `cut_all=True`, user
dictionaries (`load_userdict`, `add_word`, `del_word`, `suggest_freq`,
`Force_Split_Words`), `tokenize`, `set_dictionary` and other dictionaries, the
dictionary cache file, parallel mode (`enable_parallel`) and
`jieba_fast.analyse`. Python's `logging` output is also left out.

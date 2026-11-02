# tojyutping parity tooling

Python reference scripts for this crate. They run in the crate's own pixi
workspace (`../pixi.toml`, locked in `../pixi.lock`): Python 3.11 and
ToJyutping 3.2.0 only.

From the repository root:

```sh
pixi run --manifest-path crates/ports/tojyutping/pixi.toml fixtures
pixi run --manifest-path crates/ports/tojyutping/pixi.toml data
```

Or run `pixi run fixtures` / `pixi run data` from `crates/ports/tojyutping/`.

| Task | Script | Writes |
|---|---|---|
| `fixtures` | `gen_fixtures.py` | `tests/fixtures/constants.json`, `chars.json`, `words.json`, `sentences.json` |
| `data` | `copy_trie.py` | `data/trie.txt` (byte-for-byte copy from the ToJyutping 3.2.0 wheel) |

Inputs:

- `corpus/gpt_sovits_zh_en_yue.json`: the GPT-SoVITS sample sentences. It is
  kept as a copy in the crate, so this crate does not depend on files
  outside of it.
- Seeds are fixed in `gen_fixtures.py`. The output is deterministic.

Nothing is downloaded, so there are no download tasks, no `.cache/` contents
and no environment variables. `parity/.cache/` is gitignored for future use.

Python 3.11 is pinned because `constants.json` lists `str.isspace()` over all
code points, and that depends on the Unicode version of the interpreter.

## Results

| Fixture | Cases | Exact matches |
|---|---|---|
| `chars.json`: `get_jyutping_list` on every character in the trie | 30206 | 30206 |
| `words.json`: seeded sample of multi-character trie words | 12000 | 12000 |
| `sentences.json`: `get_jyutping_list` | 1724 | 1724 |
| `sentences.json`: `get_jyutping` | 1724 | 1724 |
| `sentences.json`: `get_jyutping_text` | 1724 | 1724 |
| `constants.json`: `punct_dict`, punctuation sets, `str.isspace` | 1 | 1 |

The sentences are hand-written edge cases for every branch of
`format_romanization_text`, the GPT-SoVITS corpus, and 1500 seeded random
mixes of trie words, word prefixes, single characters (simplified and
traditional), Han characters that are not in the trie, punctuation, digits
(including full-width and mathematical digits), Latin, Unicode spaces and
control characters.

There are no known mismatches.

## Module mapping

| Python | Rust |
|---|---|
| `ToJyutping.get_jyutping_list` / `get_jyutping` / `get_jyutping_text` (`__init__.py`) | `tojyutping::{get_jyutping_list, get_jyutping, get_jyutping_text}` |
| `ToJyutping.ToJyutping` instance, `JyutpingConverter.get_jyutping_list` / `get_jyutping` / `get_jyutping_text` | `tojyutping::to_jyutping::{ToJyutping, JyutpingConverter}` |
| `Trie.py`: `Node`, `set_default_node`, the module-level `trie.txt` decoder, `root`, `Trie.get`, `Trie.get_value` | `tojyutping::trie` (`decode`, `root()`, `Trie::get`) |
| `Jyutping.py`: `onset` / `nucleus` / `rhyme` / `coda`, `Jyutping(int)`, `JyutpingList.jyutping` | `tojyutping::jyutping` |
| `utils.py`: `punct_dict`, bracket / punctuation / `minus_signs` / `decimal_seps` / `digits` sets, `format_romanization_text` | `tojyutping::utils` |

`data/trie.txt` is the unmodified `ToJyutping/trie.txt` from the 3.2.0 wheel.
It is embedded with `include_str!` and decoded on first use with the same
algorithm as the Python module (`OnceLock`).

Not ported, in full:

- `get_jyutping_candidates` / `Trie.get_all`, `EdgeLengthToItems`,
  `flat_dedupe`.
- IPA: `get_ipa_list`, `get_ipa`, `get_ipa_text`, `get_ipa_candidates`,
  `format_ipa_text`, `Jyutping.ipa`, `jyutping2ipa`.
- `customize`, `CustomizableTrie`, `parse_jyutping`, `extract_alnum`,
  `dedupe` and the string constructor `Jyutping(str)` (with its regex).
- `g2p`, `g2p_with_puncts`, `Jyutping.g2p`, `PhonemesList`, `CustomList`.

## Behaviour kept from Python

- `Trie.get` is a greedy longest match without backtracking. A
  multi-character match maps its syllables to characters by position.
- A single character can have more than one syllable (45 entries such as
  `吋` → `jing1 cyun3`, encoded with `~` in `trie.txt`). `get_jyutping_list`
  then returns the syllables joined by a space for that character.
- Only the first value (`v[0]`) of a node is used.
- `get_jyutping_text` splits the input on the control characters
  `[\0-\x1f\x80-\x9f]` like the Python `re.sub` and keeps them. Whitespace
  checks use Python `str.isspace()` (`utils::py_isspace`).

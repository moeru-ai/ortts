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

# jieba_fast parity tooling

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

# split_lang parity tooling

Python reference scripts for this crate. They run in the crate's own pixi
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

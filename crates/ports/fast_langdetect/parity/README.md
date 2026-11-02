# fast_langdetect parity tooling

Python scripts that generate the golden fixtures of this crate. They run in
the crate's own pixi workspace (`../pixi.toml`, Python 3.11, fast-langdetect
1.1.0, fasttext-predict 0.9.2.4).

## Scripts

- `fetch_model.py`: downloads the fastText `lid.176.bin` model (131 MB) to
  `parity/.cache/lid.176.bin` and checks its MD5. It does nothing when the
  file is already present.
- `gen_fixtures.py`: runs fast-langdetect with that model and writes
  `tests/fixtures/detect.jsonl`.
- `corpus/gpt_sovits_zh_en_yue.json`: the GPT-SoVITS sentence corpus used as
  part of the fixture inputs.

## Tasks

Run from the repository root:

```sh
pixi run --manifest-path crates/ports/fast_langdetect/pixi.toml lid-model  # download the model
pixi run --manifest-path crates/ports/fast_langdetect/pixi.toml fixtures   # regenerate tests/fixtures
```

`fixtures` runs `lid-model` first. The crate has no generated files under
`data/`, so there is no `data` task.

## Cache

Downloads go to `parity/.cache/` (gitignored). `FAST_LANGDETECT_MODEL`
overrides the model path for both scripts. The file must be named
`lid.176.bin`, because fast-langdetect looks for that name in its
`cache_dir`. The Rust parity tests read the same variable:

```sh
FAST_LANGDETECT_MODEL=crates/ports/fast_langdetect/parity/.cache/lid.176.bin \
  cargo test -p fast_langdetect --release
```

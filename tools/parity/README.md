# Parity tooling

Python reference scripts for the ports in `crates/ports`. See
`crates/ports/README.md` for the conventions.

- `gpt_sovits_env.py`: import helper that puts the upstream GPT-SoVITS checkout
  on `sys.path` and points its hardcoded relative paths at `assets/`.
- `<crate>/`: fixture generators and data converters for each port crate.
- `gpt_sovits/<module>/`: fixture generators for each `gpt_sovits_text`
  module. `gpt_sovits/e2e/` holds the end-to-end generator used by
  `crates/ports/gpt_sovits_text/tests/e2e.rs`.
- `gpt_sovits/python_ref.py`: quick end-to-end dump of upstream phone ids for
  a corpus file, for ad hoc comparisons.
- `corpus/`: sentence corpora for the end-to-end comparison.
- `assets/` (gitignored): `G2PWModel/` (the G2PW ONNX model and its
  dictionaries, from `G2PWModel_1.1.zip`), `fast_langdetect/lid.176.bin`, and
  the `chinese-roberta-wwm-ext-large` tokenizer under `run/`. Rust tests read
  `G2PW_MODEL_DIR` and `FAST_LANGDETECT_MODEL` and skip the model tests when
  they are unset.

Set up the environment once with `pixi install -e parity`.

# nltk parity tooling

Python scripts that generate the golden fixtures and data tables of this
crate. They run in the crate's own pixi workspace (`../pixi.toml`, lock file
`../pixi.lock`), which pins the Python 3.11 reference environment. Nothing
here depends on files outside this crate.

## Tasks

Run them from the repository root (or use `--manifest-path pixi.toml` from the
crate directory):

```sh
pixi run --manifest-path crates/ports/nltk/pixi.toml data       # data/*.json
pixi run --manifest-path crates/ports/nltk/pixi.toml fixtures   # tests/fixtures/*
```

| Task | What it does |
|---|---|
| `data` | Runs `gen_data.py`. Depends on `nltk-data`. Takes a few seconds. |
| `fixtures` | Runs `gen_fixtures.py`. Depends on `fetch`. Takes about 2 minutes (most of it is `corpus.jsonl`). |
| `fetch` | Runs every download task below. |
| `upstream` | Downloads the GPT-SoVITS source tree at commit `04a44eb` (GitHub archive) to `.cache/GPT-SoVITS`. |
| `g2pw` | Downloads `G2PWModel_1.1.zip` (589 MB, the URL that upstream `g2pw/onnx_api.py` uses) and extracts it to `.cache/G2PWModel`. |
| `bert-tokenizer` | Downloads the tokenizer files of `hfl/chinese-roberta-wwm-ext-large` (fixed revision) to `.cache/chinese-roberta-wwm-ext-large`. |
| `lid-model` | Downloads fastText `lid.176.bin` (131 MB) to `.cache/fast_langdetect/lid.176.bin`. |
| `nltk-data` | Downloads the nltk_data packages `averaged_perceptron_tagger_eng`, `averaged_perceptron_tagger` and `cmudict` to `.cache/nltk_data`. |

The download tasks are in `fetch.py`. They skip files that are present, check
SHA-256 sums where the source is fixed, and write only into `.cache/`
(gitignored). After the first run the generators work offline.

## Scripts

- `gen_data.py`: writes `data/html_name2codepoint.json`, `data/python_str.json`
  and `data/regex_sets.json`, and checks that the committed
  `data/averaged_perceptron_tagger_eng.*.json` match nltk_data.
- `gen_fixtures.py`: writes every file in `tests/fixtures/`. `corpus.jsonl`
  runs the upstream GPT-SoVITS text frontend over
  `corpus/gpt_sovits_zh_en_yue.json` and records each text that `english.py`
  passes to `word_tokenize`.
- `gpt_sovits_env.py`: import helper for the upstream frontend. It puts the
  upstream tree on `sys.path`, and changes the working directory to
  `.cache/run`, where it links the G2PW model and the tokenizer to the
  relative paths that upstream hardcodes.
- `paths.py`: all crate-local paths.
- `fetch.py`: the download tasks.
- `corpus/gpt_sovits_zh_en_yue.json`: the sentence corpus for `corpus.jsonl`
  (a copy of the GPT-SoVITS sample sentence corpus, kept in the crate so it
  does not depend on files outside of it).

## Environment variables

All defaults are inside `parity/.cache/`. These variables override them:

| Variable | Default |
|---|---|
| `NLTK_PARITY_CACHE` | `parity/.cache` (the root for the defaults below) |
| `GPT_SOVITS_ROOT` | `<cache>/GPT-SoVITS` (an existing git checkout at `04a44eb` also works) |
| `G2PW_MODEL_DIR` | `<cache>/G2PWModel` |
| `BERT_TOKENIZER_DIR` | `<cache>/chinese-roberta-wwm-ext-large` |
| `FAST_LANGDETECT_MODEL` | `<cache>/fast_langdetect/lid.176.bin` |
| `NLTK_DATA` | `<cache>/nltk_data` |

`G2PW_MODEL_DIR` and `FAST_LANGDETECT_MODEL` have the same names as the
variables that the Rust tests of the `gpt_sovits_text` crate read. After
`pixi run ... fetch`, you can point those tests at the files in
`crates/ports/nltk/parity/.cache/`. The Rust tests of this crate do not need
any of these files.

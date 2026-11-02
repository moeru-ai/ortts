# g2p_en parity tooling

These scripts generate the golden fixtures in `tests/fixtures` and the files
in `data/` from the pinned Python packages. The crate has its own pixi
workspace (`pixi.toml`, `pixi.lock`) with only the packages that the scripts
use. Run the tasks from the crate directory, or from anywhere with
`--manifest-path crates/ports/g2p_en/pixi.toml`:

```sh
pixi run fixtures   # regenerate tests/fixtures (runs upstream and nltk-data first)
pixi run data       # copy data/ from the pinned g2p_en package
```

| Task | Script | Output |
|---|---|---|
| `upstream` | `fetch.py upstream` | GPT-SoVITS source at commit `04a44eb` in `parity/.cache/GPT-SoVITS` |
| `nltk-data` | `fetch.py nltk-data` | nltk `averaged_perceptron_tagger`, `averaged_perceptron_tagger_eng` and `cmudict` in `parity/.cache/nltk_data` |
| `fixtures` | `gen_fixtures.py` | `tests/fixtures/predict.jsonl`, `homograph2features.json`, `vocab.json` |
| `data` | `copy_data.py` | `data/checkpoint20.npz`, `data/homographs.en` (copied from g2p_en 2.1.0, SHA-1 checked) |

The download tasks do nothing when their output is already present. Delete
`parity/.cache` to download again.

`gen_fixtures.py` imports GPT-SoVITS `text.english` to record the words that
`en_G2p` really sends to `predict`. It reads them for the sentences in
`gen_fixtures.py` and in `corpus/gpt_sovits_zh_en_yue.json` (a copy of the
shared parity corpus). `gpt_sovits_env.py` puts the upstream checkout on
`sys.path` and points nltk at the cache.

## Environment overrides

| Variable | Default | Used by |
|---|---|---|
| `GPT_SOVITS_ROOT` | `parity/.cache/GPT-SoVITS` | `gpt_sovits_env.py`, `fetch.py upstream` (it does not overwrite an existing directory that this variable names) |
| `NLTK_DATA` | `parity/.cache/nltk_data` | `gpt_sovits_env.py`, `fetch.py nltk-data` |

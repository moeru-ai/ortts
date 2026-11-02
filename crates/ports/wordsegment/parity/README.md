# wordsegment parity tooling

These scripts generate the golden fixtures, the data files and `LICENSE` of
this crate from the Python package wordsegment 1.3.1. They run in the small
pixi workspace of this crate (`pixi.toml`, `pixi.lock`). Run the tasks from
the repository root:

```sh
pixi run --manifest-path crates/ports/wordsegment/pixi.toml fixtures
pixi run --manifest-path crates/ports/wordsegment/pixi.toml data
pixi run --manifest-path crates/ports/wordsegment/pixi.toml license
```

Or run `pixi run <task>` in `crates/ports/wordsegment/`.

| Task | Script | Writes |
|---|---|---|
| `fixtures` | `gen_fixtures.py` | `tests/fixtures/{segment,score,divide}.jsonl`, `tests/fixtures/meta.json` (about 9 minutes) |
| `data` | `copy_data.py` | `data/{unigrams,bigrams,words}.txt`, copied unmodified from the wordsegment 1.3.1 package |
| `license` | `make_license.py` | `LICENSE` (upstream copyright and the Apache-2.0 text) |
| `apache-license` | `fetch.py` | `parity/.cache/LICENSE-2.0.txt`, downloaded from apache.org; `license` runs it first |

`fetch.py` skips the download when the file is present with the expected
SHA-256. The cache directory is `parity/.cache/` (gitignored). Set
`WORDSEGMENT_PARITY_CACHE` to use another directory.

The fixtures use fixed seeds, so the same environment gives byte-identical
output.

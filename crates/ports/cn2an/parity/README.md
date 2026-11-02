# cn2an parity tooling

Python scripts that generate the reference data for this crate. They run in
the crate-local pixi workspace (`../pixi.toml`): Python 3.11, cn2an 0.5.24
and proces 0.1.7.

| Script | Task | Writes |
|---|---|---|
| `generate_fixtures.py` | `fixtures` | `tests/fixtures/transform_an2cn.jsonl`, `tests/fixtures/transform_an2cn_direct.jsonl`, `tests/fixtures/an2cn.jsonl` |
| `export_data.py` | `data` | `data/t2s.tsv` (proces `T2S_DICT`), `data/python_decimal_ranges.txt` (Python 3.11 `re` `\d` ranges) |

Run from the repository root:

```sh
pixi run --manifest-path crates/ports/cn2an/pixi.toml fixtures
pixi run --manifest-path crates/ports/cn2an/pixi.toml data
```

Or, from `crates/ports/cn2an`: `pixi run fixtures` and `pixi run data`.

The scripts find the crate from their own path, so the working directory does
not matter. The inputs use fixed seeds, so the output is deterministic. No
downloads are needed and no environment variables are read. `parity/.cache/`
is gitignored and is kept for local caches.

Python must stay on 3.11: `python_decimal_ranges.txt` depends on its Unicode
14.0 data. The first line of that file also records the exact interpreter
version (3.11.18 in `../pixi.lock`), so a different patch release changes
only that comment line.

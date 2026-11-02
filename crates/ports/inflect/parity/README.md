# inflect parity tooling

This directory holds the Python script that generates the golden fixtures and
the data table of this crate. The crate has its own pixi workspace
(`../pixi.toml`, `../pixi.lock`). It pins Python 3.11.\*, inflect 7.6.0 and
typeguard 4.6.0 from conda-forge. Nothing outside this crate is used.

## Scripts

| Script | Writes |
|---|---|
| `gen_fixtures.py --only data` | `data/decimal_digits.tsv`: the Unicode decimal digit ranges (`str.isdecimal`) of the pinned Python 3.11 (Unicode 14.0.0) |
| `gen_fixtures.py --only fixtures` | `tests/fixtures/range.json` and `tests/fixtures/cases.jsonl` (seeded, deterministic) |

Without `--only`, the script writes both. The script finds the crate from its
own location, so the current directory does not matter.

## Tasks

From the repository root:

```sh
pixi run --manifest-path crates/ports/inflect/pixi.toml fixtures
pixi run --manifest-path crates/ports/inflect/pixi.toml data
```

From `crates/ports/inflect`, `pixi run fixtures` and `pixi run data` do the
same. Each task takes a few seconds.

There are no download tasks and no environment variables. The generator needs
no upstream sources or model files. `parity/.cache/` is gitignored for
consistency with the other port crates, but this crate does not use it.

After you regenerate, `git status` must show no changes to `tests/fixtures/`
or `data/`. Then run `cargo test -p inflect --release`.

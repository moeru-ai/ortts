# pypinyin parity tooling

Python scripts that write the golden fixtures and the generated data of this
crate. They run in the crate's own pixi workspace (`../pixi.toml`), which
holds only Python 3.11 and pypinyin 0.55.0 from conda-forge.

| Script | Writes |
|---|---|
| `gen_fixtures.py` | `tests/fixtures/*.json` (seed `20250723`) |
| `copy_dicts.py` | `data/pinyin_dict.json`, `data/phrases_dict.json` (copied unchanged from the installed pypinyin package) |
| `gen_unicode_digits.py` | `src/_re_digit.rs` (the Python 3.11 / Unicode 14.0.0 `\d` table) |

## Tasks

Run from the repository root (or use `pixi run <task>` in the crate
directory):

```sh
pixi run --manifest-path crates/ports/pypinyin/pixi.toml fixtures
pixi run --manifest-path crates/ports/pypinyin/pixi.toml data
```

| Task | Runs |
|---|---|
| `fixtures` | `gen_fixtures.py` |
| `dicts` | `copy_dicts.py` |
| `unicode-digits` | `gen_unicode_digits.py` |
| `data` | `dicts` and `unicode-digits` |

The scripts need no downloads, no upstream checkout and no environment
variables. They find the crate from their own location. `parity/.cache/` is
gitignored for local scratch files, but no task uses it.

Keep Python at 3.11: a different minor version ships different Unicode data
and changes `src/_re_digit.rs`. The file header also records the Python patch
version, which `pixi.lock` pins.

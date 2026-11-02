# Ports of Python packages

Rust ports of the GPT-SoVITS text frontend (`gpt_sovits_text`) and the Python
packages it depends on. The goal is the same output as Python, so each crate
mirrors the package it ports: the crate, module, function and struct names are
the Python names, and each item's doc comment names the Python item it ports.
Only what GPT-SoVITS uses is ported; each crate README lists the scope.

| Crate | Ports | License |
|---|---|---|
| `pypinyin` | pypinyin 0.55.0 | MIT |
| `cn2an` | cn2an 0.5.24 | MIT |
| `wordsegment` | wordsegment 1.3.1 | Apache-2.0 |
| `inflect` | inflect 7.6.0 | MIT |
| `nltk` | nltk 3.10.4 | Apache-2.0 |
| `tojyutping` | ToJyutping 3.2.0 | BSD-2-Clause |
| `g2p_en` | g2p_en 2.1.0 | Apache-2.0 |
| `split_lang` | split-lang 2.1.1, budoux 0.10.2 | MIT, Apache-2.0 |
| `fast_langdetect` | fast-langdetect 1.1.0, fastText 0.9.2 | MIT |
| `jieba_fast` | jieba_fast 0.53 | MIT |
| `gpt_sovits_text` | GPT-SoVITS `GPT_SoVITS/text` at `04a44eb` | MIT, Apache-2.0, BSD-3-Clause |

## Crate layout

```
crates/ports/<crate>/
  src/, tests/          the port and its parity tests
  data/                 upstream data files, embedded in the crate
  tests/fixtures/       golden fixtures generated from Python
  pixi.toml, pixi.lock  Python 3.11 and the pinned packages the scripts need
  parity/               fixture and data generators, maintainer notes
```

Regenerate a crate's fixtures with:

```sh
pixi run --manifest-path crates/ports/<crate>/pixi.toml fixtures
```

Large downloads (models, the upstream GPT-SoVITS checkout) go to the
gitignored `parity/.cache/`. Tests that need a model read its path from an
environment variable and skip when it is unset.

## Rules

- Fixtures are generated, never edited by hand.
- Tests compare with Python exactly. A difference that cannot be fixed goes
  into an explicit exception list with the reason in `parity/README.md`.
- Never special-case fixture inputs or loosen an assertion to pass a test.
- Each crate keeps the upstream license (`LICENSE`, plus `NOTICE` for
  Apache-2.0).

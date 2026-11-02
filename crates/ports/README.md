# Ports of Python packages

The crates in this directory are Rust ports of the Python packages that the
GPT-SoVITS text frontend depends on, plus the frontend itself
(`gpt_sovits_text`). The goal is output parity with the Python originals, so
each crate mirrors the Python package it ports:

- The crate name is the Python package name (`pypinyin`, `g2p_en`, ...).
- Modules mirror Python modules (`pypinyin/contrib/tone_convert.py` becomes
  `pypinyin::contrib::tone_convert`). Rust requires snake_case, so
  `LangSegmenter` becomes `lang_segmenter`. Keep upstream spelling otherwise,
  including typos such as `text_normlization`.
- Functions keep their Python names. Classes become structs with the same
  name (`G2p`, `TextNormalizer`, `ToneSandhi`). Keyword arguments become an
  options struct whose fields use the Python parameter names.
- Each Rust item has a doc comment that names the Python item it ports.

Scope is what GPT-SoVITS actually calls, directly or through the call graph.
Each crate README lists what is ported and what is left out.

## Reference versions

| Crate | Python package | Version | License |
|---|---|---|---|
| `pypinyin` | pypinyin | 0.55.0 | MIT |
| `cn2an` | cn2an | 0.5.24 | MIT |
| `wordsegment` | wordsegment | 1.3.1 | Apache-2.0 |
| `inflect` | inflect | 7.6.0 | MIT |
| `nltk` | nltk | 3.10.4 | Apache-2.0 |
| `tojyutping` | ToJyutping | 3.2.0 | BSD-2-Clause |
| `g2p_en` | g2p_en | 2.1.0 | Apache-2.0 |
| `split_lang` | split-lang | 2.1.1 | MIT |
| `split_lang` (vendored part) | budoux | 0.10.2 | Apache-2.0 |
| `fast_langdetect` | fast-langdetect | 1.1.0 | MIT |
| `jieba_fast` | jieba_fast | 0.53 (exact port of `cut`, `cut_for_search`, `posseg`) | MIT |
| `gpt_sovits_text` | GPT-SoVITS `GPT_SoVITS/text` | commit `04a44eb` | MIT, with Apache-2.0 parts (PaddleSpeech, g2pW, ESPnet) and BSD-3-Clause parts (MeCab) |

`split_lang` also contains the part of budoux 0.10.2 that split-lang calls
(the parser and its Japanese and Simplified Chinese models), so it has an
Apache-2.0 `NOTICE` next to the MIT license of split-lang.

The Python sources are installed in the `parity` pixi environment:
`.pixi/envs/parity/lib/python3.11/site-packages/<package>`. The upstream
GPT-SoVITS checkout is at `$GPT_SOVITS_ROOT` (default
`~/Git/github.com/RVC-Boss/GPT-SoVITS`, commit `04a44eb`).

## Data files

Data files that the Python package ships (dictionaries, model weights) are
committed under `<crate>/data/` and embedded with `include_str!` or
`include_bytes!`. Keep the upstream file format when Rust can read it. When it
cannot (pickle, Python dict literals), convert it with a script in
`tools/parity/<crate>/` and commit that script, so the data can be
regenerated from the pinned package version.

Large runtime models that the Python code downloads (the G2PW ONNX model, the
fasttext `lid.176.bin` language ID model) are
not committed. The Rust API takes a path to them. Tests that need them read the
path from an environment variable and are skipped when it is not set.

## Licenses

A port is a derivative work of the package it ports. Each crate keeps the
upstream license:

- `LICENSE` contains the upstream license text with the upstream copyright.
- Apache-2.0 crates also have a `NOTICE` that names the upstream project and
  states that the code was translated to Rust.
- `Cargo.toml` `license` uses the upstream SPDX identifier.

## Parity tests

Parity is checked against golden fixtures that the Python package produces:

1. A generator script in `tools/parity/<crate>/` runs the pinned Python
   package and writes fixtures to `<crate>/tests/fixtures/`. Run it with
   `pixi run -e parity python tools/parity/<crate>/<script>.py`.
2. Rust tests in `<crate>/tests/` load the fixtures and assert exact equality.
   Rust tests never need Python.

Rules:

- Fixtures are generated, never edited by hand.
- Inputs must exercise every branch of the ported code. Combine hand-written
  edge cases with bulk inputs (dictionary entries, random samples with a fixed
  seed). Keep each crate's fixtures under a few MB.
- Never special-case fixture inputs in the implementation, and never loosen an
  assertion to make a test pass. If a mismatch cannot be fixed (for example a
  floating point tie), list it in the crate README with the reason, and make
  the test check the documented exception list explicitly.

## Status

The crates land in separate pull requests. The leaf crates have no
dependencies on each other; `split_lang` depends on `fast_langdetect`, and
`gpt_sovits_text` depends on all of them. Each crate README has its scope and
parity table.

Not ported:

- `GPT_SoVITS/text/korean.py` and `chinese.py` (v1 Chinese).
- pyopenjtalk is replaced by jpreprocess; the known differences are documented
  with the Japanese frontend in `gpt_sovits_text`.
- The BERT model (`chinese-roberta-wwm-ext-large`) is not part of the ports;
  `gpt_sovits_text` takes it through a trait.

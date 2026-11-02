# GPT-SoVITS frontend: integration fixtures

Generators for `gpt_sovits_text::cleaner`,
`gpt_sovits_text::tts_infer_pack::text_preprocessor` and the end-to-end
`TextPreprocessor` tests. Results are in `crates/ports/gpt_sovits_text/parity/README.md`.

| Script | Writes (`crates/ports/gpt_sovits_text/tests/fixtures/`) | Rust test |
|---|---|---|
| `parity/cleaner/gen_fixtures.py` | `cleaner/` | `tests/cleaner.rs` |
| `parity/text_preprocessor/gen_fixtures.py` | `text_preprocessor/` | `tests/text_preprocessor.rs` |
| `parity/e2e/gen_fixtures.py` | `e2e/` | `tests/e2e.rs` |

All three import `e2e_common.py`, which:

- imports upstream through `parity/gpt_sovits_env.py`, from the cache that
  the `assets` pixi task fills (`parity/.cache`);
- points fast-langdetect at `parity/.cache/fast_langdetect/lid.176.bin`;
- wraps `clean_text` so that it raises `NotPortedError` where upstream would
  import `text.korean` (v2 `ko`) or `text.chinese` (v1 `zh`). The Rust port
  returns `cleaner::Error::NotPorted` there. (In the crate's pixi environment
  `text.korean` does not even import: `jamo` is not installed.)
- defines `FakeBert`, a deterministic stand-in for
  `chinese-roberta-wwm-ext-large`: hidden state `[t, d]` of
  `hidden_states[-3]` is `((id_t * 1009 + t * 31 + d * 17) % 65521) / 64`
  (exact in `float32`), and the other 24 hidden states are constants. The
  Rust copy is `crates/ports/gpt_sovits_text/tests/common/e2e_common.rs`.
- summarizes a `[1024, n]` feature as `[shape, checksum]`, with
  `checksum = sum(bert * 64 * (1 + flat_index % 997))`, an exact integer.
- generates the edge texts and the random multi-sentence paragraphs.

The tokenizer is `AutoTokenizer.from_pretrained` of
`parity/.cache/run/GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large`, whose
`tokenizer.json` is the one embedded in the crate.

Run:

```sh
pixi run --manifest-path crates/ports/gpt_sovits_text/pixi.toml fixtures cleaner
pixi run --manifest-path crates/ports/gpt_sovits_text/pixi.toml fixtures text_preprocessor
pixi run --manifest-path crates/ports/gpt_sovits_text/pixi.toml fixtures e2e
```

The e2e generator takes about a minute on an M-series Mac. The fixtures of
the three sets are about 3 MB together.

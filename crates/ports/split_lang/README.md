# split_lang

Rust port of [split-lang](https://github.com/DoodleBears/split-lang) 2.1.1,
limited to what the GPT-SoVITS text frontend uses, plus the part of
[budoux](https://github.com/google/budoux) 0.10.2 that split-lang calls.
Language detection uses the [`fast_langdetect`](../fast_langdetect) crate.

```rust
use split_lang::split::splitter::{LangSplitter, LangSplitterOptions, to_owned_map};

let mut splitter = LangSplitter::new(LangSplitterOptions {
  lang_map: to_owned_map(&[("zh", "zh"), ("ja", "ja"), ("en", "en")]),
  ..LangSplitterOptions::default()
});
splitter.merge_across_digit = false; // as GPT-SoVITS runs it
let substrings = splitter.split_by_lang("MyGO?,你也喜欢まいご吗？")?;
```

## Ported

- `LangSplitter::split_by_lang` and every splitter method it reaches, with all `__init__` options
- `config` (lang maps), `model` (`SubString`, `SubStringSection`), `split::utils`
- `detect_lang::detector` (fast-langdetect calls, optional per-splitter detector)
- budoux `Parser.parse` with the default Japanese and Simplified Chinese models
- Python exceptions as `SplitLangError`

## Not ported

- Logging and pydantic validation
- budoux HTML processing, Thai and Traditional Chinese parsers, CLI and training scripts

## Parity

Fixtures come from the pinned Python packages:
`pixi run --manifest-path crates/ports/split_lang/pixi.toml fixtures` (it also
downloads `lid.176.bin`). The `split_by_lang` test needs
`FAST_LANGDETECT_MODEL=crates/ports/split_lang/parity/.cache/fast_langdetect/lid.176.bin`
and is skipped without it. All cases match exactly: 3571/3571 (`gsv`),
3571/3571 (`gsv_no_digit`), 715/715 (`default`) and 4104/4104 budoux texts.
See [parity/README.md](parity/README.md) for details.

## License

split-lang is MIT, Copyright (c) 2024 DoodleBear. The budoux parts are
Apache-2.0, Copyright 2021 Google LLC. See `NOTICE`, `LICENSE` and `LICENSE-APACHE`.

# fast_langdetect

Rust port of [fast-langdetect](https://github.com/LlmKira/fast-langdetect)
1.1.0, limited to what the GPT-SoVITS text frontend uses through split-lang.
It includes a pure Rust port of the fasttext-predict 0.9.2.4 prediction path.
No C or C++ code is linked.

```rust
use fast_langdetect::{DetectOptions, LangDetectConfig, LangDetector, Model};

let config = LangDetectConfig { cache_dir: "pretrained_models/fast_langdetect".into(), ..Default::default() };
fast_langdetect::set_default_detector(LangDetector::new(config));
let res = fast_langdetect::detect("你好", &DetectOptions { model: Some(Model::Full), k: 5, threshold: 0.01 }, None)?;
```

The model `lid.176.bin` (131 MB) is not committed. Pass its location with
`cache_dir` or `custom_model_path`.

## Ported

- `infer`: `LangDetectConfig`, `LangDetector`, `detect`, `set_default_detector`
- `fasttext`: model loading and `predict` with hierarchical softmax (fastText 0.9.2)

## Not ported

- Model download, MD5 check and download lock (a missing model returns `FileNotFound`)
- The lite model (`lid.176.ftz`) and quantized models
- Loss functions other than hierarchical softmax, `multilinePredict`, training
- `detect_language` and `is_japanese` (split-lang does not call them)

## Parity

Fixtures are generated from the pinned Python package with
`pixi run --manifest-path crates/ports/fast_langdetect/pixi.toml fixtures`,
which also downloads the model to `parity/.cache/`. Run the tests with
`FAST_LANGDETECT_MODEL=crates/ports/fast_langdetect/parity/.cache/lid.176.bin cargo test -p fast_langdetect`.
On macOS arm64, all 7151 cases (3341 texts) have equal labels and bit-exact
scores. See [parity/README.md](parity/README.md) for the fixture details,
platform notes and a comparison with other Rust fastText crates.

## License

MIT, from fast-langdetect (Copyright (c) 2024 LLM Kira) and fastText
(Copyright (c) 2016-present, Facebook, Inc.). Both notices are in [LICENSE](LICENSE).

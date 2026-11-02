# fast_langdetect

Rust port of the parts of [fast-langdetect](https://github.com/LlmKira/fast-langdetect)
1.1.0 that GPT-SoVITS uses through split-lang, with a pure Rust port of the
fasttext-predict 0.9.2.4 prediction path (fastText 0.9.2 C++) underneath. No C
or C++ code is linked.

GPT-SoVITS replaces the default detector with
`LangDetector(LangDetectConfig(cache_dir=.../pretrained_models/fast_langdetect))`,
and split-lang calls `detect(text, model="full")` and
`detect(text, model="full", k=5, threshold=0.01)`. The Rust equivalent:

```rust
use fast_langdetect::{DetectOptions, LangDetectConfig, LangDetector, Model};

let config = LangDetectConfig { cache_dir: "pretrained_models/fast_langdetect".into(), ..Default::default() };
fast_langdetect::set_default_detector(LangDetector::new(config));
let res = fast_langdetect::detect("你好", &DetectOptions { model: Some(Model::Full), k: 5, threshold: 0.01 }, None)?;
```

The model `lid.176.bin` (131 MB) is not committed. The API takes its location
(`cache_dir` or `custom_model_path`).

## Ported

`fast_langdetect/infer.py` (`fast_langdetect::infer`):

- `LangDetectConfig` (`cache_dir`, `custom_model_path`, `normalize_input`,
  `max_input_length`, `model`) and `CACHE_DIRECTORY` (`$FTLANG_CACHE`).
- `LangDetector.detect`, `_preprocess_text` (newline replacement, truncation
  to `max_input_length` characters), `_normalize_text` (`str.isupper()` or
  more than 80% ASCII capitals with length > 5, then `str.lower()`),
  `_get_model` (per-key model cache), result formatting
  (`__label__` removal, `min(score, 1.0)`, stable sort by score).
- `_default_detector` and the module-level `detect` (with or without
  `config`). `set_default_detector` replaces the Python attribute assignment
  that GPT-SoVITS does.

fasttext-predict / fastText (`fast_langdetect::fasttext`):

- `fasttext.load_model` and `_FastText.predict` (newline check, appended `\n`,
  `threshold` narrowed to `f32`).
- Model loading (`FastText::loadModel`, `checkModel`, `Args::load`,
  `Dictionary::load`, `DenseMatrix::load`), including the version 11 `maxn = 0`
  rule and the pruned-dictionary error.
- `Dictionary::getLine`: `readWord` (separators `' ' \n \r \t \v \f \0`, EOS
  handling, stopping at a literal `</s>`), label tokens, `hash` (FNV-1a with the
  `int8_t` sign extension), `computeSubwords` (UTF-8 aware char n-grams,
  `minn`/`maxn`, bucket modulo), `pushHash`, `addWordNgrams` (with the
  `int32_t` sign extension of word hashes), in-vocabulary subwords.
- `Model::predict` / `computeHidden` (sum of rows, multiply by
  `float(1.0 / n)`), `k = -1` (unlimited) and the `k <= 0` error.
- `HierarchicalSoftmaxLoss`: Huffman tree from label counts (`buildTree`),
  the DFS top-k search with the `std_log(threshold)` pruning (which also drops
  low-probability leaves at `threshold = 0`), the `float`/`double` mix of
  `1. / (1 + std::exp(-f))` and `std_log`, and the libc++ heap algorithms
  (`push_heap`, Floyd `pop_heap`, `sort_heap`) that order the results.

## Left out

- Model download, MD5 verification, the download lock, proxies and the Windows
  load fallbacks. A missing `lid.176.bin` returns `FileNotFound` instead of
  downloading it.
- The lite model (`model="lite"`, the bundled quantized `lid.176.ftz`) and
  quantized models in general (`QuantMatrix`, product quantization). They
  return `Unsupported` / `ModelLoad`.
- `MemoryError` fallback of `model="auto"`: `"auto"` loads the full model.
- Loss functions other than hierarchical softmax (`softmax`, `ns`, `ova`),
  non-supervised models, `multilinePredict`, and fastText training.
- `detect_language` and `is_japanese` from `fast_langdetect/__init__.py`
  (split-lang does not call them).

## Parity

`parity/gen_fixtures.py` writes `tests/fixtures/detect.jsonl` (1.5 MB). Inputs: hand-written edge cases (empty
and whitespace-only strings, every separator, `\0`, labels and `</s>` in the
text, digits, punctuation, emoji, CJK, kana, Hangul, many scripts, mixed
CJK/Latin, newlines, upper case and titlecase letters for normalization,
strings around the 80 character limit) plus bulk inputs with seed 176 (model
vocabulary words and phrases, random code points from 25 script ranges, the
GPT-SoVITS corpus and slices of it).

Configurations: `k=1, threshold=0` and `k=5, threshold=0.01` (the split-lang
calls) for every text; `k=3` with `normalize_input=False,
max_input_length=None` for every 10th text; `k=-1, threshold=0` and
`k=176, threshold=0.5` for every 50th text.

The parity tooling is local to this crate (see `parity/README.md`). It has its
own pixi workspace. From the repository root:

```sh
# Download lid.176.bin to parity/.cache/ and regenerate the fixtures.
pixi run --manifest-path crates/ports/fast_langdetect/pixi.toml fixtures
```

Run the tests with the model:

```sh
FAST_LANGDETECT_MODEL=crates/ports/fast_langdetect/parity/.cache/lid.176.bin cargo test -p fast_langdetect
```

Result (macOS arm64): 3341 texts, 7151 cases, labels equal in 7151 cases and
scores bit-exact in 7151 cases. The test asserts labels exactly (against an
empty documented exception list) and scores within 1e-5.

Notes:

- `DenseMatrix::dotRow` uses a separate multiply and add. The reference
  arm64 wheel does not use FMA there; with `f32::mul_add` only 4024 of the
  7151 cases are bit-exact (labels still match).
- `exp` and `log` call the platform libm, as the C++ code does. Other
  platforms can differ in the last bit of a score.
- The order of exactly equal scores follows libc++. A Linux wheel built with
  libstdc++ can order such ties differently (none occur in the fixtures).
- `str.isupper()` / `str.lower()` use Rust's Unicode tables, which can be newer
  than Python 3.11's Unicode 14.0 for characters assigned after 14.0.
- The test parser enables serde_json `float_roundtrip`, so fixture scores are
  read back exactly.

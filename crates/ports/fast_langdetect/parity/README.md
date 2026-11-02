# fast_langdetect parity tooling

Maintainer notes for the `fast_langdetect` port: what is ported in detail,
how the golden fixtures are made, and known platform differences.

The Python scripts that generate the golden fixtures run in the crate's own
pixi workspace (`../pixi.toml`, Python 3.11, fast-langdetect
1.1.0, fasttext-predict 0.9.2.4).

## How GPT-SoVITS calls it

GPT-SoVITS replaces the default detector with
`LangDetector(LangDetectConfig(cache_dir=.../pretrained_models/fast_langdetect))`,
and split-lang calls `detect(text, model="full")` and
`detect(text, model="full", k=5, threshold=0.01)`. The Rust equivalent is the
example in the crate README.

## Ported in detail

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

## Left out in detail

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

## Scripts

- `fetch_model.py`: downloads the fastText `lid.176.bin` model (131 MB) to
  `parity/.cache/lid.176.bin` and checks its MD5. It does nothing when the
  file is already present.
- `gen_fixtures.py`: runs fast-langdetect with that model and writes
  `tests/fixtures/detect.jsonl`.
- `corpus/gpt_sovits_zh_en_yue.json`: the GPT-SoVITS sentence corpus used as
  part of the fixture inputs.

## Tasks

Run from the repository root:

```sh
pixi run --manifest-path crates/ports/fast_langdetect/pixi.toml lid-model  # download the model
pixi run --manifest-path crates/ports/fast_langdetect/pixi.toml fixtures   # regenerate tests/fixtures
```

`fixtures` runs `lid-model` first. The crate has no generated files under
`data/`, so there is no `data` task.

## Cache

Downloads go to `parity/.cache/` (gitignored). `FAST_LANGDETECT_MODEL`
overrides the model path for both scripts. The file must be named
`lid.176.bin`, because fast-langdetect looks for that name in its
`cache_dir`. The Rust parity tests read the same variable:

```sh
FAST_LANGDETECT_MODEL=crates/ports/fast_langdetect/parity/.cache/lid.176.bin \
  cargo test -p fast_langdetect --release
```

## Fixtures

`gen_fixtures.py` writes `tests/fixtures/detect.jsonl` (1.5 MB). Inputs:
hand-written edge cases (empty and whitespace-only strings, every separator,
`\0`, labels and `</s>` in the text, digits, punctuation, emoji, CJK, kana,
Hangul, many scripts, mixed CJK/Latin, newlines, upper case and titlecase
letters for normalization, strings around the 80 character limit) plus bulk
inputs with seed 176 (model vocabulary words and phrases, random code points
from 25 script ranges, the GPT-SoVITS corpus and slices of it).

Configurations: `k=1, threshold=0` and `k=5, threshold=0.01` (the split-lang
calls) for every text; `k=3` with `normalize_input=False,
max_input_length=None` for every 10th text; `k=-1, threshold=0` and
`k=176, threshold=0.5` for every 50th text.

## Result

macOS arm64: 3341 texts, 7151 cases, labels equal in 7151 cases and scores
bit-exact in 7151 cases. The test (`tests/parity.rs`) asserts labels exactly
against `KNOWN_LABEL_MISMATCHES`, which is empty, and scores within 1e-5.

## Numeric and platform notes

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

## Comparison with existing Rust fastText crates

Two pure Rust fastText crates on crates.io can load `lid.176.bin`. They were
compared with this port (checked 2026-11-02, macOS arm64, release build):
every case of `tests/fixtures/detect.jsonl` except empty, whitespace-only and
`\0` texts (7,123 cases), with the same preprocessing and normalization, and
the same `k` and `threshold` (`k=-1` passed as 176). This port matches Python
bit for bit on all of them, so it is the reference here.

| Crate | Labels equal | Scores bit-identical | Max score difference | Load `lid.176.bin` |
|---|---|---|---|---|
| this port | 7,123 / 7,123 | 7,123 | 0 | 24 ms |
| [`fasttext`](https://github.com/messense/fasttext-rs) 0.8.0 (MIT) | 7,115 / 7,123 | 1,328 | 0.10 | 29 ms |
| [`fasttext-pure-rs`](https://github.com/silvermine/fasttext-pure-rs) 0.1.0 (MIT) | 6,947 / 7,123 | 1,071 | 0.54 | 32 ms |

Why `fasttext` 0.8.0 differs:

- Its dot products and vector additions use SIMD (NEON, SSE, AVX), which sums
  in a different order than the C++ loops, so most scores differ in the last
  bits.
- It does not split words on `\v` and `\f`, which fastText C++ treats as
  whitespace. Texts containing them get different scores and rankings.
- For a text made only of `__label__...` tokens it returns no prediction.
  fastText C++ drops the label tokens and still predicts from `</s>`.

`fasttext-pure-rs` 0.1.0 differs more often, including the ranking for single
CJK and Hangul characters and for texts that contain `</s>`.

For GPT-SoVITS only the label matters in most cases, and `fasttext` 0.8.0
returns the same labels for 99.9% of the cases. The remaining cases are close
calls or unusual whitespace, where a different label changes how split-lang
segments the text and which G2P frontend handles it. This crate keeps its own
port so the GPT-SoVITS frontend stays bit-for-bit comparable with Python end
to end. `fasttext` 0.8.0 is the closest replacement if that requirement is
dropped; it is a full fastText implementation (training, quantization) rather
than only the prediction path.

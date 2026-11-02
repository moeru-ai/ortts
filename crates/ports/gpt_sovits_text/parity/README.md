# gpt_sovits_text: parity and maintainer notes

The maintainer-facing document of the crate: module status, API notes, the
parity results with every documented exception, the known gaps, the license
details, and the Python scripts that run the upstream GPT-SoVITS text frontend
(commit `04a44eb`) and write the golden fixtures under `tests/fixtures/` and
the generated files under `data/` and `src/`. The Python environment is the
crate's own pixi workspace (`../pixi.toml`, locked in `../pixi.lock`). Rust
tests never need Python. `pyopenjtalk` is built from its sdist when the
environment is installed, which needs CMake and a C/C++ compiler. See
`crates/ports/README.md` for the conventions that all port crates follow.

## Modules

| Rust module | Upstream file (`GPT_SoVITS/...`) | Status | Details |
|---|---|---|---|
| crate root (`lib.rs`) | `text/__init__.py` | ported | [API notes](#api-notes) |
| `symbols` | `text/symbols.py` | ported | [API notes](#api-notes) |
| `symbols2` | `text/symbols2.py` | ported | [API notes](#api-notes) |
| `cleaner` | `text/cleaner.py` | ported (no `chinese`, `korean`) | [API notes](#api-notes) |
| `tts_infer_pack::text_preprocessor` | `TTS_infer_pack/TextPreprocessor.py` | ported | [API notes](#api-notes), [`e2e/README.md`](e2e/README.md) |
| `tts_infer_pack::text_segmentation_method` | `TTS_infer_pack/text_segmentation_method.py` | ported | [API notes](#api-notes) |
| `chinese2` | `text/chinese2.py` | ported | [`chinese2/README.md`](chinese2/README.md) |
| `tone_sandhi` | `text/tone_sandhi.py` | ported | [`tone_sandhi/README.md`](tone_sandhi/README.md) |
| `zh_normalization` | `text/zh_normalization/*.py` | ported | [`src/zh_normalization/mod.rs`](../src/zh_normalization/mod.rs) |
| `g2pw` | `text/g2pw/*.py` | ported | [`g2pw/README.md`](g2pw/README.md) |
| `english` | `text/english.py` | ported | [`english/README.md`](english/README.md) |
| `en_normalization` | `text/en_normalization/expend.py` | ported | [`en_normalization/README.md`](en_normalization/README.md) |
| `japanese` | `text/japanese.py` | ported (jpreprocess instead of pyopenjtalk) | [`japanese/README.md`](japanese/README.md) |
| `cantonese` | `text/cantonese.py` | ported | [`cantonese/README.md`](cantonese/README.md) |
| `lang_segmenter` | `text/LangSegmenter/langsegmenter.py` | ported | [`lang_segmenter/README.md`](lang_segmenter/README.md) |

Not ported: `text/chinese.py` (the v1 Chinese frontend) and `text/korean.py`.
GPT-SoVITS v2 inference does not call `chinese.py`; `korean.py` is only
called for Korean text. Where upstream would import one of them, the port
returns `cleaner::Error::NotPorted` (see [Known gaps](#known-gaps)).

## API notes

Language modes are the upstream ones: `all_zh`, `all_yue`, `all_ja`,
`all_ko`, `en`, `auto`, `auto_yue`, `zh`, `yue`, `ja`, `ko`. Only Chinese
(`zh`) segments run the BERT model; other segments get zeros of shape
`[1024, n_phones]`.

### Constructors and external models

- `TextPreprocessor::from_paths(bert_model, g2pw_model_dir,
  fast_langdetect_model)` builds what GPT-SoVITS sets up at import time:
  - the `chinese-roberta-wwm-ext-large` tokenizer (embedded,
    `data/g2pw/tokenizer.json`);
  - `chinese2` with the G2PW ONNX model from `g2pw_model_dir` (the
    extracted `G2PWModel_1.1.zip`; upstream `GPT_SoVITS/text/G2PWModel`);
  - fast-langdetect's default detector with `fast_langdetect_model`
    (`lid.176.bin`; upstream
    `GPT_SoVITS/pretrained_models/fast_langdetect/lid.176.bin`). As in
    Python (`langsegmenter.py` replaces the module-level detector when it
    is imported), this sets the process-wide default detector.
- `TextPreprocessor::new(bert_model, tokenizer, cleaner)` takes the parts
  directly; `TextPreprocessor::default_tokenizer()` is the embedded
  tokenizer, and `Cleaner::new(g2pw_model_dir)` or
  `Cleaner::from_chinese2(chinese2)` builds the language modules.
- The English (`g2p_en` model, CMU dictionaries) and Japanese (jpreprocess
  with `naist-jdic` and the GPT-SoVITS user dictionary) frontends are
  embedded and build their state on first use.

`cleaner::Cleaner::clean_text(text, language, version)` is the lower-level
entry point (`text/cleaner.py`): phones as symbol strings, `word2ph` (for
`zh` and `yue`) and the normalized text. `cleaned_text_to_sequence` maps the
phones to ids.

### `text/__init__.py` (crate root)

- `cleaned_text_to_sequence(cleaned_text, version=None)`. `None` reads the
  `version` environment variable (default `"v2"`). An unknown symbol returns
  `KeyError` with the symbol, where Python raises `KeyError`.
- `_symbol_to_id_v1` and `_symbol_to_id_v2` as `symbol_to_id_v1()` and
  `symbol_to_id_v2()`, built as the Python dict comprehensions build them.
- Re-exports: `Cleaner`, `TextPreprocessor`, `BertFeatureModel`,
  `BertInputs`, `PreprocessResult`.

### `text/symbols.py`, `text/symbols2.py`

All module constants, with the Python names in upper case: `PUNCTUATION`
(after the `append("-")`), `PU_SYMBOLS`, `PAD`, `C`, `V`, `V_WITHOUT_TONE`,
`JA_SYMBOLS`, `ARPA`, and in `symbols2` also `KO_SYMBOLS` and `YUE_SYMBOLS`.
`SYMBOLS` is computed the same way as in Python (`sorted(set(...))`, then the
v2 additions). The Python `set` literals (`arpa`, `yue_symbols`) are kept in
source order; their iteration order is never observable because Python only
uses them through `sorted`.

### `text/cleaner.py`

`special` (`SPECIAL`), `clean_text`, `clean_special` and `text_to_sequence`,
as methods of `Cleaner`, which holds the `chinese2` module (the only language
module that needs a constructor argument).

- Language and version handling is the upstream one: `version=None` reads the
  `version` environment variable (default `"v2"`); `"v1"` uses `symbols_v1`
  and `{zh: chinese, ja: japanese, en: english}`, every other value uses
  `symbols_v2` and `{zh: chinese2, ja: japanese, en: english, ko: korean,
  yue: cantonese}`; an unknown language becomes `en` with the text `" "`;
  `￥` and `^` in `zh` text go through `clean_special` (`SP2`, `SP3`); `en`
  results with fewer than 4 phones get a leading `","`; phones not in the
  symbol table become `UNK`; the two `zh`/`yue` assertions are kept.
- `chinese` (v1 `zh`) and `korean` (`ko`) give `Error::NotPorted` (name
  `NotPortedError`) instead of a fallback.
- `clean_special` indexes `g2p(...)[0]` and `[1]`, which only makes sense for
  the `(phones, word2ph)` modules (`zh`, `yue`). Upstream only calls it for
  `zh`. For `en` and `ja` the port returns `NotPorted`.
- `text_to_sequence` calls `clean_text(text)` without `language` upstream,
  so it always raises `TypeError`; the port returns that error.

### `TTS_infer_pack/TextPreprocessor.py`

`punctuation` (`PUNCTUATION`), `get_first`, `merge_short_text_in_array`, and
`TextPreprocessor` with `preprocess`, `pre_seg_text`,
`segment_and_extract_feature_for_text`, `get_phones_and_bert` (every
language mode, the `" {2,}"` collapse, the merge of non-English segments for
`zh`/`yue`/`ja`/`ko`, and the `"." + text` retry when there are fewer than 6
phones), `get_bert_feature` (`[1:-1]` slice, `assert len(word2ph) ==
len(text)`, per-phone repeat, transpose), `clean_text_inf`, `get_bert_inf`
(zeros `[1024, len(phones)]` for non-`zh`), `filter_text` and
`replace_consecutive_punctuation`.

- The BERT model is the `BertFeatureModel` trait: given the tokenizer output
  (`BertInputs`: `input_ids`, `token_type_ids`, `attention_mask` with
  `[CLS]`/`[SEP]`), return `torch.cat(hidden_states[-3:-2], -1)[0]` as an
  `Array2<f32>` `[n_tokens, hidden]`. Features are `ndarray` `f32`
  `[1024, n_phones]`, concatenated along the phone axis in segment order.
- Python exceptions become `text_preprocessor::Error` values with the
  exception name (`Error::name`): `torch.cat` of an empty list (a `zh`
  segment whose normalized text is empty) is `ValueError`, as in torch 2.x;
  `res[i]` past the BERT output is `IndexError`; `filter_text` and
  `get_method` give `ValueError`.
- Left out: the `print` calls, `tqdm`, `self.device`, `self.bert_lock`, and
  the i18n translation of the `filter_text` message (the port keeps the key
  `请输入有效文本`).

### `TTS_infer_pack/text_segmentation_method.py`

`punctuation` (`PUNCTUATION`), `splits` (`SPLITS`), the `METHODS` registry
with `register_method`, `get_method` and `get_method_names`, `split_big_text`
(keyword argument `max_len` in `SplitBigTextOptions`), `split`, and
`cut0`..`cut5`.

- Where Python raises, Rust returns the error with the exception name:
  `split("")`, `cut1("")` and `cut2("")` (also for input that is empty after
  `strip("\n")`) give `IndexError`; `get_method` of an unknown name gives
  `ValueError` with the Python message.
- `cut4` uses the regex `(?<!\d)\.(?!\d)` and `cut5` uses `str.isdigit()`.
  Python's `\d` (`str.isdecimal`) and `str.isdigit` differ from Rust's
  `char` methods, so the module embeds code point tables generated from
  Python 3.11 (Unicode 14.0.0) by
  `text_segmentation_method/gen_digit_tables.py`
  (`py_isdecimal`, `py_isdigit`).

### The language modules

See the module docs and the READMEs in the [module table](#modules).

## Results

Every fixture is generated from the pinned Python packages and the upstream
checkout by a script under `parity/<module>/`, and the Rust
tests compare exactly. Counts are from the module READMEs and test files.

| Module | Fixture set | Cases | Exact matches |
|---|---|---|---|
| crate root | `symbols/symbols.json`: tables and `_symbol_to_id_*` | 322 + 732 symbols | all |
| crate root | `symbols/symbols.json`: `cleaned_text_to_sequence` | 462 | 462 |
| `text_segmentation_method` | `cases.jsonl`: `split`, `cut0`..`cut5`, `split_big_text` (two `max_len`) | 1683 texts x 9 | 1683 x 9 |
| `zh_normalization` | `normalize.jsonl`, `normalize_sentence.jsonl`, `functions.json` | 6831 + 2214 + functions | all |
| `tone_sandhi` | `pre_merge`, `modified_tone`, `units`, `split_word` | 6439, 16684, 4377, 15160 | all |
| `g2pw` | `opencc`, `tokenizer`, `call` (+4 option sets), `model`, `pinyin`, `chinese2`, `correct_pronunciation` | 3162, 6052, 1702 (+4 x 180) batches, 1966 rows, 2580, 447 batches, 2308 | all (ONNX confidences within 4.9e-6) |
| `chinese2` | `sentences.jsonl` (`text_normalize`, `g2p` both branches, raw `g2p`), `punct.jsonl`, `units.jsonl` | 5607, 5606 x 2, 2143, 1561 x 4, 3670 + 1175 | all |
| `en_normalization` | `hand`, `corpus`, `bulk`, `py_int` | 345, 220, 6700, 32 | all |
| `english` | `texts.jsonl` (hand, corpus, tokens, sentences), `qryword`, `funcs`, `dict.json` | 273, 210, 1949, 3720, 2829, 645 | all |
| `japanese` | `corpus.jsonl` `g2p` | 457 | 456 (`ヵヶ`, documented) |
| `japanese` | `bulk_phrase`, `bulk_random`, `bulk_userdict` `g2p` | 600, 1000, 1544 | 598, 963, 1536 (documented) |
| `cantonese` | `sentences.jsonl`, `syllables.jsonl` | 3357, 2190 | all |
| `lang_segmenter` | `get_texts.jsonl` (4 `default_lang`), `functions.jsonl` | 3571 x 4, 1508 | all |
| `cleaner` | `clean_text.jsonl` | 2096 | 2093 (3 Japanese, listed) |
| `cleaner` | `clean_special.jsonl`, `text_to_sequence.jsonl` | 216, 3 | all |
| `text_preprocessor` | `functions.jsonl`: `get_first`, `replace_consecutive_punctuation`, `filter_text`, `pre_seg_text` (2 langs x 7 methods) | 345 texts x 17 | all |
| `text_preprocessor` | `merge_short_text_in_array.jsonl` | 600 | 600 |
| e2e | `get_phones_and_bert.jsonl` | 2682 | 2673 (9 Japanese, listed) |
| e2e | `preprocess.jsonl` (`cut0`..`cut5`) | 6264 | 6252 (12 Japanese, listed) |

### End-to-end (`tests/e2e.rs`)

`parity/e2e/gen_fixtures.py` runs upstream
`TextPreprocessor` with the G2PW model, `lid.176.bin`, and a deterministic
fake BERT model implemented the same way in Python and Rust (hidden state
`[t, d] = ((id_t * 1009 + t * 31 + d * 17) % 65521) / 64`). The BERT
features are compared by shape and an exact weighted checksum of all
entries, so a feature from the wrong token, layer or position, or repeated
the wrong number of times, fails the test. Phone ids and `norm_text` are
compared exactly. Results (`get_phones_and_bert`, cases / exact):

| Corpus | Mode | Cases | Exact |
|---|---|---|---|
| `gpt_sovits_zh_en_yue.json` | item `lang` (`zh` / `yue`) | 147 / 8 | 147 / 8 |
| `gpt_sovits_zh_en_yue.json` | `auto` | 155 | 154 |
| `gpt_sovits_zh_en_yue.json` | `auto_yue` | 155 | 154 |
| `gpt_sovits_zh_en_yue.json` | `all_zh` | 155 | 155 |
| `gpt_sovits_zh_en_yue.json` | `en` | 155 | 155 |
| `gpt_sovits_zh_en_yue.json` | `all_yue` (`yue` items) | 8 | 8 |
| `gpt_sovits_ja.json` | `ja` | 457 | 456 |
| `gpt_sovits_ja.json` | `all_ja` | 457 | 456 |
| `gpt_sovits_ja.json` | `auto` | 457 | 456 |
| 48 edge texts | `zh`, `yue`, `en`, `auto`, `auto_yue`, `all_zh`, `all_yue`, `all_ko`, `ko` | 48 each | 48 each |
| 48 edge texts | `ja`, `all_ja` | 48 each | 46 each |

`preprocess` (cases are text x method, `cut0`..`cut5`):

| Corpus | Mode | Cases | Exact |
|---|---|---|---|
| `gpt_sovits_zh_en_yue.json` | `zh` / `yue` | 882 / 48 | 882 / 48 |
| `gpt_sovits_ja.json` | `ja` | 2742 | 2736 |
| 48 edge texts | `zh`, `en`, `auto` | 288 each | 288 each |
| 48 edge texts | `ja` | 288 | 282 |
| 120 random paragraphs (seed 20261102) | `zh`, `yue`, `en`, `ja`, `auto`, `auto_yue`, `all_zh` | 336, 60, 96, 186, 660, 60, 42 | all |

The edge texts cover empty and whitespace-only input (`IndexError` from
split-lang), only punctuation, very short text (the `"."` retry), `￥` and
`^` (`clean_special`), consecutive punctuation, repeated spaces, newlines,
texts over 510 characters (`split_big_text`), Korean (`NotPortedError` in
`ko`, `all_ko`, `auto`), and `zh` segments that normalize to nothing
(`ValueError` from `torch.cat`). The random paragraphs join 2 to 8
sentences of the corpora (Mandarin, Cantonese, English, Japanese, a few
Korean) with random separators.

Every mismatch is in the Japanese frontend and listed in the tests, by
cause:

- `KANA_TABLE` (`ヵヶ`) and `CHAIN_AFTER_PAUSE` (`得体，着实让人着迷。`, a
  Japanese segment of a corpus sentence in `auto`): documented differences
  between jpreprocess and Open JTalk, items 6 and 7 of
  [`japanese/README.md`](japanese/README.md).
- `UNKNOWN_KANJINUMERIC_GROUP`: Chinese text in the `ja` modes
  (`《三体》是一部科幻小说。`, `这是一个很长的句子` x 60). MeCab groups an
  unknown word that starts with a `KANJINUMERIC` character (`一`) across the
  `KANJI` characters after it (`一部科幻小说` becomes one `記号,数` token
  with no reading); the `japanese` module's MeCab tokenizer port does not.
  This is not documented in the `japanese` module and is reported as a bug
  there. Repro: `japanese::g2p("是一部科幻小说", true)` is `["z", "e"]` in
  Python and 26 phones in Rust.

## Known gaps

- `text/korean.py` is not ported. `clean_text(..., "ko")` returns
  `cleaner::Error::NotPorted`, so `ko` and `all_ko` fail, and `auto`,
  `auto_yue` and the merging modes fail when `LangSegmenter` finds a Korean
  segment. Text without Korean works in every mode.
- `text/chinese.py` (v1 Chinese) is not ported: `clean_text(..., "zh",
  "v1")` returns `NotPorted`. v1 `ja` and `en` (with the v1 symbol table)
  work.
- The Japanese frontend uses jpreprocess instead of Open JTalk; see
  [`japanese/README.md`](japanese/README.md) for the measured differences.
  In addition, Chinese-only sentences given to the Japanese frontend
  (`clean_text(text, "ja")` on Mandarin text, which GPT-SoVITS does in the
  `ja` and `all_ja` modes) differ from Open JTalk much more often than
  Japanese text does: 27 of the 155 zh/en/yue corpus sentences, mostly
  through unknown-word grouping and accent phrases of simplified hanzi that
  are not in `naist-jdic`. These are left out of the `cleaner` fixture
  (only Japanese text runs in `ja` there) and reported against the
  `japanese` module.
- `clean_special` for `en` and `ja` (never called upstream) returns
  `NotPorted`.
- No real BERT model is bundled. `BertFeatureModel` is the plug-in point;
  the parity tests use a fake model, so they check the feature plumbing, not
  model numerics.

## License details

GPT-SoVITS is MIT licensed (`LICENSE`, Copyright (c) 2024 RVC-Boss). Parts
of the upstream code and data come from other projects under Apache-2.0
(`LICENSE-APACHE`), and the Japanese tokenizer translates parts of MeCab
(BSD-3-Clause, `LICENSE-MECAB`), so the crate license is
`MIT AND Apache-2.0 AND BSD-3-Clause`:

- `zh_normalization`, `tone_sandhi`: PaddleSpeech (Copyright (c) 2020-2021
  PaddlePaddle Authors), Apache-2.0. See `NOTICE`.
- `g2pw`: g2pW (<https://github.com/GitYCC/g2pW>), Apache-2.0, as vendored
  by GPT-SoVITS.
- `japanese::pyopenjtalk_g2p_prosody` and `_numeric_feature_by_regex`:
  copied upstream from ESPnet, Apache-2.0.
- `japanese` user dictionary tokenizer: parts of MeCab as bundled with
  Open JTalk (Copyright (c) 2001-2008 Taku Kudo, 2004-2008 NTT),
  BSD-3-Clause. See `NOTICE` and `LICENSE-MECAB`.
- `data/g2pw/tokenizer.json`: `hfl/chinese-roberta-wwm-ext-large` (HFL),
  Apache-2.0. `data/g2pw/opencc/`: OpenCC 1.4.2, Apache-2.0. See
  `data/g2pw/README.md`.
- `data/english/`: the CMU Pronouncing Dictionary files as shipped by
  GPT-SoVITS (`cmudict.rep`, `cmudict-fast.rep`, `engdict-hot.rep`,
  `namedict_cache.pickle`).
- `data/ja_userdic/userdict.csv`, `data/chinese2/opencpop-strict.txt`,
  `data/g2pw/polyphonic*.rep`: GPT-SoVITS (MIT).

The Rust dependencies keep their own licenses (`jpreprocess` and its
`naist-jdic`, `split_lang` with budoux, `fast_langdetect`, and the other
port crates in `crates/ports`).

## Tasks

Run from the repository root (or `pixi run <task>` in the crate directory):

```sh
pixi run --manifest-path crates/ports/gpt_sovits_text/pixi.toml <task> [args]
```

| Task | What it does |
|---|---|
| `upstream` | Shallow-fetches <https://github.com/RVC-Boss/GPT-SoVITS> at `04a44eb977586645f33594b72af13a035a2056c6` into `.cache/GPT-SoVITS`. Needs `git` on `PATH`. |
| `g2pw-model` | Downloads `G2PWModel_1.1.zip` from ModelScope (the URL of upstream `download_and_decompress`, about 600 MB), extracts it to `.cache/G2PWModel`, checks the file hashes, and links `.cache/run/GPT_SoVITS/text/G2PWModel` to it (a junction on Windows when symlinks are not allowed). |
| `bert-tokenizer` | Downloads the tokenizer files of `hfl/chinese-roberta-wwm-ext-large` (pinned revision, hash-checked) into `.cache/run/GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large`. The model weights are not needed: the generators use a fake BERT model. |
| `lid-model` | Downloads the fasttext `lid.176.bin` (about 125 MB, hash-checked) into `.cache/fast_langdetect`, and links upstream `GPT_SoVITS/pretrained_models/fast_langdetect` to it, where upstream `langsegmenter.py` loads it from. |
| `assets` | Runs the four tasks above. |
| `fixtures` | Runs `assets`, then every `<module>/gen_fixtures.py`, or only the modules given as arguments (`fixtures cleaner e2e`). |
| `data` | Runs `upstream` and `bert-tokenizer`, then every data generator, or only the ones given as arguments (`data g2pw`). |

The download tasks skip what is already in place, so they are cheap to run
again. `fetch.py` implements them and `regen.py` runs the generators, each in
its own interpreter. The whole `fixtures` task takes several minutes; `g2pw`,
`chinese2`, `tone_sandhi` and `e2e` are the slow ones.

`.cache/run/` is the working directory of the generators:
`gpt_sovits_env.setup()` changes to it, so the relative paths hardcoded in
upstream (`GPT_SoVITS/text/G2PWModel`, `GPT_SoVITS/pretrained_models/...`)
resolve inside the cache.

## Environment variables

All optional; the defaults are inside the crate.

| Variable | Default | Used by |
|---|---|---|
| `GPT_SOVITS_TEXT_PARITY_CACHE` | `parity/.cache` | every script: where the downloads go and are read from |
| `GPT_SOVITS_ROOT` | `<cache>/GPT-SoVITS` | the generators; the `upstream` task only checks that a checkout given here is at the pinned commit and does not move it |

The Rust tests read `G2PW_MODEL_DIR` and `FAST_LANGDETECT_MODEL` and skip the
model tests when they are unset. After `assets`:

```sh
G2PW_MODEL_DIR=$PWD/crates/ports/gpt_sovits_text/parity/.cache/G2PWModel \
FAST_LANGDETECT_MODEL=$PWD/crates/ports/gpt_sovits_text/parity/.cache/fast_langdetect/lid.176.bin \
  cargo test -p gpt_sovits_text --release
```

## Scripts

| Script | Writes | Details |
|---|---|---|
| `symbols/gen_fixtures.py` | `tests/fixtures/symbols/` | |
| `text_segmentation_method/gen_fixtures.py` | `tests/fixtures/text_segmentation_method/` | |
| `zh_normalization/gen_fixtures.py` | `tests/fixtures/zh_normalization/` | `src/zh_normalization/mod.rs` |
| `tone_sandhi/gen_fixtures.py` | `tests/fixtures/tone_sandhi/` | `tone_sandhi/README.md` |
| `chinese2/gen_fixtures.py` | `tests/fixtures/chinese2/` | `chinese2/README.md` |
| `g2pw/gen_fixtures.py` | `tests/fixtures/g2pw/` | `g2pw/README.md` |
| `english/gen_fixtures.py` | `tests/fixtures/english/` | `english/README.md` |
| `en_normalization/gen_fixtures.py` | `tests/fixtures/en_normalization/` | `en_normalization/README.md` |
| `japanese/gen_fixtures.py` | `tests/fixtures/japanese/` | `japanese/README.md` |
| `cantonese/gen_fixtures.py` | `tests/fixtures/cantonese/` | `cantonese/README.md` |
| `lang_segmenter/gen_fixtures.py` | `tests/fixtures/lang_segmenter/` | `lang_segmenter/README.md` |
| `cleaner/gen_fixtures.py` | `tests/fixtures/cleaner/` | `e2e/README.md` |
| `text_preprocessor/gen_fixtures.py` | `tests/fixtures/text_preprocessor/` | `e2e/README.md` |
| `e2e/gen_fixtures.py` | `tests/fixtures/e2e/` | `e2e/README.md` |
| `chinese2/convert_data.py` (`data chinese2`) | `data/chinese2/` | `data/chinese2/README.md` |
| `english/gen_data.py` (`data english`) | `data/english/` | `english/README.md` |
| `g2pw/convert_data.py` (`data g2pw`) | `data/g2pw/` | `data/g2pw/README.md` |
| `zh_normalization/convert_data.py` (`data zh_normalization`) | `data/zh_normalization/` | `data/zh_normalization/README.md` |
| `en_normalization/gen_tables.py` (`data en_normalization`) | `src/en_normalization/py_tables.rs` | |
| `tone_sandhi/gen_isnumeric_table.py` (`data tone_sandhi`) | the generated block of `src/tone_sandhi.rs` | |
| `text_segmentation_method/gen_digit_tables.py` (`data text_segmentation_method`) | the generated block of `src/tts_infer_pack/text_segmentation_method.rs` | |

Shared files:

- `gpt_sovits_env.py`: crate and cache paths, and `setup()`, which puts the
  upstream checkout on `sys.path` and changes to `.cache/run`.
- `e2e/e2e_common.py`: setup shared by the `cleaner`, `text_preprocessor` and
  `e2e` generators (fake BERT model, `NotPortedError`, edge texts).
- `lang_segmenter/lang_inputs.py`: the inputs of the `lang_segmenter`
  fixtures, a copy of the `split_lang` crate's inputs.
- `corpus/*.json`: the sentence corpora used by several generators.
- `python_ref.py`: dumps upstream phone ids for a corpus file, for ad hoc
  comparisons (not part of any task).

`data/ja_userdic/userdict.csv` is copied from upstream by hand;
`japanese/gen_fixtures.py` checks that it still equals the upstream file.

Python 3.11 is pinned on purpose: several generated tables are the Unicode
14.0.0 data of Python 3.11.

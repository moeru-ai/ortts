# `gpt_sovits_text::g2pw` parity

Port of `GPT_SoVITS/text/g2pw` (GPT-SoVITS `04a44eb`: `onnx_api.py`,
`dataset.py`, `utils.py`, `g2pw.py`, `__init__.py`) in
`crates/ports/gpt_sovits_text/src/g2pw/`, plus the part of OpenCC that the
converter runs. The model is `G2PWModel_1.1.zip` (not committed); the Rust
API takes the extracted directory.

## Ported scope

| Python | Rust |
|---|---|
| `g2pw/__init__.py` (`from .g2pw import *`) | `g2pw` re-exports `G2PWPinyin`, `Converter`, `correct_pronunciation` |
| `g2pw.py` `G2PWPinyin(Pinyin)` (`__init__`, `get_seg`) | `g2pw::g2pw::G2PWPinyin` (implements `pypinyin::PinyinMethods`), `G2PWPinyinOptions` (`chinese2()` = the arguments chinese2.py passes) |
| `g2pw.py` `Converter(UltimateConverter)` (`convert`, `_to_pinyin`) | `g2pw::g2pw::Converter` (implements `pypinyin::Converter`, holds the `UltimateConverter` base) |
| `_remove_dup_items`, `_remove_dup_and_empty`, `read_dict`, `get_dict`, `correct_pronunciation`, `pp_dict` | same names, `pp_dict` is `PP_DICT` |
| `onnx_api.py` `predict`, `_load_json_from_candidates`, `_find_first_existing_file`, `download_and_decompress`, `model_version` | same names (`MODEL_VERSION`); `run_session` / `argmax_labels` are the two halves of `predict` |
| `_G2PWBaseOnnxConverter` + `G2PWOnnxConverter` (`__init__`, `_convert_bopomofo_to_pinyin`, `__call__`, `_prepare_data`, `_predict`, `_predict_with_sentence_dedup`) | `onnx_api::G2PWOnnxConverter` (`new`, `call`, same method names; one struct, since the base class has a single subclass), `G2PWOnnxConverterOptions`, `G2PWStyle` |
| `dataset.py` `ANCHOR_CHAR`, `prepare_onnx_input`, `_truncate_texts`, `_truncate`, `get_phoneme_labels`, `get_char_phoneme_labels` | same names; the returned dict is `OnnxInput` |
| `utils.py` `wordize_and_map`, `tokenize_and_map`, `_load_config`, `default_config_dict`, `load_config` | same names; the config module is `Config` of `PyValue`s |
| `AutoTokenizer.from_pretrained(bert_path)` / `tokenizer.tokenize` / `convert_tokens_to_ids` | `onnx_api::tokenizer_from_json`, `utils::tokenize`, `dataset::convert_tokens_to_ids` (`tokenizers` crate, embedded `data/g2pw/tokenizer.json`) |
| `opencc.OpenCC("s2tw").convert` (OpenCC 1.4.2 C++) | `g2pw::opencc::OpenCC::s2tw().convert` |

Model inputs, label tables and the session options follow Python:

- `POLYPHONIC_CHARS.txt`, `MONOPHONIC_CHARS.txt`,
  `bopomofo_to_pinyin_wo_tune_dict.json`, `char_bopomofo_dict.json` and
  `config.py` are read from the model dir at runtime, like Python.
- `config.py` is parsed, not executed: module-level `name = <literal>`
  assignments (None/bool/int/float/str/list/dict, over several lines,
  comments). `G2PWModel/config.py` of the 1.1 zip only has those. The
  `use_default` merge of `load_config` is ported (`default_config_dict`,
  including the upstream key `"param_pos "` with its trailing space).
- Session: `GraphOptimizationLevel::All` (`ORT_ENABLE_ALL`), sequential
  execution (`ORT_SEQUENTIAL`), `intra_threads = 2`, providers `[CUDA, CPU]`
  when ONNX Runtime has the CUDA provider, else `[CPU]`.
- Environment variables `g2pw_sentence_dedup` (default on) and
  `g2pw_polyphonic_context_chars` (default 16) are read at construction, like
  Python. Both are also public fields, as they are attributes in Python.

### OpenCC branch

`G2PWPinyin.__init__` defaults to `enable_non_tradional_chinese=True` and
chinese2.py does not override it, so every sentence that GPT-SoVITS gives to
G2PW goes through `OpenCC("s2tw")` first. The branch is reachable and is
ported, including `assert len(translated_sent) == len(sent)`
(`AssertionError`). `g2pw::opencc` is a translation of the OpenCC 1.4.2
classes that `s2tw.json` uses: the normalization pass
(`CJK_Compatibility_Ideographs`, `ConfigBasedConverter`), `mmseg`
segmentation (`MaxMatchSegmentation`; unmatched runs merged into one
segment), the conversion chain with `short_circuit` and `union` dict groups
(`PrefixMatch` `LeafMatcher` / `GroupMatcher`, `CanFlattenAsUnion`, first entry
wins for a duplicate key), ideographic description sequences kept verbatim
(`NextIdeographicDescriptionSequenceLength`), and the C-string cut of a
segment at its first NUL (`"a\0b"` becomes `"a"`, which then fails the length
assertion). `SkipUnmatchable` is a speed-up that does not change the output
and is not ported. No dictionary entry changes the length, so the assertion
only fails through the NUL cut.

### Python quirks that are kept

- `_prepare_data` indexes `pinyin(sent_s, ...)` by character, but pypinyin
  returns one item per run of non-Han characters. After a run of two or more
  such characters later results are shifted, and an index past the end gives
  `IndexError`.
- `tokenize_and_map` writes token positions by the token length without
  `##`. A token longer than its text (a Hangul syllable becomes three jamo
  tokens) runs past the end: `IndexError`, which `prepare_onnx_input` catches
  (Python prints a warning and returns `{}`), so `__call__` returns the
  partial results with `None` at polyphonic positions.
- `prepare_onnx_input` lowercases the text (`str.lower`, Rust
  `to_lowercase`; both are the full Unicode mapping, `İ` becomes two
  characters) and then indexes it with the original query id, so the query
  character can shift; a character that is not a polyphonic char gives
  `KeyError`.
- `correct_pronunciation` returns the `pp_dict` entry for the word, or
  replaces `word_pinyins[i]` for each character with an entry; an index past
  the end of `word_pinyins` gives `IndexError`.

Python exceptions are `G2PWError` variants with the exception name
(`exception_name()`). Inside the pypinyin converter methods (which cannot
fail) a G2PW error panics, as Python exceptions do in the `pypinyin` crate.

## Left out

- The download in `download_and_decompress` (the port gives
  `FileNotFoundError` for a missing model dir), `onnxruntime` logger setup and
  `preload_dlls`, `warnings.filterwarnings`.
- The `polyphonic.pickle` / `polyphonic.md5` cache of `get_dict`
  (`get_file_md5`, `cache_dict`): it only stores `read_dict()`, which the port
  calls on first use.
- `model_source=None`: Python then loads `config.model_source`
  (`bert-base-chinese`) from the Hugging Face hub. The port uses the embedded
  `chinese-roberta-wwm-ext-large` tokenizer, which is what GPT-SoVITS passes
  as `bert_path`. `Some(dir)` loads `dir/tokenizer.json`.
- The second candidate dir of `_load_json_from_candidates`
  (`GPT_SoVITS/text/G2PWModel` next to the Python sources).
- `tone_sandhi` of `G2PWPinyin` / `Converter` (not in the `pypinyin` crate;
  GPT-SoVITS does not set it). OpenCC configs other than `s2tw`.
- `eval` of the `.rep` values is a parser for list literals of quoted
  strings, which is all the two files contain.

## Fixtures

`gen_fixtures.py` writes `crates/ports/gpt_sovits_text/tests/fixtures/g2pw/`
with the instance that chinese2.py builds (`text.chinese2.g2pw`). Inputs: 63
hand-written edge cases, the 147 zh items of
`parity/corpus/gpt_sovits_zh_en_yue.json`, and 3200 generated sentences
(seed `20251102`) built from pypinyin phrases that contain polyphonic
characters (行 长 乐 觉 重 还 地 传 处 为 好 曾 单 着 参 朝 调 藏 得 发 便 模 空 和
了 都 ...), filler characters and punctuation; 12% also get punctuation runs,
Latin words, digits, emoji, Hangul, kana, traditional characters, IDS and
special characters, and every 97th is a long sentence (up to 700 characters,
polyphonic characters at both ends) for the crop window and the 510-token
truncation.

For `call.jsonl` the generator first runs every sentence alone. A sentence
that raises (mostly the `IndexError` of a non-Han run) gets its own batch, so
it does not hide the results of the other sentences. The other sentences go
in batches of 1 to 4, sometimes with a repeated sentence (dedup across
sentences). The variants run on short batches only.

| Fixture | Contents | Cases | Exact |
|---|---|---|---|
| `tables.json` | config after defaults, labels (`get_phoneme_labels`), chars, char2phonemes, `polyphonic_chars_new`, dict sizes, `_convert_bopomofo_to_pinyin` for every label and bad inputs, `pp_dict` size and FNV-1a hash | 1 | all |
| `opencc.jsonl` | `OpenCC("s2tw").convert`: 1500 test sentences, all `STCharacters` / `TWVariants*` / `CJK_Compatibility_Ideographs` / regional-phrase keys in chunks, 1500 random mixes of phrase keys, partial keys and other text, IDS and NUL cases | 3162 | 3162 |
| `tokenizer.jsonl` | `tokenizer.tokenize(word)` for every word `wordize_and_map` makes from the inputs (raw and s2tw), 3000 random code points, 300 random ASCII words | 6052 | 6052 |
| `call.jsonl` `default` | `g2pw._g2pw(batch)` over 3393 sentences (3106 with a result, 287 with an exception: 278 `IndexError`, 7 `KeyError`, 2 `AssertionError`; 25 `None` results from the tokenizer `{}` path) | 1702 batches | 1702 |
| `call.jsonl` `nodedup`, `ctx0`, `ctx3`, `nodedup_ctx0` | the same with `enable_sentence_dedup` / `polyphonic_context_chars` changed | 4 x 180 batches | 4 x 180 |
| `model.jsonl` | `_prepare_data`, `prepare_onnx_input` (all six arrays) and `_predict` for 349 sentences (one over 510 tokens) | 1966 rows | 1966 argmax labels; confidences within 4.9e-6, full probabilities (7 rows x 1305 labels) within 4.9e-8 |
| `pinyin.jsonl` | `G2PWPinyin.pinyin` (TONE3; TONE with heteronym) and `lazy_pinyin` (TONE3, INITIALS, FINALS_TONE3, NORMAL) on 900 sentences | 2580 | 2580 |
| `chinese2.jsonl` | the `g2pw._g2pw` batches and `correct_pronunciation` calls of `chinese2.g2p(text_normalize(text))` for the 147 zh corpus items and 300 bulk sentences | 447 batches, 3760 calls | all |
| `correct_pronunciation.jsonl` | hand-written cases, 1500 sampled `pp_dict` words, 800 random words of `pp_dict` characters (some with a short `word_pinyins`, `IndexError`) | 2308 | 2308 |

Python runs onnxruntime 1.31.1; the `ort` crate (2.0.0-rc.11) runs ONNX
Runtime 1.23.2. The numeric drift is below 1e-5 and no argmax flips on the
fixtures, so `KNOWN_ARGMAX_FLIPS` in `tests/g2pw.rs` is empty. The tests
still compare the mismatches with that list, so a future flip has to be
listed there and here.

Branches that the fixtures do not reach, because the G2PW 1.1 config and
the upstream call sites never take them: `use_char_phoneme=True`
(`get_char_phoneme_labels`, the `pred.split(" ")[1]` step), `use_mask=False`,
`window_size` other than `None` (`_truncate_texts`), `style="bopomofo"`, and
non-literal `config.py` contents. They are ported line by line but not
checked against Python.

The fixtures are about 2.7 MB. The model tests take about 4 minutes in a
debug build (most of it is ONNX Runtime and the tokenizer).

Rust tests: `tests/g2pw_offline.rs` (no model) and `tests/g2pw.rs`
(`G2PW_MODEL_DIR` must point at the extracted `G2PWModel`; skipped otherwise).

```sh
pixi run --manifest-path crates/ports/gpt_sovits_text/pixi.toml data g2pw
pixi run --manifest-path crates/ports/gpt_sovits_text/pixi.toml fixtures g2pw
G2PW_MODEL_DIR=$PWD/crates/ports/gpt_sovits_text/parity/.cache/G2PWModel cargo test -p gpt_sovits_text --test g2pw --test g2pw_offline
```

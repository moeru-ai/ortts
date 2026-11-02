# gpt_sovits_text

Rust port of the [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) v2 text
frontend at commit `04a44eb`: `GPT_SoVITS/text` and the text parts of
`GPT_SoVITS/TTS_infer_pack`. It turns text into phone ids and BERT features,
as `TTS.py` does, with exact output parity as the goal.

```rust
use gpt_sovits_text::{BertFeatureModel, BertInputs, TextPreprocessor};
use gpt_sovits_text::tts_infer_pack::text_preprocessor::BertError;
use ndarray::Array2;

/// `chinese-roberta-wwm-ext-large`, for example as an ONNX model.
struct MyBert;

impl BertFeatureModel for MyBert {
  /// `hidden_states[-3]` for one tokenized text: `[n_tokens, 1024]`.
  fn hidden_states(&self, inputs: &BertInputs) -> Result<Array2<f32>, BertError> {
    todo!()
  }
}

// The extracted G2PWModel_1.1.zip and the fastText lid.176.bin.
let tp = TextPreprocessor::from_paths(MyBert, "/models/G2PWModel", "/models/lid.176.bin")?;
for seg in tp.preprocess("你好，世界。Hello world!", "auto", "cut1", "v2")? {
  // seg.phones (symbol ids), seg.bert_features [1024, n_phones], seg.norm_text
}
```

`from_paths` also sets the process-wide fast-langdetect detector, as
upstream does on import. Only `zh` segments run the BERT model; other
segments get zeros.

## Ported

- `TextPreprocessor` (`TTS_infer_pack/TextPreprocessor.py`): `preprocess`,
  `get_phones_and_bert`, every language mode (`all_zh` .. `auto_yue`)
- `tts_infer_pack::text_segmentation_method`: `cut0`..`cut5`, `split_big_text`
- `Cleaner` (`text/cleaner.py`), `cleaned_text_to_sequence`, `symbols`, `symbols2`
- Language modules: `chinese2` (with `g2pw`, `tone_sandhi`,
  `zh_normalization`), `english` (with `en_normalization`), `japanese`
  (jpreprocess instead of pyopenjtalk), `cantonese`, `lang_segmenter`

## Not ported

- `text/korean.py`: `ko`, `all_ko` and Korean segments in `auto` modes return `cleaner::Error::NotPorted`
- `text/chinese.py` (v1 Chinese): `clean_text(.., "zh", "v1")` returns `NotPorted`
- `clean_special` for `en` and `ja` (never called upstream) returns `NotPorted`
- No BERT model is bundled; `BertFeatureModel` is the plug-in point

## Parity

Fixtures come from the pinned upstream checkout and Python packages:
`pixi run --manifest-path crates/ports/gpt_sovits_text/pixi.toml fixtures`.
The `assets` task downloads the models; tests that need them read
`G2PW_MODEL_DIR` and `FAST_LANGDETECT_MODEL` and are skipped when unset.
Every module matches Python exactly, except the Japanese frontend; end to
end, `get_phones_and_bert` matches 2673 of 2682 cases and `preprocess` 6252
of 6264. Known differences, all Japanese:

- jpreprocess differs from Open JTalk on `ヵヶ` and one chain-after-pause case
- the MeCab port does not group unknown words that start with a `KANJINUMERIC` character (Chinese text in `ja` modes)

See [parity/README.md](parity/README.md) for every fixture set, the
exception lists and the tooling.

## License

MIT (GPT-SoVITS, Copyright (c) 2024 RVC-Boss, `LICENSE`) AND Apache-2.0
(PaddleSpeech, g2pW, ESPnet and data files, `LICENSE-APACHE`) AND
BSD-3-Clause (MeCab, `LICENSE-MECAB`). See `NOTICE` and the per-file list in
[parity/README.md](parity/README.md#license-details).

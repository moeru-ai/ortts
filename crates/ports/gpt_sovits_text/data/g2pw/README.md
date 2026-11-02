# g2pw data

Collected by `parity/g2pw/convert_data.py`. Do not edit by
hand. All files are embedded with `include_str!`.

## `polyphonic.rep`, `polyphonic-fix.rep`

`GPT_SoVITS/text/g2pw/polyphonic.rep` and `polyphonic-fix.rep` from
GPT-SoVITS commit `04a44eb` (MIT, Copyright (c) 2024 RVC-Boss), byte for
byte. One `word: ['pin1', 'yin1', ...]` entry per line. `read_dict()` reads
`polyphonic.rep` first and `polyphonic-fix.rep` second, so an entry in the
second file replaces one in the first. Upstream caches the result in
`polyphonic.pickle` (with an md5 file); the port parses the `.rep` files on
first use instead.

## `tokenizer.json`

The `tokenizer.json` of `hfl/chinese-roberta-wwm-ext-large`
(<https://huggingface.co/hfl/chinese-roberta-wwm-ext-large>), the
`bert_path` (`GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large`)
that GPT-SoVITS passes to `G2PWPinyin` as `model_source`, unchanged.
`convert_data.py` checks that the backend tokenizer that transformers 4.57.6
builds from that directory (`BertTokenizerFast`) serializes to the same JSON.

Copyright the Joint Laboratory of HIT and iFLYTEK Research (HFL). Licensed
under the Apache License, Version 2.0 (see `../../LICENSE-APACHE`). Paper:
Yiming Cui et al., "Pre-Training with Whole Word Masking for Chinese BERT"
(<https://github.com/ymcui/Chinese-BERT-wwm>).

## `opencc/`

The `s2tw` config of OpenCC 1.4.2 as shipped in the `opencc` 1.4.2 wheel of
the crate's pixi environment: `s2tw.json` unchanged, and the six dictionaries it
uses (`CJK_Compatibility_Ideographs`, `STPhrases`,
`STPhrases_GeneratedFromRegionalPhrases`, `STCharacters`,
`TWVariantsPhrases`, `TWVariants`) dumped from the wheel's `.ocd2` (marisa
trie) files to the OpenCC text format (`key<TAB>value1 value2 ...`) with the
wheel's own `opencc_dict -f ocd2 -t text`.

OpenCC (Open Chinese Convert, <https://github.com/BYVoid/OpenCC>), Copyright
2010-2026 Carbo Kuo (BYVoid) and contributors. Licensed under the Apache
License, Version 2.0 (see `../../LICENSE-APACHE`). The conversion code is
translated to Rust in `src/g2pw/opencc.rs`.

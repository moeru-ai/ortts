# `gpt_sovits_text::japanese` parity

Port of `GPT_SoVITS/text/japanese.py` (GPT-SoVITS `04a44eb`) in
`crates/ports/gpt_sovits_text/src/japanese.rs`. Upstream calls pyopenjtalk
0.4.1 (Open JTalk 1.11, dictionary `open_jtalk_dic_utf_8-1.11`). The port uses
jpreprocess 0.15.0 with its bundled `naist-jdic` (jpreprocess/naist-jdic
v0.1.3, built from the same `open_jtalk-1.11/mecab-naist-jdic` source).

## Ported scope

| Python | Rust |
|---|---|
| `text_normalize` | `japanese::text_normalize` |
| `g2p(norm_text, with_prosody=True)` | `japanese::g2p(norm_text, with_prosody)` |
| `preprocess_jap` | `japanese::preprocess_jap` |
| `pyopenjtalk_g2p_prosody(text, drop_unvoiced_vowels=True)` | `japanese::pyopenjtalk_g2p_prosody` |
| `_numeric_feature_by_regex` | `japanese::_numeric_feature_by_regex` (takes a compiled `Regex`) |
| `post_replace_ph`, `symbols_to_japanese`, `replace_consecutive_punctuation` | same names |
| `_japanese_characters`, `_japanese_marks`, `_symbols_to_japanese` | private statics |
| `text.symbols.punctuation` | private copy `PUNCTUATION` |
| module-level user dictionary setup (`mecab_dict_index` + `update_global_jtalk_with_user_dict`) | `japanese::pyopenjtalk::global_jtalk` |
| `pyopenjtalk.run_frontend`, `make_label`, `g2p`, `mecab_dict_index` | `japanese::pyopenjtalk::*` |

Left out: the Windows-only path copying at import time, the md5 cache file of
the compiled user dictionary (the dictionary is built in memory on first use),
and `if __name__ == "__main__"`.

`pyopenjtalk::run_frontend` returns `jpreprocess` `NJDNode`s instead of dicts;
`NJDNode::to_string()` gives the feature as a CSV string. `orig` is the surface
string there, because jpreprocess does not keep the base form.

## User dictionary

`crates/ports/gpt_sovits_text/data/ja_userdic/userdict.csv` is upstream's
`GPT_SoVITS/text/ja_userdic/userdict.csv`, byte for byte (the generator checks
the md5). It is embedded and compiled on first use by
`pyopenjtalk::mecab_dict_index` with jpreprocess's `JPreprocessDictionaryBuilder`
(the CSV format is the same, so no converter is needed). Two things differ from
`mecab-dict-index`, and the port handles both:

- CSV reading: `mecab-dict-index` splits only `surface,left_id,right_id,cost`
  with MeCab's `tokenizeCSV` and keeps the rest of the line raw; Open JTalk
  later splits it on `,` without unquoting. `tokenize_csv` ports
  `tokenizeCSV`, and the feature is split the same way (one row, `corps`, has
  quoted fields).
- Pronunciations: Open JTalk keeps `pron` as a string and parses it into moras
  only when it builds labels, with a katakana-only mora list, stopping at the
  first other character. jpreprocess parses `pron` while building the
  dictionary, accepts hiragana and fullwidth letters as moras, and rejects
  other characters (435 rows fail to build) and rows whose `mora_size` does not
  match. 460 rows have a `pron` with a non-katakana character (`バーベキューど`,
  `アドル・タバタバ`, `アサルトed`, `CEO`, `読売`). `jpcommon_compatible_row`
  cuts `pron` at the first character Open JTalk's label builder rejects and
  drops a `mora_size` that then no longer matches.

## Differences between pyopenjtalk 0.4.1 and jpreprocess 0.15.0

Measured on the fixtures below. The port fixes the first three; the rest are
documented exceptions.

1. **Segmentation (fixed).** jpreprocess tokenizes with lindera, whose Viterbi
   differs from MeCab: on equal path cost it keeps the first candidate where
   MeCab keeps the last (MeCab prepends to `end_node_list`), it adds no
   unknown-word candidates inside an unknown word it already added, and it
   ignores the `length` column of `char.def`. Ties are frequent with
   `userdict.csv`, whose cost depends only on the surface length: `adl-tabataba`
   is `ｔａｂ|ａｔａ|ｂａ` in MeCab and `ｔａ|ｂａｔ|ａｂａ` in lindera;
   `ハンバーガーーー` is `ハンバーガー|ーー` in MeCab and `ハン|バ|ーガーーー` in
   lindera. `pyopenjtalk::MecabTokenizer` is a port of MeCab's
   `Tokenizer::lookup` and `Viterbi` over the lindera dictionaries and is
   plugged into jpreprocess with `JPreprocess::from_tokenizer`.
2. **User dictionary pronunciations (fixed for g2p).** See above. Without the
   fix the bundled user dictionary does not build at all.
3. **Word-initial `ー` (fixed).** Open JTalk appends a `ー` that starts a word
   to the previous word and accent phrase, and drops it at the start of the
   text or after a pause. jpreprocess starts a new word at the `ー` and copies
   the previous phoneme, which is `sil` at the start (`ーー` gave `sil sil`,
   and a `sil` inside the labels made `pyopenjtalk_g2p_prosody` fail its
   assertion), and makes `べー` two accent phrases. `pyopenjtalk::make_label`
   moves the marks the way Open JTalk does before building labels.
4. **Label fields `/B:`, `/C:`, `/D:` (not fixed, not read).** jpreprocess
   leaves the next-word part of speech of the leading `sil` and the
   previous-word part of speech of the trailing `sil` as `xx`, and uses other
   codes for some parts of speech and conjugations (`感動詞` is `09` in Open
   JTalk and `25` in jpreprocess; some `助動詞` conjugation codes differ).
   `pyopenjtalk_g2p_prosody` reads only the phoneme and the `/A:`, `/E:` and
   `/F:` fields, so the test compares labels without `/B:` to `/D:`; the full
   label match rate is printed for reference.
5. **NJD notation (not fixed).** jpreprocess prints `chain_rule` in its own
   order, once per rule, without the default `@0`, and `*` instead of `-1`;
   it prints `chain_flag` `-1` where Open JTalk has `0` after digit
   processing. The test normalizes these. jpreprocess also does not model
   chain rules on `特殊助動詞` (`う` after `でしょ`, `ましょ`, `よ`) and drops
   them; the labels are the same (`TOKUSHU_JODOUSHI`).
6. **Unknown-word kana table (not fixed).** For words not in the dictionary,
   both fill the pronunciation from the surface with a kana table. Open JTalk's
   table has no `ヵ`, `ヶ`, `ゎ`, `ヮ`, so such a word becomes a pause or
   disappears; jpreprocess reads `ヶ` as `ケ` and `ゎ` as a pause, and splits
   `ヵヶ` into two words (`KANA_TABLE`). Open JTalk also reads a small kana of
   the other script after a kana as one mora (`ひェ` is `hy e`, `チぇ` is
   `ch e`, `トぅ` is `t u`); jpreprocess reads two (`SMALL_KANA`). Dictionary
   words such as `一ヶ月` or `ヶ月` are not affected.
7. **Chained word after an in-segment pause (not fixed).** When a word with
   `chain_flag` 1 follows a pause inside the segment (`八た・電ヘ`, `ぐ々会`),
   Open JTalk keeps it in the accent phrase before the pause; jpreprocess
   starts a new accent phrase (`CHAIN_AFTER_PAUSE`).
8. **User dictionary rows without any katakana mora (not fixed).** `ceo`,
   `cnn`, `gdp`, `hbo`, `jfet`, `suvs`, `yomiuri` and `churchhouse` (`pron`
   ` チャーチハウス` starts with a space) give no phoneme in Open JTalk. jpreprocess
   fills an empty pronunciation from the surface, so they are spelled out
   (`USERDICT_NO_MORA`).
9. **Unvoicing inside truncated user dictionary rows (not fixed).** Open JTalk
   unvoices no vowel in a word whose raw `pron` it cannot fully parse;
   jpreprocess unvoices the kept prefix. Only `pyopenjtalk.g2p` shows this,
   because `pyopenjtalk_g2p_prosody` lowercases unvoiced vowels
   (`USERDICT_UNVOICED`).

Numerals, dates and counters (`2024年3月15日`, `1,234,567`, `3本`, `6杯`, `1日`,
`20日`, `二十歳`), accent phrases and accent types matched on every corpus
input. No difference in dictionary entries was seen.

## Fixtures

`gen_fixtures.py` runs the upstream module with pyopenjtalk from the crate's
pixi environment and writes `crates/ports/gpt_sovits_text/tests/fixtures/japanese/`:

```
pixi run --manifest-path crates/ports/gpt_sovits_text/pixi.toml fixtures japanese
```

It compiles the user dictionary into a temporary directory and loads it, then
makes the import-time `mecab_dict_index` call of `japanese.py` fail (upstream
swallows the exception), so the upstream checkout is not written to.

| File | Inputs | Layers |
|---|---|---|
| `corpus.jsonl` | `parity/corpus/gpt_sovits_ja.json` (457 sentences) | text_normalize, g2p (both modes), and per segment NJD, labels, `pyopenjtalk.g2p`, `pyopenjtalk_g2p_prosody` |
| `bulk_phrase.jsonl` | 600 recombined corpus fragments with punctuation, seed 20240315 | same, without NJD and labels |
| `bulk_random.jsonl` | 1000 random strings over kana, kanji, digits, Latin, punctuation, whitespace | same, without NJD and labels |
| `bulk_userdict.jsonl` | 900 random `userdict.csv` surfaces, all 460 rows with non-katakana `pron`, 100 fullwidth surfaces, 100 surfaces inside Japanese | NJD, `pyopenjtalk.g2p`, `pyopenjtalk_g2p_prosody`, g2p |
| `text_normalize.jsonl` | 1000 random strings | `text_normalize`, `replace_consecutive_punctuation`, `symbols_to_japanese` |
| `post_replace_ph.jsonl` | every key of `rep_map` and some others | `post_replace_ph` |

## Results

Exact matches (`crates/ports/gpt_sovits_text/tests/japanese.rs`; every
mismatch is listed there by cause, and the test fails on any change):

| Fixture | Layer | Exact |
|---|---|---|
| corpus | text_normalize | 457/457 |
| corpus | g2p | 456/457 |
| corpus | g2p(with_prosody=False) | 456/457 |
| corpus | NJD (normalized) | 604/610 |
| corpus | NJD (raw, without `orig`) | 460/610 |
| corpus | labels without `/B:` `/C:` `/D:` | 609/610 |
| corpus | labels (full) | 222/610 |
| corpus | pyopenjtalk.g2p | 609/610 |
| corpus | pyopenjtalk_g2p_prosody | 609/610 |
| bulk_phrase | g2p | 598/600 |
| bulk_phrase | pyopenjtalk_g2p_prosody | 1562/1564 |
| bulk_random | g2p | 963/1000 |
| bulk_random | pyopenjtalk_g2p_prosody | 1446/1484 |
| bulk_userdict | g2p | 1536/1544 |
| bulk_userdict | NJD (normalized) | 1281/1544 |
| bulk_userdict | pyopenjtalk.g2p | 1507/1544 |
| text_normalize | all three functions | 1000/1000 |
| post_replace_ph | | 16/16 |

The only corpus mismatch is `ヵヶ`; the bulk_phrase mismatches also contain
`ヵヶ`. For comparison, an earlier run with jpreprocess's own lindera
tokenizer, no `ー` handling, and only the user dictionary rows that fail to
build cut gave 454/457 g2p on the corpus and 1512/1544 on the user dictionary
words.

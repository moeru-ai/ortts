"""Generate the golden fixtures of the `pypinyin` port crate.

Writes tests/fixtures/*.json of the crate from the pinned pypinyin 0.55.0
in the crate pixi workspace.

Usage: pixi run --manifest-path crates/ports/pypinyin/pixi.toml fixtures

Compact encodings (see crates/ports/pypinyin/tests/parity.rs):

- "joined" outputs (chars.json, phrases.json, syllables.json,
  tone_convert.json) are strings. For `pinyin()` results the items are joined
  with " " and the readings of one item with ","; for `lazy_pinyin()` and
  single-string results the strings are joined with " ". The generator
  asserts that no output string contains a separator and that no result is an
  empty list, so the encoding is unambiguous.
- sentences.json uses the "sep" encoding, because sentences contain spaces
  and commas: items are joined with ITEM_SEP and readings with READ_SEP, an
  empty result is EMPTY, and a Python exception is ERROR_PREFIX + its type
  name.
- syllables.json and tone_convert.json store {"same_as": i} instead of an
  outputs string that equals the outputs of case i of the same group.
- edge_cases.json stores full JSON values.
"""

import json
import random
import re
import sys
from pathlib import Path

import pypinyin
from pypinyin import Style, lazy_pinyin, pinyin
from pypinyin.constants import PHRASES_DICT, PINYIN_DICT, RE_HANS
from pypinyin.contrib import tone_convert as tc
from pypinyin.converter import DefaultConverter, UltimateConverter
from pypinyin.core import Pinyin
from pypinyin.seg import mmseg
from pypinyin.seg.simpleseg import seg as simpleseg_seg, simple_seg

assert pypinyin.__version__ == "0.55.0", pypinyin.__version__

CRATE = Path(__file__).resolve().parents[1]
OUT = CRATE / "tests" / "fixtures"
SEED = 20250723

ALL_STYLES = [
    "NORMAL", "TONE", "TONE2", "TONE3", "INITIALS", "FIRST_LETTER",
    "FINALS", "FINALS_TONE", "FINALS_TONE2", "FINALS_TONE3",
]
STRICT_STYLES = ["INITIALS", "FINALS", "FINALS_TONE", "FINALS_TONE2", "FINALS_TONE3"]
NUMBER_STYLES = ["TONE2", "TONE3", "FINALS_TONE2", "FINALS_TONE3"]


def cfg(fn, style, heteronym=False, strict=True, errors="default", v_to_u=False, n5=False):
    return {
        "fn": fn,
        "style": style,
        "heteronym": heteronym,
        "strict": strict,
        "errors": errors,
        "v_to_u": v_to_u,
        "neutral_tone_with_five": n5,
    }


def run(c, hans):
    try:
        return _run(c, hans)
    except TypeError:
        # NeutralToneWith5Mixin: right_mark_index() returns None for e.g. the
        # FINALS_TONE2 non-strict final "g" of "ng". The Rust port panics.
        return {"error": "TypeError"}


def _run(c, hans):
    style = Style[c["style"]]
    if c["fn"] == "pinyin":
        return pinyin(
            hans, style=style, heteronym=c["heteronym"], errors=c["errors"],
            strict=c["strict"], v_to_u=c["v_to_u"],
            neutral_tone_with_five=c["neutral_tone_with_five"])
    assert c["fn"] == "lazy_pinyin" and not c["heteronym"]
    return lazy_pinyin(
        hans, style=style, errors=c["errors"], strict=c["strict"],
        v_to_u=c["v_to_u"], neutral_tone_with_five=c["neutral_tone_with_five"])


def check_atom(s):
    assert isinstance(s, str) and " " not in s and "," not in s and "\n" not in s, repr(s)
    return s


def encode(fn, result):
    assert result != [], "empty result cannot be encoded"
    if fn == "pinyin":
        out = []
        for item in result:
            assert item != []
            out.append(",".join(check_atom(x) for x in item))
        return " ".join(out)
    return " ".join(check_atom(x) for x in result)


ITEM_SEP = "`"
READ_SEP = "{"
EMPTY = "\u2205"
ERROR_PREFIX = "\u2297"


def check_sep_atom(s):
    assert isinstance(s, str) and ITEM_SEP not in s and READ_SEP not in s and s != EMPTY, repr(s)
    assert ERROR_PREFIX not in s, repr(s)
    return s


def encode_sep(fn, result):
    if isinstance(result, dict):
        return ERROR_PREFIX + result["error"]
    if result == []:
        return EMPTY
    if fn == "pinyin":
        out = []
        for item in result:
            assert item != []
            out.append(READ_SEP.join(check_sep_atom(x) for x in item))
        return ITEM_SEP.join(out)
    return ITEM_SEP.join(check_sep_atom(x) for x in result)


def dedupe_cases(cases):
    """Replace an outputs value that an earlier case already has."""
    seen = {}
    for i, case in enumerate(cases):
        key = json.dumps(case["outputs"], ensure_ascii=False)
        if key in seen:
            del case["outputs"]
            case["same_as"] = seen[key]
        else:
            seen[key] = i
    return cases


def encode_atoms(values):
    return " ".join(check_atom(x) for x in values)


def dump(name, data):
    path = OUT / name
    path.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"{name}: {path.stat().st_size / 1e6:.2f} MB")


# Configurations that GPT-SoVITS uses, plus the defaults.
GPT_SOVITS_CONFIGS = [
    # g2pw/onnx_api.py
    cfg("pinyin", "TONE3", n5=True),
    # chinese.py / chinese2.py / tone_sandhi.py
    cfg("lazy_pinyin", "INITIALS", n5=True),
    cfg("lazy_pinyin", "FINALS_TONE3", n5=True),
    # g2pw/g2pw.py falls back to UltimateConverter.convert(..., Style.TONE, ...);
    # heteronym=True also covers the raw dictionary data.
    cfg("pinyin", "TONE", heteronym=True),
]


# The bulk dictionary fixtures keep the size down: the TONE output is the raw
# dictionary data, which TONE3 with heteronym=True (chars) or TONE3 (phrases)
# determines exactly, and syllables.json checks every style per syllable.
CHAR_CONFIGS = [
    cfg("pinyin", "TONE3", heteronym=True, n5=True),
    cfg("lazy_pinyin", "INITIALS", n5=True),
    cfg("lazy_pinyin", "FINALS_TONE3", n5=True),
]
PHRASE_CONFIGS = [
    cfg("pinyin", "TONE3", n5=True),
    cfg("lazy_pinyin", "INITIALS", n5=True),
    cfg("lazy_pinyin", "FINALS_TONE3", n5=True),
]


def gen_chars():
    chars = "".join(chr(k) for k in sorted(PINYIN_DICT))
    cases = []
    for c in CHAR_CONFIGS:
        cases.append({"config": c, "outputs": [encode(c["fn"], run(c, ch)) for ch in chars]})
    for case in cases:
        case["outputs"] = "\n".join(case["outputs"])
    dump("chars.json", {"inputs": chars, "cases": cases})


def gen_phrases(rng):
    keys = sorted(PHRASES_DICT)
    sample = rng.sample(keys, 20000)
    # Always include the phrases with several readings per character.
    for k in keys:
        if any(len(i) > 1 for i in PHRASES_DICT[k]) and k not in sample:
            sample.append(k)
    configs = PHRASE_CONFIGS
    cases = []
    for c in configs:
        cases.append({"config": c, "outputs": "\n".join(encode(c["fn"], run(c, p)) for p in sample)})
    dump("phrases.json", {"inputs": "\n".join(sample), "cases": cases})


# --- mixed sentences -------------------------------------------------------

PUNCT = list("，。！？、；：“”‘’（）《》【】—…·,.!?;:'\"()[]-~@#$%^&*_+=/\\|<>")
LATIN = ["hello", "World", "GPT", "SoVITS", "ok", "A", "x", "iPhone", "CPU", "v", "lv", "nü", "abc"]
DIGITS = list("0123456789") + ["2024", "3.14", "１２３", "٣", "Ⅻ", "½"]
SPACE = [" ", "  ", "\t", "\n", "\u3000"]
EMOJI = ["😀", "👍", "🎉", "❤\ufe0f", "🇨🇳", "👨\u200d👩\u200d👧", "🐱", "🌸"]
OTHER = ["ã", "é", "ß", "Ω", "あ", "カ", "한", "ㄅ", "〇", "\ue815", "\U00020000", "\U0002a6df", "\U00030edd", "\U0003134a",
         "\u3005", "\uff21", "￥", "\u2028", "\x00"]
COMMON_HAN = (
    "的一是不了人我在有他这中大来上国个到说们为子和你地出道也时年得就那要下以生会自着去之过家学对可她里后小么心多"
    "天而能好都然没日于起还发成事只作当想看文无开手十用主行方又如前所本见经头面公同三已老从动两长知民样现分将外但"
    "身些与高意进把法此实回二理美点月明其种声全工己话儿者向情部正名定女问力机给等几很业最间新什打便位因重被走电四"
    "第门相次东政海口使教西再平真听世气信北少关并内加化由却代军产入先山五太水万市眼体别处总才场师书比住员九笑性通"
    "目华报立马命张活难神数件安表原车白应路期叫死常提感金何更反合放做系计或司利受光王果亲界及今京务制解各任至清物"
    "台象记边共风战干接它许八特觉望直服毛林题建南度统色字请交爱让认算论百吃义科怎元社术结六功指思非流每青管夫连远"
    "资队跟带花快条院变联言权往展该领传近留红治决周保达办运武半候七必城父强步完革深区即求品士转量空甚众技轻程告江"
    "语英基派满式李息写呢识极令黄德收脸钱党倒未持取设始版双历越史商千片容研像找友孩站广改议形委早房音火际则首单据"
    "朝阳那些行长重庆银行音乐还有了解调查参差不齐睡觉觉得一会儿好好地什么东西"
)
POLYPHONIC = ["行", "长", "重", "乐", "还", "了", "调", "参", "觉", "得", "地", "着", "朝", "那", "和", "差", "种", "传", "便", "降"]


def rand_han_run(rng, phrase_keys, single_keys):
    r = rng.random()
    if r < 0.45:
        return rng.choice(phrase_keys)
    if r < 0.75:
        return "".join(rng.choice(COMMON_HAN) for _ in range(rng.randint(1, 6)))
    if r < 0.85:
        return "".join(rng.choice(POLYPHONIC) for _ in range(rng.randint(1, 4)))
    # rare characters, including ones that are not in PINYIN_DICT but match RE_HANS
    s = ""
    for _ in range(rng.randint(1, 3)):
        if rng.random() < 0.7:
            s += chr(rng.choice(single_keys))
        else:
            s += chr(rng.choice(NO_PINYIN_HANS))
    return s


def rand_other(rng):
    r = rng.random()
    if r < 0.3:
        return rng.choice(PUNCT)
    if r < 0.5:
        return rng.choice(LATIN)
    if r < 0.65:
        return rng.choice(DIGITS)
    if r < 0.75:
        return rng.choice(SPACE)
    if r < 0.88:
        return rng.choice(EMOJI)
    return rng.choice(OTHER)


def gen_sentence(rng, phrase_keys, single_keys):
    parts = []
    for _ in range(rng.randint(1, 8)):
        if rng.random() < 0.65:
            parts.append(rand_han_run(rng, phrase_keys, single_keys))
        else:
            parts.append(rand_other(rng))
    return "".join(parts)


def han_without_pinyin():
    out = []
    ranges = [(0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF), (0xE815, 0xE864),
              (0x20000, 0x2A6DF), (0x2B740, 0x2B81D), (0x30000, 0x3134A)]
    for a, b in ranges:
        for cp in range(a, b + 1):
            if cp not in PINYIN_DICT and RE_HANS.match(chr(cp)):
                out.append(cp)
    return out


NO_PINYIN_HANS = han_without_pinyin()

SENTENCE_CONFIGS = GPT_SOVITS_CONFIGS
# Every style / strict / v_to_u / neutral_tone_with_five combination is
# checked per syllable in syllables.json. The sentence subset checks the rest
# of the pipeline (segmentation, heteronym, errors) with a few of them.
SUBSET_CONFIGS = (
    [cfg("pinyin", "TONE2", heteronym=True), cfg("pinyin", "FIRST_LETTER"),
     cfg("pinyin", "FINALS", strict=False, heteronym=True),
     cfg("pinyin", "FINALS_TONE2", strict=False, heteronym=True, v_to_u=True, n5=True),
     cfg("lazy_pinyin", "NORMAL"), cfg("lazy_pinyin", "NORMAL", v_to_u=True), cfg("lazy_pinyin", "TONE2", n5=True),
     cfg("lazy_pinyin", "INITIALS", strict=False)]
    + [cfg("pinyin", "TONE3", n5=True, errors=e) for e in ("ignore", "replace")]
    + [cfg("pinyin", "TONE", heteronym=True, errors=e) for e in ("ignore", "replace")]
    + [cfg("lazy_pinyin", "FINALS_TONE3", n5=True, errors=e) for e in ("ignore", "replace")]
)


SUBSET_SIZE = 400


def gen_sentences(rng):
    phrase_keys = sorted(PHRASES_DICT)
    single_keys = sorted(PINYIN_DICT)
    sentences = [gen_sentence(rng, phrase_keys, single_keys) for _ in range(2000)]
    cases = []
    for i, s in enumerate(sentences):
        assert ITEM_SEP not in s and READ_SEP not in s and ERROR_PREFIX not in s
        configs = SENTENCE_CONFIGS + (SUBSET_CONFIGS if i < SUBSET_SIZE else [])
        cases.append({"text": s, "outputs": [encode_sep(c["fn"], run(c, s)) for c in configs]})
    dump("sentences.json", {
        "configs": SENTENCE_CONFIGS,
        "subset_configs": SUBSET_CONFIGS,
        "subset_size": SUBSET_SIZE,
        "cases": cases,
    })


# --- syllable level style conversion -------------------------------------

def all_dict_syllables():
    s = set()
    for v in PINYIN_DICT.values():
        s.update(v.split(","))
    for v in PHRASES_DICT.values():
        for item in v:
            s.update(item)
    return sorted(s)


EDGE_SYLLABLES = [
    "", "m", "n", "ng", "hm", "hng", "ê", "ê̄", "ế", "ê̌", "ề", "m̄", "ḿ", "m̀", "ń", "ň", "ǹ",
    "yo", "yō", "yu", "yū", "yi", "y", "w", "wu", "wū", "wa", "yue", "yüè", "lüè", "lvè", "nü",
    "jü", "ju", "qu", "xu", "jū", "jú", "jǔ", "jù", "lü", "lǘ", "er", "ēr", "ér",
    "zhong\n", "jū\n", "guī\n", "niú\n", "lùn\n", "ju\nx", "a\nb", "\n",
    "zh", "ch", "sh", "z", "abc", "ńg", "ňg", "ǹg", "hḿ", "xiou", "guei", "luen", "iou", "uei", "uen",
    "ā", "āō", "zhōng1", "zho1ng", "zhong5", "zhong0", "a5", "5", "１", "zhōng٣",
]


def convert_style_or_error(conv, s, style, strict):
    try:
        return conv.convert_style("", s, style, strict)
    except TypeError:
        # NeutralToneWith5Mixin: right_mark_index() returned None.
        return "!TypeError"
    except IndexError:
        # FIRST_LETTER of a pinyin whose NORMAL form is empty (e.g. "5").
        return "!IndexError"


def gen_syllables():
    # "\n" edge cases are in tone_convert.json (they cannot be "\n"-joined).
    inputs = all_dict_syllables() + [e for e in EDGE_SYLLABLES if e and "\n" not in e]
    # FIRST_LETTER raises on an empty pinyin; "" is covered by tone_convert.json.
    configs = []
    for style in ALL_STYLES:
        for strict in ((True, False) if style in STRICT_STYLES else (True,)):
            for v_to_u, n5 in ((False, False), (True, True), (False, True), (True, False)):
                if (v_to_u, n5) in ((False, True), (True, False)) and style not in NUMBER_STYLES + ["NORMAL", "FINALS"]:
                    continue
                configs.append({"style": style, "strict": strict, "v_to_u": v_to_u, "neutral_tone_with_five": n5})
    cases = []
    for c in configs:
        conv = UltimateConverter(v_to_u=c["v_to_u"], neutral_tone_with_five=c["neutral_tone_with_five"])
        outs = [convert_style_or_error(conv, s, Style[c["style"]], c["strict"]) for s in inputs]
        assert all("\n" not in o for o in outs)
        cases.append({"config": c, "outputs": "\n".join(outs)})
    for case in cases:
        assert len(case["outputs"].split("\n")) == len(inputs)
    dump("syllables.json", {"inputs": inputs, "cases": dedupe_cases(cases)})


# --- tone_convert -----------------------------------------------------------

TC_TO = [
    ("to_normal", {}), ("to_normal", {"v_to_u": True}),
    ("to_tone", {}),
    ("to_tone2", {}), ("to_tone2", {"v_to_u": True, "neutral_tone_with_five": True}),
    ("to_tone3", {}), ("to_tone3", {"v_to_u": True, "neutral_tone_with_five": True}),
    ("to_tone3", {"neutral_tone_with_five": True}),
    ("to_initials", {}), ("to_initials", {"strict": False}),
    ("to_finals", {}), ("to_finals", {"strict": False, "v_to_u": True}),
    ("to_finals_tone", {}), ("to_finals_tone", {"strict": False}),
    ("to_finals_tone2", {}), ("to_finals_tone2", {"strict": False, "v_to_u": True, "neutral_tone_with_five": True}),
    ("to_finals_tone3", {}), ("to_finals_tone3", {"neutral_tone_with_five": True}),
    ("to_finals_tone3", {"strict": False, "v_to_u": True, "neutral_tone_with_five": True}),
]
TC_TONE = [
    ("tone_to_normal", {}), ("tone_to_normal", {"v_to_u": True}),
    ("tone_to_tone2", {}), ("tone_to_tone2", {"v_to_u": True, "neutral_tone_with_five": True}),
    ("tone_to_tone3", {}), ("tone_to_tone3", {"v_to_u": True, "neutral_tone_with_five": True}),
]
TC_TONE2 = [
    ("tone2_to_normal", {}), ("tone2_to_normal", {"v_to_u": True}),
    ("tone2_to_tone", {}),
    ("tone2_to_tone3", {}), ("tone2_to_tone3", {"v_to_u": True}),
]
TC_TONE3 = [
    ("tone3_to_normal", {}), ("tone3_to_normal", {"v_to_u": True}),
    ("tone3_to_tone", {}),
    ("tone3_to_tone2", {}), ("tone3_to_tone2", {"v_to_u": True}),
]


def gen_tone_convert():
    tone = all_dict_syllables()
    bases = sorted({tc.to_normal(s) for s in tone} | {tc.to_normal(s, v_to_u=True) for s in tone})
    tone3 = [b + str(t) for b in bases for t in range(1, 6)]
    tone3_set = set(tone3)
    gen_tone = sorted({tc.to_tone(s) for s in tone3} - set(tone))
    tone2 = sorted({tc.to_tone2(s, neutral_tone_with_five=True) for s in tone3} - tone3_set)
    edge = EDGE_SYLLABLES
    groups = [
        ("tone", tone + gen_tone, TC_TO + TC_TONE),
        ("tone3", tone3, TC_TO + TC_TONE3),
        ("tone2", tone2, [x for x in TC_TO if x[0] in ("to_tone", "to_tone3", "to_normal", "to_finals_tone3")] + TC_TONE2),
        ("edge", edge, TC_TO + TC_TONE + TC_TONE2 + TC_TONE3),
    ]
    out = []
    for name, inputs, funcs in groups:
        cases = []
        for fn, kwargs in funcs:
            f = getattr(tc, fn)
            outs = [f(s, **kwargs) for s in inputs]
            cases.append({"fn": fn, "kwargs": kwargs, "outputs": outs if name == "edge" else "\n".join(outs)})
            if name != "edge":
                assert all("\n" not in o for o in outs), fn
        if name != "edge":
            assert all("\n" not in s for s in inputs)
        out.append({"group": name, "inputs": inputs if name == "edge" else "\n".join(inputs),
                    "cases": dedupe_cases(cases)})
    dump("tone_convert.json", out)


# --- edge cases: segmentation, list input, errors, subclassing ---------------

EDGE_TEXTS = [
    "", " ", "\n", "中", "中\n", "中国\n", "\n中", "a", "abc", "你好", "你好吗", "你好，世界！",
    "我是中国人，我爱我的祖国", "朝阳", "那些", "长城很长", "重庆的重量", "银行行长", "音乐乐队",
    "一丁点儿", "一不小心", "123中文abc", "中文123", "👍好的😀", "〇", "\ue815\ue864", "\U00020000",
    "\U00030edd\U00030ede", "\u4dc0", "\u9fff", "㐀", "ㄅㄆㄇ", "あいう", "中\u3000国", "中\t国",
    "zhong国", "国zhong", "了解了", "还有还", "觉觉得", "一会儿", "好好地", "参差不齐",
    "睡觉觉得什么东西", "中华人民共和国", "中华人民共和国国歌", "朝阳区那些行长",
    "\x00中", "中\u2028国", "!!中文", "中!?a", "好吗？", "中国很好吗", "战略", "女", "绿", "旅游", "月亮", "云", "鱼", "嗯", "呣", "噷", "欸",
]


def callable_errors(chars):
    # Deterministic callback, reimplemented in the Rust test.
    n = len(chars)
    if n % 4 == 0:
        return None
    if n % 4 == 1:
        return "<" + chars + ">"
    if n % 4 == 2:
        return ["#" + c for c in chars]
    return [["@" + c, "&" + c] for c in chars]


class NoPinyinOverrideConverter(UltimateConverter):
    """Overrides the no-pinyin hooks; reimplemented in the Rust test."""

    def pre_handle_nopinyin(self, chars, style, heteronym, errors, strict, **kwargs):
        if chars.startswith("!"):
            return ["bang"] * len(chars)
        return None

    def post_handle_nopinyin(self, chars, style, heteronym, errors, strict, pinyin, **kwargs):
        if chars.isdigit():
            return [["d" + c, "D" + c] for c in chars]
        return None

    def post_pinyin(self, han, heteronym, pinyin, **kwargs):
        data = super().post_pinyin(han, heteronym, pinyin, **kwargs)
        if han == "中国":
            return [["zhòng"], ["guó"]]
        return data

    def pre_convert_style(self, han, orig_pinyin, style, strict, **kwargs):
        if han == "好":
            return "hào"
        return None

    def post_convert_style(self, han, orig_pinyin, converted_pinyin, style, strict, **kwargs):
        data = super().post_convert_style(han, orig_pinyin, converted_pinyin, style, strict, **kwargs)
        if han == "吗":
            return data.upper()
        return data


class SimpleSegPinyin(Pinyin):
    """Like GPT-SoVITS G2PWPinyin.get_seg."""

    def get_seg(self, **kwargs):
        return simple_seg


class PrePostSegPinyin(Pinyin):
    def pre_seg(self, hans, **kwargs):
        if hans.startswith("#"):
            return list(hans[1:])
        return None

    def post_seg(self, hans, seg_data, **kwargs):
        if len(seg_data) > 2:
            return seg_data[::-1]
        return None


# Copy of GPT-SoVITS GPT_SoVITS/text/g2pw/g2pw.py Converter (commit 04a44eb)
# with a deterministic fake G2PW model; reimplemented in the Rust test.
def _remove_dup_items(lst, remove_empty=False):
    new_lst = []
    for item in lst:
        if remove_empty and not item:
            continue
        if item not in new_lst:
            new_lst.append(item)
    return new_lst


def _remove_dup_and_empty(lst_list):
    new_lst_list = []
    for lst in lst_list:
        lst = _remove_dup_items(lst, remove_empty=True)
        if lst:
            new_lst_list.append(lst)
        else:
            new_lst_list.append([""])
    return new_lst_list


def fake_g2pw(han):
    if len(han) % 5 == 0:
        return []
    row = []
    for c in han:
        if ord(c) % 3 == 0 or ord(c) not in PINYIN_DICT:
            row.append(None)
        else:
            row.append(tc.to_tone3(PINYIN_DICT[ord(c)].split(",")[-1]))
    return [row]


class G2PWLikeConverter(UltimateConverter):
    def __init__(self, g2pw_instance, v_to_u=False, neutral_tone_with_five=False, tone_sandhi=False, **kwargs):
        super().__init__(v_to_u=v_to_u, neutral_tone_with_five=neutral_tone_with_five, tone_sandhi=tone_sandhi, **kwargs)
        self._g2pw = g2pw_instance

    def convert(self, words, style, heteronym, errors, strict, **kwargs):
        pys = []
        if RE_HANS.match(words):
            pys = self._to_pinyin(words, style=style, heteronym=heteronym, errors=errors, strict=strict)
            post_data = self.post_pinyin(words, heteronym, pys)
            if post_data is not None:
                pys = post_data
            pys = self.convert_styles(pys, words, style, heteronym, errors, strict)
        else:
            py = self.handle_nopinyin(words, style=style, errors=errors, heteronym=heteronym, strict=strict)
            if py:
                pys.extend(py)
        return _remove_dup_and_empty(pys)

    def _to_pinyin(self, han, style, heteronym, errors, strict, **kwargs):
        pinyins = []
        g2pw_pinyin = self._g2pw(han)
        if not g2pw_pinyin:
            return super().convert(han, Style.TONE, heteronym, errors, strict, **kwargs)
        for i, item in enumerate(g2pw_pinyin[0]):
            if item is None:
                py = super().convert(han[i], Style.TONE, heteronym, errors, strict, **kwargs)
                pinyins.extend(py)
            else:
                pinyins.append([tc.to_tone(item)])
        return pinyins


class G2PWLikePinyin(Pinyin):
    def __init__(self, v_to_u=False, neutral_tone_with_five=False):
        self._converter = G2PWLikeConverter(fake_g2pw, v_to_u=v_to_u, neutral_tone_with_five=neutral_tone_with_five)

    def get_seg(self, **kwargs):
        return simple_seg


def gen_edge_cases(rng):
    phrase_keys = sorted(PHRASES_DICT)
    single_keys = sorted(PINYIN_DICT)
    texts = EDGE_TEXTS + [gen_sentence(rng, phrase_keys, single_keys) for _ in range(80)]

    # The Rust NoPinyinOverrideConverter implements str.isdigit() with \d.
    re_digit = re.compile(r"\d")
    for t in texts:
        for ch in t:
            assert ch.isdigit() == bool(re_digit.match(ch)), repr(ch)

    cases = []

    def add(kind, args, result):
        cases.append({"kind": kind, "args": args, "result": result})

    for t in texts:
        add("simple_seg", {"hans": t}, simple_seg(t))
        add("seg", {"hans": t}, simpleseg_seg(t))
        add("mmseg", {"hans": t}, list(mmseg.seg.cut(t)))
        add("re_hans", {"hans": t}, bool(RE_HANS.match(t)))

    # Non-strict mmseg (no_non_phrases=False) over a small custom prefix set.
    custom = mmseg.PrefixSet()
    custom.train(["中国", "中国人", "中华人民", "人民", "我爱"])
    loose = mmseg.Seg(custom, no_non_phrases=False)
    strict_custom = mmseg.Seg(custom, no_non_phrases=True)
    for t in texts:
        add("mmseg_custom", {"hans": t, "no_non_phrases": False}, list(loose.cut(t)))
        add("mmseg_custom", {"hans": t, "no_non_phrases": True}, list(strict_custom.cut(t)))

    # List input.
    for t in texts[:100]:
        parts = []
        i = 0
        while i < len(t):
            n = rng.randint(1, 4)
            parts.append(t[i:i + n])
            i += n
        if rng.random() < 0.1:
            parts.append("")
        add("simple_seg_list", {"hans": parts}, simple_seg(parts))
        for style in ("TONE", "TONE3", "FINALS_TONE3"):
            add("pinyin_list", {"hans": parts, "style": style, "converter": "ultimate"},
                pinyin(parts, style=Style[style], neutral_tone_with_five=True, heteronym=True))
            add("pinyin_list", {"hans": parts, "style": style, "converter": "default"},
                Pinyin(DefaultConverter()).pinyin(parts, style=Style[style], heteronym=True))
            add("lazy_pinyin_list", {"hans": parts, "style": style},
                lazy_pinyin(parts, style=Style[style], neutral_tone_with_five=True))

    # errors as a callable, and the converter / Pinyin hooks.
    for t in texts:
        for h in (False, True):
            add("callable_errors", {"hans": t, "heteronym": h},
                pinyin(t, style=Style.TONE3, heteronym=h, errors=callable_errors))
        add("hooks_converter", {"hans": t},
            Pinyin(NoPinyinOverrideConverter(neutral_tone_with_five=True)).pinyin(t, style=Style.TONE3, heteronym=True))
        add("simple_seg_pinyin", {"hans": t},
            SimpleSegPinyin(UltimateConverter(neutral_tone_with_five=True)).pinyin(t, style=Style.TONE3))
        add("pre_post_seg_pinyin", {"hans": t},
            PrePostSegPinyin(UltimateConverter()).lazy_pinyin(t, style=Style.TONE))
        add("pre_post_seg_pinyin", {"hans": "#" + t},
            PrePostSegPinyin(UltimateConverter()).lazy_pinyin("#" + t, style=Style.TONE))
        for style in ("TONE3", "FINALS_TONE3", "INITIALS"):
            add("g2pw_like", {"hans": t, "style": style},
                G2PWLikePinyin(neutral_tone_with_five=True).pinyin(t, style=Style[style]))
        add("default_converter", {"hans": t},
            Pinyin().pinyin(t, style=Style.TONE3, heteronym=True))
        try:
            r = pinyin(t, errors="replace")
        except TypeError:
            r = None
        add("replace_errors", {"hans": t}, r)

    dump("edge_cases.json", cases)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    gen_chars()
    gen_phrases(rng)
    gen_sentences(rng)
    gen_syllables()
    gen_tone_convert()
    gen_edge_cases(rng)


if __name__ == "__main__":
    sys.exit(main())

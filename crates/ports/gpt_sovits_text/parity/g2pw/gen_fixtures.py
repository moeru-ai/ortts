"""Generate golden fixtures for the `gpt_sovits_text::g2pw` port.

    pixi run --manifest-path crates/ports/gpt_sovits_text/pixi.toml fixtures g2pw

Writes `crates/ports/gpt_sovits_text/tests/fixtures/g2pw/`:

- `tables.json`: the converter tables built from the model dir (config after the
  defaults, labels, chars, char2phonemes, polyphonic_chars_new, monophonic dict size),
  and the size and FNV-1a 64 hash of `pp_dict`.
- `opencc.jsonl`: `OpenCC("s2tw").convert(text)` (the instance the converter holds).
- `tokenizer.jsonl`: `tokenizer.tokenize(word)` for every word `wordize_and_map` produces
  from the inputs below, plus random code points.
- `call.jsonl`: `g2pw._g2pw(batch)` (`G2PWOnnxConverter.__call__`, the call chinese2.py
  makes) on batches of sentences, for the default settings and for variants of
  `enable_sentence_dedup` / `polyphonic_context_chars`.
- `model.jsonl`: for a subset, `_prepare_data`, `prepare_onnx_input` and the raw model
  output of `_predict` (argmax labels, confidences, and full probabilities for a few rows).
- `pinyin.jsonl`: `G2PWPinyin.pinyin` / `lazy_pinyin` (TONE3, INITIALS, FINALS_TONE3).
- `chinese2.jsonl`: for every zh item of parity/corpus/gpt_sovits_zh_en_yue.json and
  some bulk sentences, the `g2pw._g2pw` batches and the `correct_pronunciation` calls that
  `chinese2.g2p(text_normalize(text))` makes, captured by wrapping both.
- `correct_pronunciation.jsonl`: hand-written and sampled `correct_pronunciation` cases.

The G2PW instance is `text.chinese2.g2pw`, constructed by chinese2.py exactly as GPT-SoVITS
does (`G2PWPinyin(model_dir="GPT_SoVITS/text/G2PWModel", model_source=bert_path,
v_to_u=False, neutral_tone_with_five=True)`, so OpenCC s2tw is on).
"""

import contextlib
import copy
import json
import os
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import gpt_sovits_env  # noqa: E402

CORPUS = gpt_sovits_env.CORPUS_DIR / "gpt_sovits_zh_en_yue.json"
OUT = gpt_sovits_env.FIXTURES / "g2pw"

os.environ.pop("g2pw_sentence_dedup", None)
os.environ.pop("g2pw_polyphonic_context_chars", None)

gpt_sovits_env.setup()

with contextlib.redirect_stdout(sys.stderr):
    from pypinyin import Style  # noqa: E402
    from pypinyin.constants import PHRASES_DICT  # noqa: E402

    from text import chinese2  # noqa: E402
    from text.g2pw import g2pw as g2pw_module  # noqa: E402
    from text.g2pw.dataset import prepare_onnx_input  # noqa: E402
    from text.g2pw.utils import wordize_and_map  # noqa: E402

SEED = 20251102
G2PW = chinese2.g2pw  # G2PWPinyin
CONV = G2PW._g2pw  # G2PWOnnxConverter

POLY = "行长乐觉重还地传处为好曾单着参朝调藏得发便模空和了都看教应假种量将血数恶几差系相缝难省薄卡给露尽背弹降奔散干划扎冲当中正称结丧盛宿兴转率模度载会少要强切着折更否观宁累的"
FILLER = "我你他她们是在有不人这那个一上来到大说时地出就分对成会可主发年动同工也能下过子民前面都然后自想去家看天心起所学还如样多经么十把只于小知道又生将事作方很最"
PUNCT_SINGLE = "，。！？、；："
PUNCT_RUNS = ["……", "——", "。“", "”，", "？！", "...", "!?", "（", "）", "《", "》", "「", "」"]
LATIN = ["AI", "GPT", "ok", "iPhone", "abc", "SoVITS", "x", "Q3"]
OTHER = ["😀", "🎉", "한국", "かな", "カタカナ", "İ", "ß", "é", "Ⅷ", "①", "½", "　", " ", "  ", "\t", "​", "⿰木木", "⿱⿰日月木", "丽", "爲", "臺灣", "後來", "乾燥", "頭髮"]


def dump_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")


def fnv1a64(s: str) -> str:
    h = 0xCBF29CE484222325
    for b in s.encode("utf-8"):
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


def dedup_rows(rows):
    """Store a list with repeated items as {"unique": [...], "index": [...]}."""
    unique, index, seen = [], [], {}
    for row in rows:
        key = json.dumps(row, ensure_ascii=False)
        if key not in seen:
            seen[key] = len(unique)
            unique.append(row)
        index.append(seen[key])
    return {"unique": unique, "index": index}


def err_name(e):
    return type(e).__name__


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(sys.stderr):
        return fn(*args, **kwargs)


def poly_phrases():
    out = []
    for phrase in PHRASES_DICT:
        if 2 <= len(phrase) <= 4 and any(c in POLY for c in phrase):
            out.append(phrase)
    out.sort()
    return out


def gen_sentences(rng, phrases, n):
    sents = []
    for k in range(n):
        parts = []
        wild = rng.random() < 0.12
        for _ in range(rng.randint(1, 6)):
            r = rng.random()
            if r < 0.6:
                parts.append(rng.choice(phrases))
            elif r < 0.8:
                parts.append("".join(rng.choice(FILLER) for _ in range(rng.randint(1, 4))))
            else:
                parts.append(rng.choice(POLY))
            if rng.random() < 0.4:
                parts.append(rng.choice(PUNCT_SINGLE))
            if wild and rng.random() < 0.5:
                parts.append(rng.choice(PUNCT_RUNS + LATIN + OTHER + [str(rng.randint(0, 999))]))
        s = "".join(parts)
        if k % 97 == 0:
            # long sentence: polyphonic characters far apart (crop window; every third one
            # is over 510 tokens, for the _truncate path)
            n_fill = rng.randint(520, 700) if k % 291 == 0 else rng.randint(40, 200)
            s = rng.choice(phrases) + "".join(rng.choice(FILLER) for _ in range(n_fill)) + rng.choice(phrases)
        sents.append(s)
    return sents


EDGE = [
    "",
    "，。",
    "你好",
    "银行行长",
    "重庆重新",
    "长长长长长",
    "还还还",
    "他长得很好看，长大了还要长高。",
    "快乐的音乐",
    "睡觉觉得",
    "我觉得这个地方的地道小吃很好吃",
    "行行好吧",
    "一不和",
    "攢似",
    "AI行业的发展",
    "abc 行 def",
    "行 行",
    "你好,.吗还",
    "了。“还",
    "还了abc",
    "한국의 행",
    "한국",
    "行한长",
    "长한，还한了",
    "行ᄀ长",
    "İ行",
    "行İ",
    "ß长",
    "⿰木木行长",
    "头发长了",
    "臺灣的銀行",
    "後來還是長大了",
    "丽行",
    "爲了",
    "a\x00b行",
    "行\x00",
    "１２３行长",
    "Ⅷ长",
    "😀长大😀",
    "长" * 600,
    "行" + "的" * 520 + "长",
    "行" + "的" * 30 + "长" + "的" * 30 + "重",
    "\t长　行​",
    "朝阳朝代",
    "调查调皮",
    "藏族宝藏",
    "得到得了",
    "发现头发",
    "方便便宜",
    "模样模型",
    "空间空闲",
    "和平和面",
    "了解完了",
    "都市都是",
    "单于单位",
    "曾经曾国藩",
    "着急着落看着",
    "参加人参",
    "传说传记",
    "处理处所",
    "为了作为",
    "好人爱好",
    "地方慢慢地",
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    phrases = poly_phrases()
    corpus = json.loads(CORPUS.read_text())
    zh_texts = [it["text"] for it in corpus if it["lang"] == "zh"]

    bulk = gen_sentences(rng, phrases, 3200)
    sentences = EDGE + zh_texts + bulk

    # --- tables ---
    cfg = {k: getattr(CONV.config, k) for k in vars(CONV.config) if not k.startswith("__")}
    pp = g2pw_module.pp_dict
    assert pp == g2pw_module.read_dict()
    canonical = json.dumps(sorted(pp.items()), ensure_ascii=False, separators=(",", ":"))
    tables = {
        "config": cfg,
        "model_source": CONV.model_source,
        "labels": CONV.labels,
        "chars": CONV.chars,
        "char2phonemes": CONV.char2phonemes,
        "polyphonic_chars_new": sorted(CONV.polyphonic_chars_new),
        "monophonic_chars_dict_len": len(CONV.monophonic_chars_dict),
        "monophonic_chars_dict_sample": dict(sorted(CONV.monophonic_chars_dict.items())[::50]),
        "bopomofo_convert_dict_len": len(CONV.bopomofo_convert_dict),
        "char_bopomofo_dict_len": len(CONV.char_bopomofo_dict),
        "enable_sentence_dedup": CONV.enable_sentence_dedup,
        "polyphonic_context_chars": CONV.polyphonic_context_chars,
        "enable_opencc": CONV.enable_opencc,
        "pp_dict_len": len(pp),
        "pp_dict_fnv1a64": fnv1a64(canonical),
        "convert_bopomofo_to_pinyin": {
            b: quiet(CONV._convert_bopomofo_to_pinyin, b)
            for b in sorted({p for _, p in CONV.polyphonic_chars} | {p for _, p in CONV.monophonic_chars})
        },
    }
    for bad in ["ㄅㄚ6", "ㄅ", "xyz1", "1"]:
        try:
            tables["convert_bopomofo_to_pinyin"][bad] = quiet(CONV._convert_bopomofo_to_pinyin, bad)
        except Exception as e:  # noqa: BLE001
            tables["convert_bopomofo_to_pinyin"][bad] = {"error": err_name(e)}
    (OUT / "tables.json").write_text(json.dumps(tables, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    # --- opencc ---
    cc = CONV.cc
    st_chars = [line.split("\t")[0] for line in (gpt_sovits_env.DATA / "g2pw/opencc/STCharacters.txt").read_text().splitlines()]
    tw_keys = [line.split("\t")[0] for f in ("TWVariants.txt", "TWVariantsPhrases.txt", "CJK_Compatibility_Ideographs.txt", "STPhrases_GeneratedFromRegionalPhrases.txt") for line in (gpt_sovits_env.DATA / "g2pw/opencc" / f).read_text().splitlines()]
    st_phrases = [line.split("\t")[0] for line in (gpt_sovits_env.DATA / "g2pw/opencc/STPhrases.txt").read_text().splitlines()]
    oc_inputs = list(sentences[:1500])
    oc_inputs += ["".join(st_chars[i : i + 60]) for i in range(0, len(st_chars), 60)]
    oc_inputs += ["".join(tw_keys[i : i + 20]) for i in range(0, len(tw_keys), 20)]
    oc_rng = random.Random(SEED + 1)
    for _ in range(1500):
        pieces = []
        for _ in range(oc_rng.randint(1, 8)):
            r = oc_rng.random()
            if r < 0.5:
                pieces.append(oc_rng.choice(st_phrases))
            elif r < 0.7:
                pieces.append(oc_rng.choice(st_chars))
            elif r < 0.8:
                pieces.append(oc_rng.choice(tw_keys))
            elif r < 0.9:
                pieces.append(oc_rng.choice(OTHER + PUNCT_RUNS + LATIN))
            else:
                p = oc_rng.choice(st_phrases)
                pieces.append(p[: oc_rng.randint(1, len(p))])
        oc_inputs.append("".join(pieces))
    oc_inputs += ["⿰", "⿰木", "⿰木木木", "⿲木木", "⿲木木木", "⿾木", "⿿⿿", "⿰" * 20 + "木" * 21, "⿰⿰⿰木木木木头发", "头⿰发发头", "a\x00头发", "头\x00发", "\x00", "头发\x00头发"]
    dump_jsonl(OUT / "opencc.jsonl", [{"text": t, "out": cc.convert(t)} for t in oc_inputs])

    # --- tokenizer ---
    words = set()
    for s in sentences + [cc.convert(s) for s in sentences[:2000]]:
        w, _, _ = wordize_and_map(s.lower())
        words.update(w)
    tok_rng = random.Random(SEED + 2)
    for _ in range(3000):
        block = tok_rng.choice([(0x20, 0x7F), (0xA0, 0x24F), (0x300, 0x36F), (0x370, 0x52F), (0x2000, 0x2BFF), (0x3000, 0x30FF), (0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xAC00, 0xD7A3), (0xF900, 0xFAFF), (0xFF00, 0xFFEF), (0x1F300, 0x1FAFF), (0x20000, 0x2A6DF), (0x0, 0x20)])
        words.add(chr(tok_rng.randint(*block)))
    for _ in range(300):
        words.add("".join(tok_rng.choice("abcdefghijklmnopqrstuvwxyz0123456789") for _ in range(tok_rng.randint(1, 30))))
    words = sorted(words)
    dump_jsonl(OUT / "tokenizer.jsonl", [{"word": w, "tokens": CONV.tokenizer.tokenize(w)} for w in words])

    # --- call ---
    def run_call(batch):
        try:
            return {"result": quiet(CONV, batch)}
        except Exception as e:  # noqa: BLE001
            return {"error": err_name(e)}

    # A sentence that raises on its own (mostly the IndexError of a non-Han run, see the
    # README) gets a batch of its own, so it does not hide the results of the others.
    fails = {s for s in sentences if "error" in run_call([s])}
    batches = []
    brng = random.Random(SEED + 3)
    pending = []
    for s in sentences:
        if s in fails:
            batches.append([s])
            continue
        pending.append(s)
        if len(pending) >= brng.randint(1, 4):
            batch = pending
            if brng.random() < 0.1:
                batch = batch + [brng.choice(batch)]
            batches.append(batch)
            pending = []
    if pending:
        batches.append(pending)
    call_records = []
    for b in batches:
        call_records.append({"variant": "default", "sentences": b, **run_call(b)})
    variants = [("nodedup", False, 16), ("ctx0", True, 0), ("ctx3", True, 3), ("nodedup_ctx0", False, 0)]
    # Variants on short batches only (without dedup, a long sentence is one 512-token row
    # per polyphonic character).
    short_batches = [b for b in batches if sum(len(s) for s in b) <= 300]
    for name, dedup, ctx in variants:
        CONV.enable_sentence_dedup, CONV.polyphonic_context_chars = dedup, ctx
        for b in short_batches[:120] + short_batches[-60:]:
            call_records.append({"variant": name, "sentences": b, **run_call(b)})
    CONV.enable_sentence_dedup, CONV.polyphonic_context_chars = True, 16
    dump_jsonl(OUT / "call.jsonl", call_records)

    # --- model ---
    model_records = []
    full_prob_rows = 0
    model_inputs = [s for s in sentences[:500] + sentences[-200:] if len(s) <= 150][::2]
    model_inputs.append("行" + "的" * 520 + "长")  # the _truncate path
    for s in model_inputs:
        rec = {"sentence": s}
        try:
            sents = [cc.convert(s)]
            texts, mq, rq, sids, partial = quiet(CONV._prepare_data, sents)
        except Exception as e:  # noqa: BLE001
            rec["error"] = err_name(e)
            model_records.append(rec)
            continue
        rec.update({"texts": dedup_rows(texts), "model_query_ids": mq, "result_query_ids": rq, "sent_ids": sids, "partial_results": partial})
        if texts:
            try:
                mi = quiet(
                    prepare_onnx_input,
                    tokenizer=CONV.tokenizer,
                    labels=CONV.labels,
                    char2phonemes=CONV.char2phonemes,
                    chars=CONV.chars,
                    texts=texts,
                    query_ids=mq,
                    use_mask=CONV.config.use_mask,
                    window_size=None,
                    char2id=CONV.char2id,
                    char_phoneme_masks=CONV.char_phoneme_masks,
                )
            except Exception as e:  # noqa: BLE001
                rec["error"] = err_name(e)
                model_records.append(rec)
                continue
            if not mi:
                rec["onnx_input"] = None
            else:
                lengths = [int(row.sum()) for row in mi["attention_masks"]]
                assert not mi["token_type_ids"].any()
                assert all(list(row) == [1] * n + [0] * (len(row) - n) for row, n in zip(mi["attention_masks"].tolist(), lengths))
                rec["onnx_input"] = {
                    # token_type_ids are all 0 and attention_masks are 1 for the first
                    # attention_lengths[i] columns (asserted above), so only the lengths are stored
                    "shape": list(mi["input_ids"].shape),
                    "input_ids": dedup_rows([row[:n] for row, n in zip(mi["input_ids"].tolist(), lengths)]),
                    "attention_lengths": lengths,
                    "phoneme_mask_nonzero": [[int(j) for j in row.nonzero()[0]] for row in mi["phoneme_masks"]],
                    "char_ids": mi["char_ids"].tolist(),
                    "position_ids": mi["position_ids"].tolist(),
                }
                probs = CONV.session_g2pw.run([], {
                    "input_ids": mi["input_ids"],
                    "token_type_ids": mi["token_type_ids"],
                    "attention_mask": mi["attention_masks"],
                    "phoneme_mask": mi["phoneme_masks"],
                    "char_ids": mi["char_ids"],
                    "position_ids": mi["position_ids"],
                })[0]
                preds, confs = CONV._predict(mi)
                rec["preds"] = preds
                rec["confidences"] = confs
                if full_prob_rows < 6:
                    rec["probs"] = [[round(float(x), 7) for x in row] for row in probs[:2]]
                    full_prob_rows += len(rec["probs"])
        model_records.append(rec)
    dump_jsonl(OUT / "model.jsonl", model_records)

    # --- pinyin ---
    pinyin_records = []
    for k, s in enumerate(sentences[:900]):
        rec = {"text": s}
        try:
            with contextlib.redirect_stdout(sys.stderr):
                rec["pinyin_tone3"] = G2PW.pinyin(s, style=Style.TONE3)
                rec["lazy_tone3"] = G2PW.lazy_pinyin(s, style=Style.TONE3)
                if k % 3 == 0:
                    rec["lazy_initials"] = G2PW.lazy_pinyin(s, style=Style.INITIALS)
                    rec["lazy_finals_tone3"] = G2PW.lazy_pinyin(s, style=Style.FINALS_TONE3)
                if k % 10 == 0:
                    rec["pinyin_tone_heteronym"] = G2PW.pinyin(s, style=Style.TONE, heteronym=True)
                    rec["lazy_normal"] = G2PW.lazy_pinyin(s)
        except Exception as e:  # noqa: BLE001
            rec = {"text": s, "error": err_name(e)}
        pinyin_records.append(rec)
    dump_jsonl(OUT / "pinyin.jsonl", pinyin_records)

    # --- chinese2 capture ---
    orig_call = G2PW._g2pw
    orig_cp = chinese2.correct_pronunciation
    captured = {}

    class Recorder:
        def __call__(self, batch):
            out = orig_call(batch)
            captured["calls"].append({"sentences": list(batch), "result": copy.deepcopy(out)})
            return out

    def cp_wrapper(word, word_pinyins):
        before = list(word_pinyins)
        try:
            out = orig_cp(word, word_pinyins)
        except Exception as e:  # noqa: BLE001
            captured["cp"].append({"word": word, "word_pinyins": before, "error": err_name(e)})
            raise
        captured["cp"].append({"word": word, "word_pinyins": before, "out": list(out)})
        return out

    G2PW._g2pw = Recorder()
    chinese2.correct_pronunciation = cp_wrapper
    c2_records = []
    try:
        for text in zh_texts + bulk[:300]:
            captured["calls"], captured["cp"] = [], []
            rec = {"text": text}
            try:
                with contextlib.redirect_stdout(sys.stderr):
                    norm = chinese2.text_normalize(text)
                    rec["norm"] = norm
                    chinese2.g2p(norm)
            except Exception as e:  # noqa: BLE001
                rec["error"] = err_name(e)
            rec["g2pw_calls"] = captured["calls"]
            rec["correct_pronunciation"] = captured["cp"]
            c2_records.append(rec)
    finally:
        G2PW._g2pw = orig_call
        chinese2.correct_pronunciation = orig_cp
    dump_jsonl(OUT / "chinese2.jsonl", c2_records)

    # --- correct_pronunciation ---
    cp_records = []
    cprng = random.Random(SEED + 4)
    keys = sorted(pp)
    single = [k for k in keys if len(k) == 1]
    cases = [("", []), ("湖泊", ["hu2", "bo2"]), ("唉", ["ai1"]), ("银行", ["yin2", "xing2"]), ("唉声", ["ai1", "sheng1"]), ("唉声", ["ai1"]), ("声唉", ["sheng1"]), ("差不多", ["cha1", "bu4", "duo1"])]
    for k in cprng.sample(keys, 1500):
        cases.append((k, [cprng.choice(["a1", "b2", "xing2"]) for _ in k]))
    for _ in range(800):
        w = "".join(cprng.choice(single + list(FILLER)) for _ in range(cprng.randint(1, 4)))
        n = len(w) if cprng.random() < 0.95 else max(0, len(w) - 1)
        cases.append((w, [f"p{j}" for j in range(n)]))
    for word, wp in cases:
        try:
            out = orig_cp(word, list(wp))
            cp_records.append({"word": word, "word_pinyins": wp, "out": list(out)})
        except Exception as e:  # noqa: BLE001
            cp_records.append({"word": word, "word_pinyins": wp, "error": err_name(e)})
    dump_jsonl(OUT / "correct_pronunciation.jsonl", cp_records)


if __name__ == "__main__":
    main()

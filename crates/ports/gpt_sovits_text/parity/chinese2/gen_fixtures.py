"""Generate golden fixtures for the `gpt_sovits_text::chinese2` port.

    pixi run --manifest-path crates/ports/gpt_sovits_text/pixi.toml fixtures chinese2

Runs upstream `GPT_SoVITS/text/chinese2.py` (04a44eb) as GPT-SoVITS imports it
(`is_g2pw = True`, the G2PW model from the cache directory). Writes
crates/ports/gpt_sovits_text/tests/fixtures/chinese2/:

- sentences.jsonl: one record per input text `t` with
  `norm` = text_normalize(t),
  `g2p` = g2p(norm) (what cleaner.clean_text computes for "zh"),
  `py` = g2p(norm) with `chinese2.is_g2pw = False` (the pypinyin branch), and,
  for the hand-written inputs and every third bulk input, `raw` = g2p(t) on
  the unnormalized text (Latin, digits, whitespace and characters outside
  the cleaner's output reach the error branches).
- punct.jsonl: replace_punctuation, replace_punctuation_with_en and
  replace_consecutive_punctuation of a text, and normalizer(text).
- units.jsonl: _merge_erhua(initials, finals, word, pos) and
  _get_initials_finals(word) cases.

A g2p result is {"o": [" ".join(phones), "".join(map(str, word2ph))]}
(phones never contain a space; word2ph items are 1 or 2). Any result is
{"e": [exception type, args]} when Python raises; args are str(arg) for each
item of e.args, with a single tuple argument (the `assert pinyin in
pinyin_to_symbol_map` message) flattened. When text_normalize raises, `g2p`
and `py` are null. stdout (the pypinyin branch and ToneSandhi print) is
discarded.
"""

import contextlib
import io
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gpt_sovits_env  # noqa: E402

gpt_sovits_env.setup()

with contextlib.redirect_stdout(sys.stderr):
    from pypinyin import Style, lazy_pinyin  # noqa: E402
    from pypinyin.constants import PHRASES_DICT  # noqa: E402

    from text import chinese2  # noqa: E402
    from text.g2pw import g2pw as g2pw_module  # noqa: E402
    from text.zh_normalization import char_convert  # noqa: E402

OUT = gpt_sovits_env.FIXTURES / "chinese2"
CORPUS = gpt_sovits_env.CORPUS_DIR / "gpt_sovits_zh_en_yue.json"
JIEBA_DICT = Path(sys.modules["jieba_fast"].__file__).parent / "dict.txt"
SEED = 20261102
N_BULK = 5200

assert chinese2.is_g2pw is True


def err(e):
    args = list(e.args)
    if len(args) == 1 and isinstance(args[0], tuple):
        args = list(args[0])
    return {"e": [type(e).__name__, [str(a) for a in args]]}


def run(fn, *args):
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return {"o": fn(*args)}
    except Exception as e:  # noqa: BLE001 - every exception is part of the contract
        return err(e)


def g2p_compact(text, use_g2pw=True):
    chinese2.is_g2pw = use_g2pw
    try:
        r = run(chinese2.g2p, text)
    finally:
        chinese2.is_g2pw = True
    if "o" in r:
        phones, word2ph = r["o"]
        assert all(" " not in p and p for p in phones), phones
        assert all(w in (1, 2) for w in word2ph), word2ph
        r = {"o": [" ".join(phones), "".join(map(str, word2ph))]}
    return r


HAND = [
    "",
    " ",
    "\n",
    "\u3000",
    "你好",
    "你好，很高兴认识你。",
    "啊——但是《原神》是由,米哈\\游自主，研发的一款全.新开放世界.冒险游戏",
    "呣呣呣～就是…大人的鼹鼠党吧？",
    "嗯，我知道了。",
    "嗯嗯嗯！",
    "冒号：分号；逗号，句号。叹号！问号？",
    "换行\n第二行\n\n第三行",
    "间隔号·顿号、省略号...美元$斜杠/破折号—波浪号~全角波浪～",
    "等等......再说",
    "等等.....再说",
    "等等....再说",
    "等等..再说",
    "……",
    "她说：“你好！”‘再见’",
    "（括号）(half)《书名》【标签】[方括号]「引號」『雙引號』",
    "!!!???...,,,---",
    "!?…,.-",
    "a,b.c!d?e-f",
    "，，，。。。",
    "你好，  世界！  再见。",
    "你好,\t世界!\n再见.",
    "你好，\u3000世界",
    "  前导空格",
    "结尾空格  ",
    "Hello world",
    "我用iPhone 15 Pro Max",
    "GPT-SoVITS是一个TTS模型",
    "ABC啊ABC",
    "A,B,C",
    "x",
    "2024年3月15日",
    "今天是2024-03-15",
    "现在是下午3:45",
    "12:30:45",
    "价格是$12.5",
    "百分之五十，即50%",
    "温度是-5°C",
    "3/4杯水",
    "1.5kg",
    "100km/h",
    "10~20个人",
    "电话：13812345678",
    "固话0755-12345678",
    "第1名和第22名",
    "1+1=2",
    "2的3次方是2^3",
    "版本v1.2.3",
    "٣个",
    "１２３４５",
    "ＡＢＣ，ＤＥＦ",
    "1,000,000元",
    "0.001",
    "-3.14",
    "我有3个苹果",
    "123",
    "12345678901234567890",
    "一",
    "一一",
    "一二三四五",
    "第一",
    "一个",
    "一样",
    "一天",
    "一起",
    "看一看",
    "试一试",
    "一会儿",
    "统一",
    "不",
    "不对",
    "不是",
    "不要",
    "说不说",
    "去不去",
    "好不好",
    "不得不",
    "对不起",
    "我很好",
    "展览馆",
    "小老虎",
    "你给我买五把雨伞",
    "我也想买好几把雨伞",
    "老李想买五百斤米",
    "勇敢",
    "水果",
    "小院儿",
    "胡同儿",
    "范儿",
    "老汉儿",
    "撒欢儿",
    "寻老礼儿",
    "妥妥儿",
    "媳妇儿",
    "女儿",
    "婴儿",
    "花儿",
    "鸟儿",
    "少儿",
    "儿",
    "儿子",
    "一点儿",
    "一会儿",
    "哪儿",
    "这儿",
    "那儿",
    "玩儿",
    "好玩儿",
    "小孩儿",
    "面条儿",
    "聊天儿",
    "大伙儿",
    "老头儿",
    "事儿",
    "味儿",
    "画儿",
    "小猫儿",
    "金鱼儿",
    "儿儿儿",
    "王小二儿",
    "张三李四",
    "银行行长",
    "长大后他成了行长",
    "重庆的重量很重",
    "音乐让我快乐",
    "觉得睡觉很舒服",
    "还钱还是还书",
    "地上的地",
    "传记和传说",
    "处理处所",
    "为了为人",
    "好人好事爱好",
    "曾经的曾先生",
    "单于和单位",
    "着急着凉看着",
    "参加人参",
    "朝阳朝代",
    "调查调皮",
    "西藏宝藏",
    "得到得了",
    "发现头发",
    "便宜方便",
    "模样模型",
    "空气空闲",
    "和平暖和和面",
    "了解好了",
    "首都都是",
    "看守看见",
    "教书教育",
    "应该答应",
    "假期假如",
    "种子种地",
    "数量量体",
    "将来将军",
    "血液流血",
    "数学数一数",
    "恶心可恶",
    "几乎几个",
    "差不多出差",
    "关系系鞋带",
    "相信相貌",
    "缝补缝隙",
    "困难难民",
    "节省省份",
    "薄饼单薄",
    "卡车关卡",
    "给予给你",
    "露水露面",
    "尽管尽力",
    "背包背后",
    "弹琴子弹",
    "降落投降",
    "奔跑投奔",
    "散步散文",
    "干净干活",
    "划船计划",
    "扎针挣扎",
    "冲锋冲床",
    "当时上当",
    "中国中奖",
    "正月正常",
    "称呼相称",
    "结果结实",
    "丧失丧事",
    "盛饭旺盛",
    "宿舍星宿",
    "兴趣兴奋",
    "转动转身",
    "效率率领",
    "温度度量",
    "记载载重",
    "会计会议",
    "少年少数",
    "要求需要",
    "强大勉强",
    "切菜一切",
    "折断折腾",
    "更加更改",
    "否则否极泰来",
    "观看道观",
    "宁可安宁",
    "累计劳累",
    "的确目的",
    "我們今天去臺灣旅遊",
    "這個問題很複雜，需要時間來解決。",
    "頭髮乾燥",
    "後來發現",
    "龍飛鳳舞",
    "爲什麼",
    "丽",
    "〇",
    "々",
    "𠮶",
    "龘",
    "鿏",
    "㐀",
    "😀笑",
    "한국어",
    "かな漢字",
    "İ",
    "ß",
    "é",
    "Ⅷ",
    "①",
    "½",
    "\u200b",
    "a\u0300",
    "⿰木木",
    "_",
    "下划线_测试",
    "~",
    "～～～",
    "——",
    "\u0000",
    "测试\x7f",
    "￥100",
    "100^2",
    "@用户",
    "#话题#",
    "电子邮件abc@example.com",
    "网址https://example.com",
    # Interjections whose readings have no final (pypinyin branch errors).
    "噷", "噷噷", "哼", "欸", "嘸", "呒", "姆", "唔", "噷?", "噷儿", "呒儿", "嗯儿", "欸儿", "哦儿", "一儿", "噷,", ",噷", "儿噷",
    # One Hangul syllable next to a polyphonic character: the tokenizer `{}`
    # path of G2PW leaves None in the result (TypeError in `pinyin[0]`).
    "한暝", "暝한", "한行", "가服了", "한异的", "한曾经",
]

WORD_PUNCT = ["，", "。", "！", "？", "、", "；", "：", ",", ".", "!", "?", "…", "……", "——", "-", "~", "～", "·", "$", "/", "...", "“", "”", "‘", "’", "《", "》", "（", "）", "(", ")", "「", "」", "【", "】", "\n", " ", "\u3000"]
TERMINAL = ["。", "！", "？", "…", "，", ".", "!", "?", ""]
LATIN = ["AI", "GPT", "ok", "OK", "iPhone", "abc", "SoVITS", "x", "Q3", "CPU", "Hello", "TTS", "Python", "B站", "app"]
NUMBERS = [
    "3", "12", "100", "2024", "3.14", "-5", "50%", "1/3", "3/4", "12:30", "8:05:09", "2024年3月15日",
    "2023-12-31", "2023/01/02", "13812345678", "0755-12345678", "400-800-1234", "1,000", "1,234,567",
    "3.5kg", "100km", "20℃", "-5°C", "36.5度", "$12.5", "￥99", "10~20", "5-10", "第3", "1+1=2",
    "3×4", "2^10", "0.5", "007", "１２３", "Ⅲ", "②", "½", "3:2", "99.9%", "15cm", "2.5L", "1e5",
]
UNITS_AFTER_NUM = ["个", "年", "月", "日", "号", "点", "分", "秒", "元", "块", "米", "公里", "斤", "岁", "次", "倍", "页", "层", "楼", "号线"]


def load_jieba_words():
    words = []
    with open(JIEBA_DICT, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if parts and all("一" <= ch <= "龥" for ch in parts[0]):
                words.append((parts[0], int(parts[1]), parts[2] if len(parts) > 2 else ""))
    return words


def main():
    rng = random.Random(SEED)
    OUT.mkdir(parents=True, exist_ok=True)

    jieba_words = load_jieba_words()
    common = [w for w, f, _ in jieba_words if f >= 200 and len(w) <= 4]
    erhua_words = [w for w, f, _ in jieba_words if w.endswith("儿") and len(w) >= 2]
    yibu_words = [w for w, f, _ in jieba_words if ("一" in w or "不" in w) and f >= 50 and len(w) <= 4]
    phrases = [p for p in PHRASES_DICT if all("一" <= ch <= "龥" for ch in p)]
    tone3_words = []
    for p in phrases:
        py = PHRASES_DICT[p]
        if len(py) >= 2 and all(len(s) == 1 for s in py):
            t3 = lazy_pinyin(p, style=Style.TONE3, neutral_tone_with_five=True)
            if all(x.endswith("3") for x in t3):
                tone3_words.append(p)
    pp_words = [w for w in g2pw_module.pp_dict if all("一" <= ch <= "龥" for ch in w)]
    poly_chars = "行长乐觉重还地传处为好曾单着参朝调藏得发便模空和了都看教应假种量将血数恶几差系相缝难省薄卡给露尽背弹降奔散干划扎冲当中正称结丧盛宿兴转率度载会少要强切折更否观宁累的"
    print(
        f"common={len(common)} erhua={len(erhua_words)} yibu={len(yibu_words)} tone3={len(tone3_words)} pp={len(pp_words)}",
        file=sys.stderr,
    )

    def sentence(pools, n_words=(2, 8), punct_p=0.15):
        out = []
        for _ in range(rng.randint(*n_words)):
            out.append(rng.choice(rng.choice(pools)))
            if rng.random() < punct_p:
                out.append(rng.choice(WORD_PUNCT))
        out.append(rng.choice(TERMINAL))
        return "".join(out)

    def numbers_sentence():
        out = []
        for _ in range(rng.randint(1, 4)):
            out.append(rng.choice(common))
            n = rng.choice(NUMBERS)
            out.append(n)
            if rng.random() < 0.5:
                out.append(rng.choice(UNITS_AFTER_NUM))
        out.append(rng.choice(TERMINAL))
        return "".join(out)

    def punct_sentence():
        out = []
        for _ in range(rng.randint(2, 7)):
            out.append(rng.choice(common))
            for _ in range(rng.randint(1, 3)):
                out.append(rng.choice(WORD_PUNCT))
        return "".join(out)

    def latin_sentence():
        out = []
        for _ in range(rng.randint(2, 6)):
            r = rng.random()
            if r < 0.3:
                out.append(rng.choice(LATIN))
            elif r < 0.45:
                out.append(rng.choice(NUMBERS))
            else:
                out.append(rng.choice(common))
            if rng.random() < 0.2:
                out.append(rng.choice([" ", "，", ",", ".", "-"]))
        out.append(rng.choice(TERMINAL))
        return "".join(out)

    def poly_sentence():
        out = []
        for _ in range(rng.randint(2, 7)):
            r = rng.random()
            if r < 0.5:
                out.append(rng.choice(pp_words))
            elif r < 0.7:
                out.append(rng.choice(poly_chars))
            else:
                out.append(rng.choice(common))
            if rng.random() < 0.12:
                out.append(rng.choice(WORD_PUNCT))
        out.append(rng.choice(TERMINAL))
        return "".join(out)

    def random_han():
        return "".join(chr(rng.randint(0x4E00, 0x9FA5)) for _ in range(rng.randint(1, 12))) + rng.choice(TERMINAL)

    bulk = []
    kinds = [
        ("prose", lambda: sentence([common], (3, 12))),
        ("poly", poly_sentence),
        ("num", numbers_sentence),
        ("punct", punct_sentence),
        ("erhua", lambda: sentence([erhua_words, erhua_words, common], (1, 6))),
        ("yibu", lambda: sentence([yibu_words, yibu_words, common, ["一", "不"]], (1, 7))),
        ("tone3", lambda: sentence([tone3_words, tone3_words, common], (1, 6), 0.08)),
        ("trad", lambda: char_convert.simplified_to_traditional(sentence([common, pp_words], (2, 9)))),
        ("latin", latin_sentence),
        ("han", random_han),
    ]
    weights = [16, 16, 12, 10, 10, 10, 10, 8, 6, 2]
    for _ in range(N_BULK):
        _, fn = rng.choices(kinds, weights)[0]
        bulk.append(fn())

    corpus = [item["text"] for item in json.loads(CORPUS.read_text()) if item["lang"] == "zh"]

    inputs = [(t, True) for t in HAND] + [(t, True) for t in corpus] + [(t, i % 3 == 0) for i, t in enumerate(bulk)]
    seen = set()
    records = []
    for i, (t, with_raw) in enumerate(inputs):
        if t in seen:
            continue
        seen.add(t)
        rec = {"t": t}
        norm = run(chinese2.text_normalize, t)
        rec["norm"] = norm
        if "o" in norm:
            rec["g2p"] = g2p_compact(norm["o"])
            rec["py"] = g2p_compact(norm["o"], use_g2pw=False)
        else:
            rec["g2p"] = None
            rec["py"] = None
        if with_raw:
            rec["raw"] = g2p_compact(t)
        records.append(rec)
        if i % 500 == 0:
            print(f"{i}/{len(inputs)}", file=sys.stderr)

    with open(OUT / "sentences.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")

    # punct.jsonl: the string functions on hand-written inputs, rep_map keys
    # alone and in runs, and a sample of the bulk inputs.
    punct_inputs = list(HAND)
    punct_inputs += [k for k in chinese2.rep_map] + ["嗯", "呣", "嗯呣", "....", ".....", "......", "…...", "~~", "～~"]
    punct_inputs += ["".join(rng.choice(WORD_PUNCT + ["a", "B", "中", "1", "嗯", "呣"]) for _ in range(rng.randint(1, 12))) for _ in range(600)]
    punct_inputs += rng.sample(bulk, 600)
    punct_inputs += [rng.choice(NUMBERS) + rng.choice(UNITS_AFTER_NUM) for _ in range(100)]
    punct_records = []
    seen = set()
    for t in punct_inputs:
        if t in seen:
            continue
        seen.add(t)
        punct_records.append(
            {
                "t": t,
                "rp": run(chinese2.replace_punctuation, t),
                "rpe": run(chinese2.replace_punctuation_with_en, t),
                "rcp": run(chinese2.replace_consecutive_punctuation, t),
                "an2cn": run(chinese2.normalizer, t),
            }
        )
    with open(OUT / "punct.jsonl", "w", encoding="utf-8") as f:
        for r in punct_records:
            f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")

    # units.jsonl: _merge_erhua and _get_initials_finals.
    units = []
    words = list(chinese2.must_erhua) + list(chinese2.not_erhua) + erhua_words[:: max(1, len(erhua_words) // 300)]
    words += [w[1:] for w in chinese2.not_erhua if len(w) > 2] + ["儿", "儿儿", "一儿", "a儿", "……儿"]
    poss = ["n", "a", "j", "nr", "v", "x", "r", "", "eng"]
    for w in sorted(set(words)):
        ini = lazy_pinyin(w, neutral_tone_with_five=True, style=Style.INITIALS)
        fin = lazy_pinyin(w, neutral_tone_with_five=True, style=Style.FINALS_TONE3)
        cases = [(ini, fin)]
        if fin and w.endswith("儿"):
            cases.append((ini, fin[:-1] + ["er1"]))
            cases.append((ini, fin[:-1] + ["er5"]))
            if len(fin) >= 2:
                cases.append((ini, fin[:-2] + [""] + fin[-1:]))
                cases.append((ini[:-1], fin))
            cases.append((ini, fin + ["er2"]))
            cases.append((ini, fin[1:]))
        for j, (ini_c, fin_c) in enumerate(cases):
            for pos in rng.sample(poss, 3 if j == 0 else 1):
                units.append(
                    {
                        "f": "_merge_erhua",
                        "a": [ini_c, fin_c, w, pos],
                        "r": run(chinese2._merge_erhua, list(ini_c), list(fin_c), w, pos),
                    }
                )
    gif_words = HAND[:] + rng.sample(common, 800) + rng.sample(erhua_words, 100) + ["……", "嗯", "呣", "哼", "噷", "欸"]
    for w in gif_words:
        units.append({"f": "_get_initials_finals", "a": [w], "r": run(chinese2._get_initials_finals, w)})
    with open(OUT / "units.jsonl", "w", encoding="utf-8") as f:
        for r in units:
            f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")

    print(f"sentences={len(records)} punct={len(punct_records)} units={len(units)}", file=sys.stderr)


if __name__ == "__main__":
    main()

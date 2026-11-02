"""Generate golden fixtures for the `gpt_sovits_text::cantonese` port.

    pixi run --manifest-path crates/ports/gpt_sovits_text/pixi.toml fixtures cantonese

Writes crates/ports/gpt_sovits_text/tests/fixtures/cantonese/:

- sentences.jsonl: one record per input text `i` with
  `rp` = replace_punctuation(text),
  `norm` = text_normalize(text),
  `jy` = get_jyutping(norm) and `g2p` = g2p(norm) (what cleaner.clean_text
  computes for language "yue"), and `raw_g2p` = g2p(text) (unnormalized
  text, which reaches the TypeError / ValueError branches).
  Inputs: hand-written edge cases, every item of
  parity/corpus/gpt_sovits_zh_en_yue.json (yue and zh), and bulk
  inputs generated with a fixed seed.
- syllables.jsonl: jyuping_to_initials_finals_tones(i) for every distinct
  reading that ToJyutping gives a single BMP character (as a one-element
  list), hand-written syllables, and seeded random lists.

Each result is {"o": output} or, when Python raises,
{"e": [exception type, str(exception.args[0])]}. When normalization raises,
`jy` and `g2p` are null.
"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gpt_sovits_env  # noqa: E402

gpt_sovits_env.setup()

import ToJyutping  # noqa: E402
from text import cantonese  # noqa: E402
from text.zh_normalization import char_convert  # noqa: E402

REPO = gpt_sovits_env.CRATE_DIR
OUT = gpt_sovits_env.FIXTURES / "cantonese"
CORPUS = gpt_sovits_env.CORPUS_DIR / "gpt_sovits_zh_en_yue.json"


def run(fn, *args):
    try:
        return {"o": fn(*args)}
    except Exception as e:  # noqa: BLE001 - every exception is part of the contract
        return {"e": [type(e).__name__, str(e.args[0]) if e.args else ""]}


HAND = [
    "",
    " ",
    "\n",
    "\n\n",
    "　",
    "佢個鋤頭太短啦。",
    "啊！但是《原神》是由,米哈\\游自主，  [研发]的一款全.新开放世界.冒险游戏",
    "你食咗飯未呀？",
    "我哋今日去邊度？",
    "嗰度有好多嘢食，啲嘢好好味！",
    "佢唔喺度，冇人知佢去咗邊。",
    "嚟啦嚟啦，唔好遲到呀。",
    "咁樣唔得嘅，你要諗清楚先。",
    "我真係好攰……",
    "等陣...先",
    "等陣......先",
    "等陣.....先",
    "“你好”，‘再见’",
    "（括号）(half)《书名》【标签】[方括号]「引號」",
    "破折号——波浪号～tilde~",
    "冒号：分号；顿号、间隔号·美元$",
    "a,b.c!d?e-f",
    "!!!???...,,,---",
    "!?…,.-",
    "_",
    "下划线_测试",
    "Hello world",
    "我用iPhone 15 Pro Max",
    "GPT-SoVITS係一個TTS模型",
    "ABC啊ABC",
    "2024年3月15日",
    "今日係2024-03-15",
    "而家係下晝3:45",
    "12:30:45",
    "價錢係$12.5",
    "百分之五十，即係50%",
    "溫度係-5°C",
    "3/4杯水",
    "1.5kg",
    "100km/h",
    "10~20個人",
    "電話：13812345678",
    "固話0755-12345678",
    "第1名同第22名",
    "1+1=2",
    "2的3次方係2^3",
    "版本v1.2.3",
    "٣个",
    "１２３４５",
    "ＡＢＣ，ＤＥＦ",
    "𠮶個",
    "𡃁仔",
    "冇問題",
    "乜嘢",
    "唔該晒",
    "啱啱好",
    "嘅",
    "咗",
    "嘢",
    "佢",
    "唔",
    "哋",
    "喺",
    "啲",
    "冇",
    "嚟",
    "龘",
    "鿏",
    "〇",
    "々",
    "ㄅㄆㄇ",
    "한국어",
    "日本語のテキスト",
    "émoji 😀 测试",
    "tab\there",
    "多行\n文本\n测试",
    "嗯",
    "呣",
    "噉",
    "啩",
    "囉",
    "喎",
    "嘞",
    "咩",
    "啫",
    "吋",
    "一吋",
    "呎",
    "唔使客氣，下次再嚟。",
    "佢话：“我唔去喇！”",
    "你喺边度？我喺屋企。",
]

# Cantonese colloquial vocabulary (traditional, with some simplified forms).
YUE_WORDS = """
我 你 佢 我哋 你哋 佢哋 呢個 嗰個 呢度 嗰度 邊度 邊個 乜嘢 咩 點解 點樣 幾多 幾時
係 唔係 有 冇 喺 唔喺 嚟 去 返 食 飲 睇 聽 講 諗 知 識 想 要 唔要 鍾意 唔鍾意 得 唔得
咗 緊 過 埋 晒 返嚟 出去 入嚟 上嚟 落去 起身 瞓覺 返工 放工 返學 放學 開心 唔開心 攰 餓 飽
嘅 啲 嘢 噉 咁 好 好好 好似 真係 其實 不過 但係 所以 因為 如果 雖然 即係 而家 頭先 琴日 今日 聽日
飯 茶 咖啡 奶茶 點心 蝦餃 燒賣 叉燒包 菠蘿包 雞蛋仔 車仔麵 雲吞麵 牛腩 魚蛋 腸粉
朋友 屋企 學校 公司 老細 同事 阿媽 阿爸 細路 老豆 老母 師傅 靚仔 靚女 後生
巴士 地鐵 的士 電車 天星小輪 機場 銀行 醫院 餐廳 茶餐廳 街市 超市 商場
啦 喇 呀 啊 嘞 囉 喎 咩 啩 啫 吖 呢 吓 嘛 㗎 咋 嚱 哩 嗱 嘩
唔該 多謝 對唔住 冇問題 唔緊要 無所謂 搞掂 發生咩事 邊個話 係咪 得唔得 好唔好 有冇 使唔使
睇戲 行街 打邊爐 飲茶 食飯 揸車 搭車 落雨 好熱 好凍 天氣 香港 廣州 澳門 深圳 九龍 新界
我哋 你哋 佢哋 嗰啲 呢啲 啲人 冇嘢 有嘢 咩嘢 嚟緊 去咗 食咗 講咗 嘅嘢
""".split()

ZH_WORDS = """
我们 今天 天气 非常 好 学习 工作 生活 中国 北京 上海 广东 香港 经济 发展 技术 人工智能
语音 合成 模型 训练 数据 文本 处理 测试 系统 问题 解决 方法 时间 地方 朋友 家人 电脑 手机
這個 那個 學習 語音 開發 測試 問題 時間 國家 經濟 發展 電話 電視 圖書館 醫生 銀行 機場
""".split()

PUNCT = list("，。！？、；：·…—～~,.!?;:()（）《》【】[]「」“”‘’\"'-$_\n") + ["...", "……", "——"]
LATIN = ["ok", "OK", "Hello", "iPhone", "TTS", "AI", "WiFi", "app", "email", "Uber", "MTR", "KFC", "x", "abc"]
UNITS = ["km", "kg", "cm", "m", "g", "ml", "L", "°C", "%", "元", "蚊", "個", "隻", "次", "分鐘", "小時", "歲", "米", "公斤"]


def numberish(rng):
    k = rng.randrange(12)
    if k == 0:
        return str(rng.randrange(0, 100000))
    if k == 1:
        return f"{rng.randrange(0, 1000)}.{rng.randrange(0, 100)}"
    if k == 2:
        return f"{rng.randrange(1990, 2030)}年{rng.randrange(1, 13)}月{rng.randrange(1, 32)}日"
    if k == 3:
        return f"{rng.randrange(1990, 2030)}-{rng.randrange(1, 13):02d}-{rng.randrange(1, 29):02d}"
    if k == 4:
        return f"{rng.randrange(0, 24)}:{rng.randrange(0, 60):02d}"
    if k == 5:
        return f"{rng.randrange(0, 1000)}{rng.choice(UNITS)}"
    if k == 6:
        return f"{rng.randrange(1, 10)}/{rng.randrange(2, 20)}"
    if k == 7:
        return f"-{rng.randrange(1, 100)}"
    if k == 8:
        return f"{rng.randrange(1, 50)}~{rng.randrange(50, 100)}"
    if k == 9:
        return "1" + "".join(rng.choice("0123456789") for _ in range(10))
    if k == 10:
        return f"{rng.randrange(1, 100)}%"
    return "".join(chr(0xFF10 + int(c)) for c in str(rng.randrange(0, 1000)))


def bulk(rng, n, trad_chars):
    out = []
    for _ in range(n):
        parts = []
        for _ in range(rng.randrange(1, 12)):
            r = rng.random()
            if r < 0.45:
                parts.append(rng.choice(YUE_WORDS))
            elif r < 0.55:
                parts.append(rng.choice(ZH_WORDS))
            elif r < 0.68:
                parts.append(rng.choice(PUNCT))
            elif r < 0.76:
                parts.append(numberish(rng))
            elif r < 0.82:
                parts.append(rng.choice(LATIN))
            elif r < 0.88:
                parts.append(rng.choice(trad_chars))
            elif r < 0.93:
                parts.append(chr(rng.randrange(0x4E00, 0x9FA6)))
            elif r < 0.96:
                parts.append(rng.choice([" ", "  ", "　", "\t"]))
            else:
                base = rng.choice([0x3400, 0x20000, 0x3040, 0xAC00, 0x0400, 0x00C0, 0x9FA6])
                parts.append(chr(base + rng.randrange(64)))
        out.append("".join(parts))
    return out


def g2p_list(text):
    phones, word2ph = cantonese.g2p(text)
    return [phones, word2ph]


def jyuping_list(syllables):
    phones, word2ph = cantonese.jyuping_to_initials_finals_tones(syllables)
    return [phones, word2ph]


def sentence_record(text):
    rec = {"i": text, "rp": cantonese.replace_punctuation(text)}
    norm = run(cantonese.text_normalize, text)
    rec["norm"] = norm
    if "o" in norm:
        rec["jy"] = run(cantonese.get_jyutping, norm["o"])
        rec["g2p"] = run(g2p_list, norm["o"])
    else:
        rec["jy"] = None
        rec["g2p"] = None
    rec["raw_g2p"] = run(g2p_list, text)
    return rec


def write_jsonl(name, records):
    path = OUT / name
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"{path.relative_to(REPO)}: {len(records)} records, {path.stat().st_size} bytes", file=sys.stderr)


HAND_SYLLABLES = [
    "", "_", "!", "?", "…", ",", ".", "-", "..", "!!", " ", "1", "0", "a", "a0", "aa", "aa3", "aai1",
    "aak3", "ng", "ng5", "nga", "nga4", "ngaa4", "ngo5", "m4", "hm4", "hng6", "n", "xyz1", "q", "zz",
    "a٣", "si１", "ci\U0001d7d9", "sp", "spl", "sil", "spn1", "ong", "ung1", "uk1", "jing1 cyun3",
    "liu4 ge3", "Y", "ABC", "嘅", "ge3\n", "ge3 ",
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20241102)
    corpus = [item["text"] for item in json.loads(CORPUS.read_text())]
    trad_chars = sorted(char_convert.traditional_characters)

    inputs = []
    seen = set()
    for t in HAND + corpus + bulk(rng, 3200, trad_chars):
        if t not in seen:
            seen.add(t)
            inputs.append(t)
    write_jsonl("sentences.jsonl", [sentence_record(t) for t in inputs])

    readings = set()
    for cp in range(0x3000, 0x10000):
        for _, v in ToJyutping.get_jyutping_list(chr(cp)):
            if v is not None:
                readings.add(v)
    readings = sorted(readings)
    syllables = [{"i": [s], **run(jyuping_list, [s])} for s in HAND_SYLLABLES + readings]
    pool = readings + HAND_SYLLABLES[1:]
    for _ in range(300):
        lst = [rng.choice(pool) for _ in range(rng.randrange(1, 8))]
        syllables.append({"i": lst, **run(jyuping_list, lst)})
    write_jsonl("syllables.jsonl", syllables)


if __name__ == "__main__":
    main()

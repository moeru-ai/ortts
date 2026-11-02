"""Generate golden fixtures for the jieba_fast port.

Run: pixi run --manifest-path crates/ports/jieba_fast/pixi.toml fixtures

Writes tests/fixtures/<set>.jsonl in the crate, one JSON object per
input sentence:

- "text":   the input
- "lcut":   jieba_fast.lcut(text) (cut_all=False, HMM=True) as word lengths in
            characters; the words partition the text (checked below)
- "search": jieba_fast.lcut_for_search(text) (HMM=True) as a list of words, or
            null when it equals lcut
- "pos":    jieba_fast.posseg.lcut(text) (HMM=True): the word lengths, or null
            when they equal lcut
- "flags":  the posseg flags joined with spaces

Sets (all random inputs use fixed seeds):

- hand:    hand-written edge cases (whitespace, punctuation, ASCII mixes, the
           regex character classes, Han range boundaries, non-BMP characters,
           DAG positions with more than 12 word ends, ...)
- corpus:  every text of parity/gpt_sovits_zh_en_yue.json
- prose:   sentences from the upstream GPT-SoVITS Chinese docs and UI locale
           files (zh_CN, zh_TW, zh_HK, zh_SG), read from the checkout that
           fetch_upstream.py makes (parity/.cache/GPT-SoVITS, or
           $GPT_SOVITS_DIR)
- words:   random dictionary-word sentences with punctuation, simplified and
           converted to traditional with OpenCC
- hmm:     random character strings that leave single characters for the HMM
           (new words), rare characters and characters outside the HMM tables
- mixed:   mixed Chinese, English, digits, units, percentages, symbols
- long:    long sentences (several hundred characters)
"""

import json
import random
import string
from pathlib import Path

import jieba_fast
import jieba_fast.posseg as psg
import opencc

from fetch_upstream import upstream_dir

CRATE = Path(__file__).resolve().parents[1]
OUT = CRATE / "tests/fixtures"
CORPUS = CRATE / "parity/gpt_sovits_zh_en_yue.json"
UPSTREAM = upstream_dir()

jieba_fast.setLogLevel(60)
jieba_fast.initialize()
FREQ = jieba_fast.dt.FREQ

S2T = opencc.OpenCC("s2t")


def encode(text):
  lc = jieba_fast.lcut(text)
  assert "".join(lc) == text, (text, lc)
  search = jieba_fast.lcut_for_search(text)
  pos = psg.lcut(text)
  pw = [p.word for p in pos]
  assert "".join(pw) == text, (text, pw)
  flags = [p.flag for p in pos]
  assert all(f and " " not in f for f in flags), flags
  return {
    "text": text,
    "lcut": [len(w) for w in lc],
    "search": None if search == lc else search,
    "pos": None if pw == lc else [len(w) for w in pw],
    "flags": " ".join(flags),
  }


def dump(name, texts):
  seen = set()
  rows = []
  for t in texts:
    if t in seen:
      continue
    seen.add(t)
    rows.append(encode(t))
  OUT.mkdir(parents=True, exist_ok=True)
  path = OUT / f"{name}.jsonl"
  with open(path, "w", encoding="utf-8", newline="\n") as f:
    for row in rows:
      f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
  print(f"{name}.jsonl: {len(rows)} rows, {path.stat().st_size} bytes")
  return len(rows)


# ------------------------------------------------------------------ helpers

dict_words = []
dict_freqs = []
with open(Path(jieba_fast.__file__).parent / "dict.txt", encoding="utf-8") as f:
  for line in f:
    w, fr, _ = line.strip().split(" ")
    dict_words.append(w)
    dict_freqs.append(int(fr))

han_words = [w for w in dict_words if all("\u4e00" <= c <= "\u9fd5" for c in w)]
common_chars = sorted({c for w in han_words[:] for c in w if FREQ.get(c)})
finalseg_chars = sorted(jieba_fast.finalseg.emit_P["B"].keys() | jieba_fast.finalseg.emit_P["S"].keys())
all_han = [chr(c) for c in range(0x4E00, 0x9FD6)]
no_emit = [c for c in all_han if c not in jieba_fast.finalseg.emit_P["S"] and c not in psg.char_state_tab_P]
single_only = [c for c in common_chars if len(c) == 1]
PUNCT = list("，。！？、；：「」『』（）《》“”‘’—…·,.!?;:()[]-~ ")
ASCII_WORDS = [
  "AI", "GPT", "SoVITS", "iPhone", "USB", "Wi-Fi", "Python", "C++", "C#", "Rust", "OK", "NBA", "CEO",
  "hello", "world", "test", "model", "v2", "v4", "ProPlus", "TTS", "API", "GPU", "CPU", "x86_64",
  "AT&T", "R&D", "3D", "4K", "5G", "MP3", "UTF-8", "e.g.", "Mr.", "Dr.", "No.1", "_init_", "a+b",
]
UNITS = ["kg", "km", "km/h", "cm", "mm", "m²", "℃", "°C", "%", "‰", "元", "块", "万", "亿", "个", "米", "公斤",
         "小时", "分钟", "秒", "GB", "MB", "Hz", "dB", "mAh", "W", "V", "岁", "年", "月", "日", "号", "次", "倍"]


def rand_dict_sentence(rng, lo=3, hi=15):
  n = rng.randint(lo, hi)
  ws = rng.choices(dict_words, weights=dict_freqs, k=n)
  out = []
  for w in ws:
    out.append(w)
    if rng.random() < 0.12:
      out.append(rng.choice(PUNCT))
  if rng.random() < 0.8:
    out.append(rng.choice("。！？…"))
  return "".join(out)


def rand_rare_word_sentence(rng, lo=3, hi=12):
  # uniform over dictionary entries, so rare and long words show up
  return "".join(rng.choice(dict_words) for _ in range(rng.randint(lo, hi)))


def number(rng):
  k = rng.random()
  if k < 0.3:
    return str(rng.randint(0, 10 ** rng.randint(1, 9)))
  if k < 0.55:
    return f"{rng.randint(0, 9999)}.{rng.randint(0, 999)}"
  if k < 0.7:
    return f"{rng.randint(0, 100)}%"
  if k < 0.8:
    return f"{rng.randint(1, 2030)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
  if k < 0.9:
    return f"{rng.randint(0, 23)}:{rng.randint(0, 59):02d}"
  return "".join(rng.choice("０１２３４５６７８９") for _ in range(rng.randint(1, 4)))


# --------------------------------------------------------------------- hand

def dag_over_12_words():
  """Dictionary words with more than 12 word-prefixes (the C DAG cap)."""
  found = []
  for w in han_words:
    if len(w) < 13:
      continue
    ends = [i for i in range(len(w)) if FREQ.get(w[: i + 1])]
    if len(ends) > 12:
      found.append(w)
  return found


def hand_inputs():
  hand = [
    "", " ", "  ", "\t", "\n", "\r\n", "\r", "\r\r\n\n", "\u3000", "\xa0", "\x1c\x1d\x1e\x1f", "\x85",
    "\u2028\u2029", "\u200a\u202f\u205f", "\u200b", "a", "A", "1", ".", "%", "+", "#", "&", "_",
    "我", "的", "了", "一", "\u4e00", "\u9fd5", "\u9fd6", "\u4dff", "\u3400", "\U00020000", "\U0002a6d6",
    "𠮷野家", "𪚥𪚥", "中\u9fd6国", "中\u9fd5国", "我爱北京天安门", "他来到了网易杭研大厦",
    "小明硕士毕业于中国科学院计算所，后在日本京都大学深造", "我来到北京清华大学",
    "工信处女干事每月经过下属科室都要亲口交代24口交换机等技术性器件的安装工作",
    "结婚的和尚未结婚的", "南京市长江大桥", "永和服装饰品有限公司", "乒乓球拍卖完了",
    "研究生命的起源", "这个门把手坏了", "他说的确实在理", "长春市长春节讲话", "人要是行，干一行行一行",
    "下雨天留客天留我不留", "我们中出了一个叛徒", "李小福是创新办主任也是云计算方面的专家",
    "今天天气不错", "我是拖拉机学院手扶拖拉机专业的", "不用多久，我就会升职加薪，当上CEO，走上人生巅峰",
    "江州市长江大桥参加了长江大桥的通车仪式", "张三丰和李四光在王五家吃饭", "欧阳娜娜和上官婉儿",
    "123", "3.14", "3.14.15", "50%", "50.5%", "100%的人", "1.2.3版本", ".5", "5.", "..", "...",
    "a.b", "a_b", "a+b=c", "C++和C#", "AT&T公司", "R&D部门", "x86_64架构", "#话题#", "100kg", "20km/h",
    "第1名", "第一名", "1st", "2nd", "3rd", "No.1", "iPhone15ProMax", "GPT-SoVITS", "v2Pro", "v2ProPlus",
    "Hello World", "hello，世界", "你好world", "你好 world", "你好\tworld", "你好\r\nworld", "你好\nworld",
    "abc123def", "123abc", "ABC", "aBc", "a1", "1a", "Ａ", "ａｂｃ", "１２３", "１２３ＡＢＣ", "全角，标点。",
    "!!!", "？？？", "……", "——", "～", "《红楼梦》", "「你好」", "『世界』", "(括号)", "[方括号]", "{花括号}",
    "😀", "我😀你", "👨\u200d👩\u200d👧", "こんにちは", "日本語のテキスト", "한국어", "Привет", "ελληνικά", "עברית",
    "مرحبا", "ไทย", "café", "naïve", "ß", "İ", "ﬁ", "Ⅻ", "①②③", "㈠㈡", "〇", "〇一二", "零一二三",
    "一二三四五六七八九十", "百千万亿", "壹贰叁肆伍陆柒捌玖拾", "廿卅", "这是一个句子。这是另一个句子！",
    "句子一\n句子二\r\n句子三", "  前后有空格  ", "中 文 空 格", "中\u3000文", "中\xa0文",
    "繁體中文測試", "臺灣", "香港特別行政區", "這個問題很難解決", "我們一起去學校吧", "電腦軟體",
    "龘", "靐", "齉", "爩", "鱻", "麤", "龖", "厵", "灪", "籱", "饢", "驫", "鸜", "麷", "鼺",
    "龘靐齉爩", "鱻麤龖厵灪", "我龘你", "龘龘龘", "嘤嘤嘤", "哈哈哈哈哈哈", "啊啊啊啊啊啊啊啊啊啊",
    "的的的的的的", "了了了了", "是是是", "一一一一", "不不不不", "我我我我我我我我我我我我我我我我",
    "噫吁嚱危乎高哉", "鱼鱼雅雅", "饕餮", "魑魅魍魉", "耄耋", "囹圄", "犄角旮旯", "圐圙", "嘚瑟", "嘚啵",
    "王小明", "张伟", "诸葛亮", "司马懿", "阿里巴巴", "腾讯", "字节跳动", "华为技术有限公司",
    "中华人民共和国", "中华人民共和国国务院", "中华人民共和国中央人民政府", "联合国教科文组织",
    "上海浦东新区", "北京市海淀区中关村", "广东省深圳市南山区", "重庆", "重新", "银行", "行走",
    # HMM must join runs of characters that are not in the dictionary
    "仳涏", "忐睧儿", "他仳涏了", "忐睧儿的事",
  ]
  hand += [w for w in dag_over_12_words()]
  hand += ["前" + w + "后" for w in dag_over_12_words()[:20]]
  # single regex-class characters next to Han text
  for c in "+#&._%" + string.digits + "aZ":
    hand += [f"中{c}文", f"{c}中", f"中{c}", f"{c}{c}中{c}{c}"]
  # every Python whitespace character between Han characters
  for c in [chr(c) for c in range(0x110000) if chr(c).isspace()]:
    hand += [f"中{c}文", f"a{c}b"]
  return hand


# ------------------------------------------------------------------- corpus

def corpus_inputs():
  return [row["text"] for row in json.loads(CORPUS.read_text(encoding="utf-8"))]


# -------------------------------------------------------------------- prose

def prose_inputs():
  if not (UPSTREAM / "docs/cn").is_dir():
    raise SystemExit(f"{UPSTREAM} has no docs/cn; run the 'upstream' pixi task first")
  texts = []
  for md in sorted((UPSTREAM / "docs/cn").glob("*.md")):
    for line in md.read_text(encoding="utf-8").splitlines():
      line = line.strip()
      if line:
        texts.append(line)
  for loc in ["zh_CN", "zh_TW", "zh_HK", "zh_SG"]:
    data = json.loads((UPSTREAM / f"tools/i18n/locale/{loc}.json").read_text(encoding="utf-8"))
    texts.extend(v for v in data.values() if isinstance(v, str))
  # also split long lines into sentences
  out = []
  for t in texts:
    out.append(t)
    buf = ""
    for c in t:
      buf += c
      if c in "。！？；":
        out.append(buf)
        buf = ""
    if buf and buf != t:
      out.append(buf)
  return out


# -------------------------------------------------------------------- words

def words_inputs():
  rng = random.Random(53001)
  out = []
  for _ in range(3000):
    out.append(rand_dict_sentence(rng))
  for _ in range(1200):
    out.append(rand_rare_word_sentence(rng))
  trad_src = [rand_dict_sentence(rng) for _ in range(1200)]
  out += [S2T.convert(s) for s in trad_src]
  return out


# ---------------------------------------------------------------------- hmm

def hmm_inputs():
  rng = random.Random(53002)
  out = []
  for _ in range(1500):
    # strings of common single characters, which the DAG leaves as runs
    out.append("".join(rng.choice(single_only) for _ in range(rng.randint(2, 10))))
  for _ in range(800):
    # names and new words in context
    name = "".join(rng.choice(finalseg_chars) for _ in range(rng.randint(2, 4)))
    out.append(rand_dict_sentence(rng, 1, 3) + name + rand_dict_sentence(rng, 1, 3))
  for _ in range(600):
    # uniform Han characters, including ones outside the HMM tables
    out.append("".join(rng.choice(all_han) for _ in range(rng.randint(1, 8))))
  for _ in range(400):
    s = list(rand_dict_sentence(rng, 2, 6))
    for _ in range(rng.randint(1, 3)):
      s.insert(rng.randint(0, len(s)), rng.choice(no_emit))
    out.append("".join(s))
  for _ in range(300):
    # CJK extensions and compatibility ideographs next to Han text
    ext = rng.choice([(0x3400, 0x4DBF), (0x20000, 0x2A6DF), (0xF900, 0xFAFF), (0x9FD6, 0x9FFF), (0x2E80, 0x2FDF)])
    c = chr(rng.randint(*ext))
    out.append(rand_dict_sentence(rng, 1, 3) + c + rand_dict_sentence(rng, 1, 3))
  for _ in range(400):
    # single characters mixed with ASCII inside one re_han_default block
    parts = []
    for _ in range(rng.randint(2, 6)):
      parts.append(rng.choice([rng.choice(single_only), rng.choice(finalseg_chars), number(rng),
                               rng.choice(ASCII_WORDS), rng.choice("+#&._%")]))
    out.append("".join(parts))
  return out


# -------------------------------------------------------------------- mixed

def mixed_inputs():
  rng = random.Random(53003)
  out = []
  for _ in range(2500):
    parts = []
    for _ in range(rng.randint(2, 8)):
      k = rng.random()
      if k < 0.35:
        parts.append(rand_dict_sentence(rng, 1, 4))
      elif k < 0.5:
        parts.append(number(rng) + rng.choice(UNITS))
      elif k < 0.6:
        parts.append(number(rng))
      elif k < 0.75:
        parts.append(rng.choice(ASCII_WORDS))
      elif k < 0.85:
        parts.append(rng.choice(PUNCT + ["\n", "\r\n", "\t", "\u3000", "  "]))
      elif k < 0.92:
        parts.append("".join(rng.choice(string.ascii_letters + string.digits + "+#&._%-/@") for _ in range(rng.randint(1, 8))))
      else:
        parts.append(rng.choice(["😀", "♥", "★", "→", "©", "®", "™", "§", "¥", "$", "€", "£", "@", "^", "*", "=", "|", "\\", "/", "<", ">", "'", '"', "`"]))
    out.append("".join(parts))
  return out


# --------------------------------------------------------------------- long

def long_inputs():
  rng = random.Random(53004)
  out = []
  for _ in range(120):
    out.append("".join(rand_dict_sentence(rng, 8, 20) for _ in range(rng.randint(5, 25))))
  for _ in range(30):
    out.append("".join(rng.choice(single_only) for _ in range(rng.randint(100, 400))))
  for _ in range(30):
    out.append(S2T.convert("".join(rand_dict_sentence(rng, 8, 20) for _ in range(10))))
  return out


def main():
  total = 0
  total += dump("hand", hand_inputs())
  total += dump("corpus", corpus_inputs())
  total += dump("prose", prose_inputs())
  total += dump("words", words_inputs())
  total += dump("hmm", hmm_inputs())
  total += dump("mixed", mixed_inputs())
  total += dump("long", long_inputs())
  print("total", total)


if __name__ == "__main__":
  main()

"""Generate golden fixtures for the `gpt_sovits_text::tone_sandhi` port.

    pixi run --manifest-path crates/ports/gpt_sovits_text/pixi.toml fixtures tone_sandhi

The inputs are realistic: each text goes through the `chinese2.py` steps that
run before `ToneSandhi` (`text_normalize`, the sentence split of `g2p`, the
`[a-zA-Z]+` removal of `_g2p` and `jieba_fast.posseg.lcut`). The fixtures
record the inputs that `ToneSandhi` receives, so the Rust test does not need
jieba parity for `pre_merge_for_modify` and only needs it for
`_split_word` (see split_word.jsonl).

Writes crates/ports/gpt_sovits_text/tests/fixtures/tone_sandhi/:

- pre_merge.jsonl: {"i": seg, "o": pre_merge_for_modify(seg), "f": [names of
  the merge steps that failed and printed "<name> failed"]}. A seg is written
  flat: [word0, pos0, word1, pos1, ...].
- modified_tone.jsonl: {"w": word, "p": pos, "f": finals, "o": result} (or
  "e": [exception type, message]) for every word of every merged seg, with
  finals from `lazy_pinyin(word, neutral_tone_with_five=True,
  style=Style.FINALS_TONE3)` (`chinese2._get_initials_finals`) and, when they
  differ, also with the finals of the G2PW branch of `chinese2._g2p`
  (`correct_pronunciation` + `to_finals_tone3`). Deduplicated.
- units.jsonl: {"fn": method name, "a": [args], "o"/"e"} for hand-written
  cases of every method, plus every method on a fixed-seed sample of the bulk
  modified_tone inputs.
- split_word.jsonl: {"i": word, "cfs": list(jieba_fast.cut_for_search(word)),
  "o": _split_word(word)} (or "e") for every word that reaches `_split_word`
  in the other fixtures.
"""

import contextlib
import io
import itertools
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gpt_sovits_env  # noqa: E402

gpt_sovits_env.setup()

with contextlib.redirect_stdout(sys.stderr):
    import jieba_fast  # noqa: E402
    import jieba_fast.posseg as psg  # noqa: E402
    from pypinyin import Style, lazy_pinyin  # noqa: E402
    from pypinyin.contrib.tone_convert import to_finals_tone3  # noqa: E402
    from text import chinese2  # noqa: E402
    from text.g2pw import correct_pronunciation  # noqa: E402
    from text.symbols import punctuation  # noqa: E402
    from text.tone_sandhi import ToneSandhi  # noqa: E402

OUT = gpt_sovits_env.FIXTURES / "tone_sandhi"
CORPUS = gpt_sovits_env.CORPUS_DIR / "gpt_sovits_zh_en_yue.json"
JIEBA_DICT = Path(jieba_fast.__file__).parent / "dict.txt"

TS = ToneSandhi()
SEED = 20251102
N_BULK = 5200
N_CHAR = 200
N_UNIT_SAMPLE = 400

# Hand-written texts. They go through the same preprocessing as the corpus.
HAND_TEXTS = [
    # 不
    "我不怕，你不要走，他看不懂这本书。",
    "不好不坏，不三不四，不知不觉。",
    "你去不去？我不。",
    "不",
    "不不不，这不对。",
    "说不定他会来，来不来随便。",
    "我不是不想去，是不能去。",
    # 一
    "一天一夜，一段时间，一样东西。",
    "看一看，听一听，想一想，试一试，走一走。",
    "第一名，第一次，第一天，第十一。",
    "一零零，二一零，一一零，十一，一百一十一。",
    "统一，一，一？一！一。",
    "我一个人一口气吃了一碗饭。",
    "一模一样，一心一意，一五一十，一来二去。",
    "他一说一笑，一边走一边唱。",
    "一下子，一会儿，一点儿。",
    "唯一的办法是看看一看。",
    "号码是一二三四五六七八九零。",
    "第一一，第一",
    # reduplication
    "奶奶和爷爷在家看看电视，试试新衣服。",
    "妈妈爸爸哥哥姐姐弟弟妹妹宝宝。",
    "人人都说哈哈，虎虎生威，整整一天，想想就好。",
    "花花草草，家家户户，熙熙攘攘，莘莘学子，冉冉升起。",
    "高高兴兴，干干净净，清清楚楚，明明白白。",
    "走走走，说说说，看看看看。",
    "谢谢谢谢，星星点点，常常常常。",
    # 儿
    "小孩儿在公园儿玩儿，一会儿就回家。",
    "花儿，鸟儿，女儿，婴儿，儿子，儿童。",
    "这事儿办得好，那个人儿真不错。",
    "儿",
    "老头儿和老伴儿在门口儿聊天儿。",
    # neutral tone particles
    "走了，看着，去过，吃吧，好呢，是啊，行嘛，好哦。",
    "我的书，慢慢地走，跑得快。",
    "我们，你们，他们，孩子，桌子，椅子，男子，女子，电子。",
    "桌上，地下，家里，屋里，天上，树下。",
    "上来，下去，进来，出去，回来，过去，起来，开去。",
    "这个，那个，一个，几个，两个，各个，每个，整个，个人，个。",
    "三个人，五个苹果，100个，半个月，多个国家。",
    "麻烦，骨头，豆腐，衣服，眼睛，耳朵，东西，事情，地方，时候。",
    "什么，怎么，这么，那么，多少，多么，知识，认识，意思，告诉。",
    "老实人很聪明，他的脑袋很灵活，朋友们都喜欢他。",
    "吧呢哈啊呐噻嘛吖嗨哦哒额滴哩哟喽啰耶喔诶",
    "嘻嘻哈哈，娃哈哈，干嘛呢，局地有雨，以下内容。",
    "量子力学，原子弹，分子生物，石子路，瓜子仁。",
    # three tones
    "你好，我也想买五把雨伞。",
    "蒙古包，纸老虎，展览馆，管理组，洗脸水。",
    "所有人，好喜欢，很美好，我很好，请你走。",
    "小老鼠，老古董，处理好，选举法，买保险。",
    "我想你也很好，我们走了以后你给我打电话。",
    "水果，老板，手表，理想，旅馆，选举，展览，母语。",
    "你只有五百五十五米远。",
    "岂有此理，永远美好，鼓舞我们，许许多多。",
    "我很想你，你也很想我吗？",
    "小李有九把好雨伞。",
    "海水很美，冷水洗脸，好几百，等你买米。",
    # punctuation and mixed
    "你好！今天怎么样？还好：一般般；嗯。",
    "他说：“一，二，三！”然后‘一’笑。",
    "a一b，GPT一下，iPhone不好用。",
    "1一2，3不4，一a，不b。",
    "嗯嗯，呣，呒，嘸。",
    "……一……不……",
    "一—二—三～四~五",
    "价格是12.5元，一共3.14万。",
    "2024年1月1日，第1名，第一个。",
    "电话是13800138000。",
    # longer realistic
    "今天天气真不错，我们去公园散步吧。",
    "这是一个用于测试语音合成前端的句子。",
    "他一边吃饭，一边看电视，一点儿也不着急。",
    "我觉得这个问题不是很难，你再想一想就明白了。",
    "春天来了，小草从地下探出头来，花儿也都开了。",
    "老师告诉我们，学习要一步一个脚印，不能急于求成。",
    "她说话的声音很好听，大家都喜欢听她唱歌。",
    "这家饭馆的菜不太好吃，但是价格很便宜。",
    "我们一起去看电影吧，听说那部电影很有意思。",
    "爷爷每天早上都要出去走走，锻炼锻炼身体。",
    "你把那本书拿过来给我看看好吗？",
    "这件衣服的颜色不错，就是有点儿贵。",
    "他们几个人在屋里商量了半天，也没商量出个结果来。",
    "小明考了第一名，爸爸妈妈都很高兴。",
    "外面下着大雨，我们只好待在家里。",
    "请你帮我把窗户关上，外面太吵了。",
    "这个周末我打算去图书馆看书，你要不要一起去？",
    "他的中文说得越来越好了，简直像中国人一样。",
    "我们应该互相帮助，一起克服困难。",
    "冬天到了，北方下起了鹅毛大雪。",
    "孩子们在操场上跑来跑去，玩得很开心。",
    "这道题我想了好久也没想出来，你能不能教教我？",
    "一年之计在于春，一日之计在于晨。",
    "不入虎穴，焉得虎子。",
    "百闻不如一见，百见不如一干。",
    "千里之行，始于足下。",
    "三人行，必有我师焉。",
    "学而不思则罔，思而不学则殆。",
    "早上好，晚上好，下午好，大家好。",
    "对不起，没关系，谢谢你，不客气。",
    "你吃饭了吗？我吃了，你呢？",
    "这儿那儿哪儿，这里那里哪里。",
    "我想买一斤苹果，两斤香蕉，还有三斤橘子。",
    "他是我的老朋友了，我们认识已经十几年了。",
    "这个地方真漂亮，我以后还想再来一次。",
    "请问去火车站怎么走？一直往前走，到了路口往左拐。",
    "我们学校有三千多个学生，一百多个老师。",
    "妈妈做的饭最好吃了，我每次都要吃两碗。",
    "这本小说写得很好，我一口气就看完了。",
    "他每天晚上都学习到十一二点，真是太辛苦了。",
    "医生说我需要好好休息，不能再熬夜了。",
    "我们的生活越来越好，大家的脸上都露出了笑容。",
]


def make_bulk_texts(rng):
    """Random sentences of jieba dictionary words with sandhi triggers."""
    words, weights, tags = [], [], {}
    with open(JIEBA_DICT, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) != 3:
                continue
            w, freq, tag = parts[0], int(parts[1]), parts[2]
            if not re.fullmatch(r"[一-龥]+", w) or len(w) > 4:
                continue
            words.append(w)
            weights.append(freq**0.8)
            tags[w] = tag
    cum = list(itertools.accumulate(weights))
    verbs = [w for w in words if tags[w] == "v" and len(w) == 1]
    nouns = [w for w in words if tags[w] == "n" and len(w) <= 2]
    tone3 = [w for w in words if len(w) <= 2 and all(f[-1:] == "3" for f in lazy_pinyin(w, style=Style.FINALS_TONE3))]
    must = sorted(TS.must_neural_tone_words)
    must_not = sorted(TS.must_not_neural_tone_words)
    particles = list("了着过的地得吧呢啊嘛哦们子儿个上下里来去")
    puncts = list("，。！？、：；…") + ["……", "——", "“", "”", "~", "1", "2024年", "3.5", "100", "%"]
    numbers = ["一", "二", "三", "十一", "一百", "一千零一", "第一", "第二", "两", "半", "几", "多", "每", "各"]

    def piece():
        r = rng.random()
        if r < 0.55:
            return rng.choices(words, cum_weights=cum)[0]
        if r < 0.62:
            return rng.choice(particles)
        if r < 0.67:
            return "不" + rng.choice(words[:20000])
        if r < 0.72:
            return rng.choice(numbers) + rng.choice(["个", "天", "次", "件", "下", "点", "样", "段", ""])
        if r < 0.75:
            v = rng.choice(verbs)
            return rng.choice([v + "一" + v, v + v, v + v + "看"])
        if r < 0.78:
            n = rng.choice(nouns)
            return n + n if len(n) == 1 else n[0] + n[0] + n[1] + n[1]
        if r < 0.86:
            return rng.choice(tone3)
        if r < 0.90:
            return rng.choice(must)
        if r < 0.91:
            return rng.choice(must_not)
        if r < 0.93:
            return rng.choice(nouns) + "儿"
        return rng.choice(puncts)

    out = []
    for _ in range(N_BULK):
        out.append("".join(piece() for _ in range(rng.randint(2, 7))))
    return out, tone3


def make_char_texts(rng):
    """Random runs of single characters, biased towards tone three."""
    chars = [chr(c) for c in range(0x4E00, 0x9FA6)]
    t3 = [c for c in chars if lazy_pinyin(c, style=Style.FINALS_TONE3)[0][-1:] == "3"]
    out = []
    for _ in range(N_CHAR):
        pool = t3 if rng.random() < 0.7 else chars
        out.append("".join(rng.choice(pool) for _ in range(rng.randint(2, 10))))
    return out


SPLIT_PATTERN = r"(?<=[{0}])\s*".format("".join(punctuation))


def preprocess(text):
    """chinese2.text_normalize + the split of g2p + the regex of _g2p."""
    norm = chinese2.text_normalize(text)
    sentences = [i for i in re.split(SPLIT_PATTERN, norm) if i.strip() != ""]
    return [re.sub("[a-zA-Z]+", "", seg) for seg in sentences]


def call(fn, *args):
    try:
        return {"o": fn(*args)}
    except Exception as e:  # noqa: BLE001 - every exception is part of the contract
        return {"e": [type(e).__name__, str(e.args[0]) if e.args else ""]}


def pairs(seg):
    """A seg as a flat list: [word0, pos0, word1, pos1, ...]."""
    return [x for w, p in seg for x in (w, p)]


def unpairs(flat):
    return [(flat[k], flat[k + 1]) for k in range(0, len(flat), 2)]


def pre_merge(seg):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = TS.pre_merge_for_modify(seg)
    failed = [line.removesuffix(" failed") for line in buf.getvalue().splitlines()]
    return {"i": pairs(seg), "o": pairs(out), "f": failed}


SPLIT_WORDS = set()


def note_split(word):
    SPLIT_WORDS.add(word)


def lazy_finals(word):
    return chinese2._get_initials_finals(word)[1]


def g2pw_finals(word, word_pinyins):
    word_pinyins = correct_pronunciation(word, list(word_pinyins))
    out = []
    for pinyin in word_pinyins:
        if pinyin[0].isalpha():
            out.append(to_finals_tone3(pinyin, neutral_tone_with_five=True))
        else:
            out.append(pinyin)
    return out


# Hand-written direct calls of every method.
HAND_SEGS = [
    [],
    [("不", "d")],
    [("不", "d"), ("不", "d")],
    [("不", "d"), ("好", "a")],
    [("我", "r"), ("不", "d"), ("去", "v"), ("不", "d")],
    [("听", "v"), ("一", "m"), ("听", "v")],
    [("听", "v"), ("一", "m"), ("听", "v"), ("一", "m"), ("听", "v")],
    [("一", "m"), ("听", "v"), ("一", "m"), ("听", "v")],
    [("听", "n"), ("一", "m"), ("听", "v")],
    [("听", "v"), ("一", "m"), ("看", "v")],
    [("一", "m")],
    [("一", "m"), ("一", "m"), ("一", "m")],
    [("一", "m"), ("天", "q"), ("一", "m")],
    [("看", "v"), ("看", "v"), ("看", "v")],
    [("看看", "v"), ("看看", "v")],
    [("你", "r"), ("好", "a")],
    [("你好", "l"), ("我", "r"), ("也", "d"), ("好", "a")],
    [("奶奶", "n"), ("你", "r")],
    [("蒙古", "ns"), ("包", "n")],
    [("小", "a"), ("老虎", "n")],
    [("所有", "b"), ("人", "n")],
    [("我", "r"), ("想", "v"), ("你", "r"), ("也", "d"), ("很", "d"), ("好", "a")],
    [("儿", "n")],
    [("#", "x"), ("儿", "n")],
    [("花", "n"), ("儿", "n"), ("儿", "n")],
    [("小孩", "n"), ("儿", "n"), ("玩", "v"), ("儿", "n")],
    [("呒", "x"), ("你", "r")],
    [("你", "r"), ("呒", "x")],
    [("嗯", "e"), ("好", "a")],
    [("", "x"), ("好", "a")],
    [("好", "a"), ("", "x")],
    [("，", "x"), ("我", "r"), ("。", "x")],
    [("……", "x"), ("你", "r")],
    [("不", "d"), ("一", "m"), ("不", "d")],
    [("一", "m"), ("不", "d"), ("一", "m")],
]

HAND_WORDS = [
    # (word, pos, finals); finals=None means lazy_pinyin finals
    ("看不懂", "v", None),
    ("不怕", "d", None),
    ("不好", "d", None),
    ("不不", "d", None),
    ("不", "d", None),
    ("对不起", "l", None),
    ("不要不", "d", None),
    ("一天", "m", None),
    ("一段", "m", None),
    ("一", "m", None),
    ("看一看", "v", None),
    ("第一", "m", None),
    ("第一名", "m", None),
    ("第一一", "m", None),
    ("一零零", "m", None),
    ("二一零", "m", None),
    ("一一", "m", None),
    ("十一", "m", None),
    ("一，", "m", None),
    ("一。", "m", None),
    ("一?", "m", None),
    ("一a", "m", None),
    ("一1", "m", None),
    ("1一", "m", None),
    ("一一一", "m", None),
    ("唯一", "d", None),
    ("一模一样", "i", None),
    ("一二三", "m", None),
    ("奶奶", "n", None),
    ("试试", "v", None),
    ("旺旺", "a", None),
    ("人人", "n", None),
    ("看看", "d", None),
    ("哈哈", "n", None),
    ("哈哈", "e", None),
    ("吃吧", "v", None),
    ("好的", "a", None),
    ("走了", "v", None),
    ("了", "ul", None),
    ("着", "uz", None),
    ("过", "ug", None),
    ("过", "v", None),
    ("我们", "r", None),
    ("孩子", "n", None),
    ("男子", "n", None),
    ("桌子", "v", None),
    ("桌上", "s", None),
    ("地下", "f", None),
    ("家里", "s", None),
    ("家里", "n", None),
    ("上来", "v", None),
    ("下去", "v", None),
    ("过去", "t", None),
    ("这个", "r", None),
    ("一个", "m", None),
    ("三个", "m", None),
    ("100个", "m", None),
    ("几个", "m", None),
    ("个", "q", None),
    ("个人", "n", None),
    ("那个", "r", None),
    ("麻烦", "a", None),
    ("太麻烦", "a", None),
    ("老实人", "n", None),
    ("东西", "n", None),
    ("你好", "l", None),
    ("蒙古包", "n", None),
    ("纸老虎", "n", None),
    ("所有人", "r", None),
    ("好喜欢", "v", None),
    ("展览馆", "n", None),
    ("岂有此理", "i", None),
    ("永远美好", "i", None),
    ("水果", "n", None),
    ("好好", "a", None),
    ("小孩儿", "n", None),
    ("，", "x", None),
    ("……", "x", None),
    ("", "x", None),
    ("嗯", "e", None),
    ("呒", "x", None),
    ("你呒", "x", None),
    # explicit finals (G2PW style, short lists, empty finals)
    ("一", "m", []),
    ("不怕", "d", ["u4"]),
    ("看不懂", "v", ["an4"]),
    ("一天", "m", ["i1"]),
    ("一天", "m", ["i1", ""]),
    ("不怕", "d", ["u4", ""]),
    ("你好", "l", ["i3", ""]),
    ("你好", "l", ["", "ao3"]),
    ("你好", "l", ["i2", ""]),
    ("好", "a", [""]),
    ("哈哈", "", ["a1", "a1"]),
    ("哈哈", "n", ["a1"]),
    ("吃吧", "v", ["i1"]),
    ("吃吧", "v", []),
    ("个", "q", []),
    ("麻烦", "a", ["a2"]),
    ("蒙古包", "n", ["eng3", "u3"]),
    ("所有人", "r", ["uo3", "ou3"]),
    ("岂有此理", "i", ["i3", "ou3", "i3"]),
    ("，", "x", ["，"]),
    ("一，", "m", ["i1", "，"]),
    ("你。", "x", ["i3", "。"]),
    ("你好吗", "l", ["i3", "ao3", "a5", "x5"]),
    ("你好", "l", ["i3", "ao3", "i3"]),
    ("好好", "a", ["ao3", "ao3", "ao3"]),
]

HAND_FINALS = [[], [""], ["a3"], ["a3", ""], ["", "a3"], ["a1", ""], ["i3", "ao3"], ["i3", "ao4"], ["，"], ["3"]]
HAND_REDUP = ["", "a", "好好", "好的", "好好好", "哈哈哈哈", "aa", "，，"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    corpus = [item["text"] for item in json.loads(CORPUS.read_text())]
    bulk, _ = make_bulk_texts(rng)
    char_texts = make_char_texts(rng)
    texts = HAND_TEXTS + corpus + bulk + char_texts
    print(f"{len(texts)} texts", file=sys.stderr, flush=True)

    segments = []
    for text in texts:
        with contextlib.redirect_stdout(sys.stderr):
            segments.extend(preprocess(text))
    print(f"{len(segments)} segments", file=sys.stderr)

    batch_inputs = [seg for seg in segments if seg]
    with contextlib.redirect_stdout(sys.stderr):
        g2pw_results = chinese2.g2pw._g2pw(batch_inputs) if batch_inputs else []
    cursor = 0

    pre_merge_cases = {}
    word_cases = {}

    def add_word(word, pos, finals):
        key = json.dumps([word, pos, finals], ensure_ascii=False)
        if key in word_cases:
            return
        rec = {"w": word, "p": pos, "f": finals}
        rec.update(call(TS.modified_tone, word, pos, list(finals)))
        word_cases[key] = rec
        note_split(word)

    for seg in segments:
        pinyins = []
        if seg:
            pinyins = g2pw_results[cursor]
            cursor += 1
        seg_cut = psg.lcut(seg)
        key = json.dumps(pairs(seg_cut), ensure_ascii=False)
        if key not in pre_merge_cases:
            pre_merge_cases[key] = pre_merge(seg_cut)
        merged = unpairs(pre_merge_cases[key]["o"])
        pre = 0
        for word, pos in merged:
            now = pre + len(word)
            if pos == "eng":
                pre = now
                continue
            lf = lazy_finals(word)
            add_word(word, pos, lf)
            try:
                gf = g2pw_finals(word, pinyins[pre:now])
            except TypeError:
                # G2PW returned None for a character; chinese2._g2p raises
                # here too, so modified_tone never sees these finals.
                print(f"skip G2PW finals of {word!r}", file=sys.stderr)
                gf = lf
            if gf != lf:
                add_word(word, pos, gf)
            pre = now

    for seg in HAND_SEGS:
        key = json.dumps(pairs(seg), ensure_ascii=False)
        if key not in pre_merge_cases:
            pre_merge_cases[key] = pre_merge(seg)

    units = []

    def unit(name, *args):
        rec = {"fn": name, "a": [pairs(a) if name.startswith(("_merge", "pre_merge")) else a for a in args]}
        copied = [list(a) if isinstance(a, list) else a for a in args]
        res = call(getattr(TS, name), *copied)
        if "o" in res and name.startswith(("_merge", "pre_merge")):
            res["o"] = pairs(res["o"])
        rec.update(res)
        units.append(rec)

    def word_units(word, pos, finals):
        if finals is None:
            finals = lazy_finals(word)
        add_word(word, pos, finals)
        unit("_bu_sandhi", word, finals)
        unit("_yi_sandhi", word, finals)
        unit("_neural_sandhi", word, pos, finals)
        unit("_three_sandhi", word, finals)
        unit("_split_word", word)
        unit("_is_reduplication", word)
        unit("_all_tone_three", finals)
        note_split(word)

    for word, pos, finals in HAND_WORDS:
        word_units(word, pos, finals)
    for finals in HAND_FINALS:
        unit("_all_tone_three", finals)
    for word in HAND_REDUP:
        unit("_is_reduplication", word)
    seg_units = list(HAND_SEGS) + [
        unpairs(c["i"]) for c in rng.sample(list(pre_merge_cases.values()), N_UNIT_SAMPLE // 4)
    ]
    for seg in seg_units:
        seg = list(seg)
        for name in (
            "_merge_bu",
            "_merge_yi",
            "_merge_reduplication",
            "_merge_continuous_three_tones",
            "_merge_continuous_three_tones_2",
            "_merge_er",
        ):
            unit(name, seg)
    bulk_words = list(word_cases.values())
    for rec in rng.sample(bulk_words, N_UNIT_SAMPLE):
        word_units(rec["w"], rec["p"], rec["f"])

    split_cases = []
    for word in sorted(SPLIT_WORDS):
        rec = {"i": word, "cfs": list(jieba_fast.cut_for_search(word))}
        rec.update(call(TS._split_word, word))
        split_cases.append(rec)

    def dump(name, rows):
        with open(OUT / name, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        print(f"{name}: {len(rows)}", file=sys.stderr)

    dump("pre_merge.jsonl", list(pre_merge_cases.values()))
    dump("modified_tone.jsonl", list(word_cases.values()))
    dump("units.jsonl", units)
    dump("split_word.jsonl", split_cases)


if __name__ == "__main__":
    main()

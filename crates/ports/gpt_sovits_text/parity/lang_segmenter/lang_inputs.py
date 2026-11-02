"""Shared fixture inputs for the `split_lang` and `gpt_sovits_text::lang_segmenter` ports.

`inputs()` returns a list of `(category, text)` pairs:

- `hand`: hand-written edge cases (empty and whitespace-only strings,
  newlines, digits, punctuation, apostrophes, emoji, kana-only, kanji-only,
  Hangul, compatibility jamo, half/full width forms, names and acronyms that
  fasttext tends to misdetect, mixed scripts).
- `word`: every fragment of the pools below on its own.
- `bulk`: random concatenations of pool fragments and separators
  (`random.Random(SEED)`).
- `corpus`: every `text` in `parity/corpus/*.json`.

A copy of the `split_lang` crate's parity inputs, so that the `split_lang` and
`lang_segmenter` fixture sets share inputs; keep the two copies in sync.
"""

import json
import random
from pathlib import Path

CORPUS_DIR = Path(__file__).resolve().parents[1] / "corpus"
SEED = 20240611
BULK_COUNT = 2300

ZH = [
    "你好", "今天天气真不错", "我们去公园散步吧", "衬衫", "的", "一", "中国", "北京", "上海", "人工智能",
    "语音合成", "这是一个测试", "我喜欢", "吃饭了吗", "谢谢你", "没问题", "请稍等", "非常感谢", "对不起",
    "为什么", "怎么办", "电脑", "手机", "国外移民", "今晚", "日本人", "学生", "老师", "朋友", "工作",
    "春眠不觉晓", "处处闻啼鸟", "长江", "黄河", "我叫小明", "他说", "她们", "价格是", "一共", "大约",
    "中文", "汉字", "也", "吗", "呢", "啊", "了", "就是", "因为", "所以",
]
YUE = [
    "佢哋喺度食緊飯", "你食咗飯未呀", "唔該", "冇問題", "係咪咁樣", "我哋一齊去", "好耐冇見", "點解",
    "邊度", "咁樣唔得", "啱啱", "嘢", "乜嘢", "唔係",
]
ZH_TW = [
    "繁體中文", "臺灣", "這是一個測試", "謝謝你", "語音合成", "電腦", "學習", "開始", "應該", "問題",
]
JA = [
    "こんにちは", "ありがとう", "ありがとうございます", "すみません", "おはよう", "さようなら", "はい",
    "いいえ", "これ", "それ", "です", "ます", "でした", "の", "は", "が", "を", "に", "ね", "よ",
    "カタカナ", "コンピューター", "テスト", "プログラム", "アニメ", "ゲーム", "ラーメン", "スマホ",
    "東京タワー", "日本語の勉強", "私は学生です", "今日はいい天気ですね", "食べたい", "行きます",
    "知ってる", "君の瞳", "星空みたいに", "キラキラ", "まいご", "ねえ", "勉強してるんだ", "お願いします",
    "々", "時々", "人々", "ゝ", "ヽ", "ー", "ヴ", "ゔ", "ゐ", "ヰ", "゛", "゜", "゠", "ァ", "ヺ",
]
KANJI_ONLY = [
    "東京", "日本", "日本語", "勉強", "天文学", "最近", "大阪", "京都", "新幹線", "漢字", "山田太郎",
    "電話", "会社", "社長", "先生", "学校", "図書館", "自動車",
]
KO = [
    "안녕하세요", "감사합니다", "한국어", "사랑해요", "네", "아니요", "서울", "김치", "고마워", "괜찮아요",
    "한국어 텍스트", "저는 학생입니다", "오늘 날씨가 좋네요", "가", "힘내세요", "맛있어요", "ㅋㅋ", "ㅎㅎ",
    "ㅋㅋㅋ", "ㄱㄴㄷ", "ᄀᄁ", "ᅡᅢ",
]
EN = [
    "hello", "Hello", "HELLO", "world", "Hello world", "I", "a", "OK", "ok", "AI", "GPT", "NASA", "USB",
    "CPU", "GPU", "iPhone", "ThinkPad", "MyGO", "GPT-SoVITS", "Tom", "Alice", "Bob", "John Smith",
    "Mr.", "e.g.", "U.S.A.", "don't", "it's", "rock'n'roll", "the", "of", "and", "is", "in", "test",
    "Python", "Rust", "OpenAI", "Google", "Advanced Dock", "PCIe", "T60", "v2", "x86", "USB-C",
    "COVID-19", "Wi-Fi", "Bluetooth", "Taylor Swift", "New York", "Tokyo", "Seoul", "Beijing",
    "good morning", "thank you", "see you", "what", "yes", "no", "wow", "lol", "hmm", "Ha",
    "Xin", "Li", "Wang", "Zhang", "Kim", "Park", "Sato", "Suzuki", "x", "z", "Q", "AB", "abc",
]
OTHER_LATIN = [
    "Bonjour", "Danke", "Hola", "Ciao", "Olá", "Merci beaucoup", "Guten Tag", "naïve", "café",
    "Straße", "Zürich", "Dobar dan", "hvala", "señor", "Ångström",
]
OTHER_SCRIPT = [
    "Привет", "Здравствуйте", "สวัสดี", "مرحبا", "שלום", "नमस्ते", "Γειά σου", "ｱｲｳ", "ﾃｽﾄ", "ＡＢＣ",
    "ｈｅｌｌｏ", "Ⅻ", "①②③", "ⓐ", "™", "©", "°C", "½",
]
DIGIT = [
    "1", "0", "12", "123", "2024", "3.14", "10,000", "50%", "1st", "2nd", "100元", "5点", "3時",
    "１２３", "２０２４", "²", "³", "¹", "٣", "७", "1.5", "007", "9.15", "12:30", "2024-06-11", "+86",
    "#1", "$5", "￥100", "１",
]
PUNCT = list(
    """〜~,.;:!?，。！？；：、·([{<（【《〈「『“‘)]}>）】》〉」』”’"-_——#$%&……￥'*+<=>?@[\\]^_`{|}"""
) + ["…", "...", "!!", "?!", "。。", "、、", "—", "–", "/", "\\", "|", "・", "「」", "『』", "（）", "〜〜"]
EMOJI = ["😀", "👍", "🇯🇵", "❤️", "🎉", "😂😂", "👨‍👩‍👧", "🍣", "✨", "☆", "★", "♪"]
SPACE = [" ", "  ", "\n", "\r\n", "\n\n", "\t", "\u3000", "\r", " \n ", "\u00a0", "\u2003", "\x0b"]

HAND = [
    "", " ", "  ", "\n", "\n\n", "\r\n", "\t", "\u3000", " \n ", "\x1c", "\x1f",
    "a", "A", "I", "x", "1", "0", "123", "3.14", "２０２４", "²", "①", "!", "?", "!!", "。", "，", "...",
    "'", "''", "'a'", "a'b", " 'quoted' ", "don't", "rock'n'roll", "Tom's 猫", "'你好'",
    "你", "あ", "ア", "한", "ㅋ", "々", "ー", "😀", "😀😀😀", "👍 ok", "你好😀", "こんにちは😀",
    "MyGO?,你也喜欢まいご吗？",
    "ねえ、知ってる？最近、僕は天文学を勉強してるんだ。君の瞳が星空みたいにキラキラしてるからさ。",
    "当时ThinkPad T60刚刚发布，一同推出的还有一款名为Advanced Dock的扩展坞配件。",
    "这款扩展坞通过连接T60底部的插槽，扩展出包括PCIe在内的一大堆接口，并且自带电源，让T60可以安装桌面显卡来提升性能。",
    "我喜欢 Python 编程", "hello 你好", "你好 hello", "iPhone 15 Pro 发布了", "GPT-SoVITS 是一个项目",
    "東京 Tokyo 도쿄", "今天是2024年6月11日", "价格是100元", "我有3个苹果和5个梨",
    "第1名", "1个", "100%", "3.14159是圆周率", "电话号码是13800138000", "123abc", "abc123",
    "123你好", "你好123", "123こんにちは", "こんにちは123", "123안녕", "안녕123", "1 2 3",
    "1。", "1.", "你好，123。", "Hello, 123.", "123, hello", "第1章。开始", "abc。123",
    "你好\n世界", "hello\nworld", "hello\r\nworld 你好", "\n\nabc", "abc\n\n", "中文\n日本語\nEnglish",
    "一行\r\n二行", "a\nb\nc", "你好\n\n\nhello", "こんにちは\n안녕하세요\nhello",
    "日本語と中国語", "私は中国人です", "我是日本人", "日本人", "今晚", "国外移民",
    "東京タワーに行きます", "カタカナだけ", "ひらがなだけ", "漢字", "漢字漢字漢字", "東京大阪京都",
    "这是中文，これは日本語です。", "中文中文中文中文中文中文中文中文中文中文中文あ",
    "あ中文中文中文中文中文中文中文中文中文中文中文", "あいうえお中", "中あいうえお",
    "한국어와 English", "안녕하세요, 저는 Tom입니다.", "사랑해 我爱你 愛してる I love you",
    "한국어 텍스트입니다。中文", "ㅋㅋㅋ 웃겨", "ㅋㅋㅋ", "ｱｲｳｴｵ", "ﾃｽﾄです", "ＡＢＣ", "ｈｅｌｌｏ 世界",
    "AI", "GPT", "NASA", "USB", "OK", "OK吗", "OK です", "AI技术", "NASA发射了火箭", "我用USB连接",
    "Tom", "Tom说", "John Smith 先生", "山田さんとTomさん", "Kim 씨", "MyGO", "Ave Mujica",
    "Bonjour 你好", "Danke schön", "Hola amigo", "Привет 世界", "สวัสดี 你好", "مرحبا",
    "佢哋喺度食緊飯", "你食咗飯未呀？", "唔該晒", "繁體中文測試", "這是一個測試。", "臺灣",
    "『括弧』", "「こんにちは」", "（你好）", "【测试】", "《书名》", "“引号”", "‘单引号’",
    "~~~", "〜", "——", "……", "￥", "#hashtag", "@mention", "a+b=c", "x<y>z", "[1]", "{a}",
    "https://example.com", "user@example.com", "C++ / Rust / Go", "50% off!", "Q&A",
    "    leading spaces", "trailing spaces    ", "  both  ", "mid  dle", "\ttab\t",
    "你好 ，世界", "你好， 世界", "你 好", "あ い う", "한 국", "a b c", "A B C D E F",
    "This is a long English sentence used for testing the language splitter behaviour.",
    "This is English. これは日本語です。这是中文。이것은 한국어입니다.",
    "数字1和2和3", "1と2と3", "1과 2와 3", "one 1 two 2", "第一，第二，第三",
    "まいご", "MyGO", "?,", "?,你", "你?,", "x?", "…你好…", "!你好!", "。你好。",
    "中", "中a", "a中", "中1", "1中", "中。", "。中", "中\n", "\n中", "中 ", " 中",
    "2024年", "3時", "5点", "100元", "12:30", "2024-06-11", "+86 138", "$5", "￥100",
    "東京2020", "iPhone15", "Win10", "T60", "x86_64", "COVID-19疫情", "5G网络", "4K视频",
    # _special_merge_for_zh_ja: short head substring merged into a 10x longer right neighbour
    "ねえ这是一个非常非常长的中文句子没有任何标点符号的", "了[10,000。いいえPython",
    "!\nabc", "abc!\ndef", "abc\ndef", "!\n你好", "你好\n!", "1\n2", "1!2", "!1", "1!", "1\n你好",
]


def _bulk(rng):
    pools = [
        (ZH, 6), (YUE, 1), (ZH_TW, 1), (JA, 5), (KANJI_ONLY, 2), (KO, 3), (EN, 6),
        (OTHER_LATIN, 1), (OTHER_SCRIPT, 1), (DIGIT, 3), (PUNCT, 3), (EMOJI, 1), (SPACE, 2),
    ]
    weights = [w for _, w in pools]
    out = []
    seen = set()
    while len(out) < BULK_COUNT:
        n = rng.choice([1, 2, 2, 3, 3, 3, 4, 4, 5, 6, 7, 8])
        parts = []
        for _ in range(n):
            pool = rng.choices(pools, weights=weights)[0][0]
            parts.append(rng.choice(pool))
            r = rng.random()
            if r < 0.15:
                parts.append(" ")
            elif r < 0.25:
                parts.append(rng.choice(PUNCT))
        text = "".join(parts)
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def corpus_texts():
    texts = []
    for path in sorted(CORPUS_DIR.glob("*.json")):
        for item in json.loads(path.read_text()):
            texts.append(item["text"])
    return texts


def inputs():
    rng = random.Random(SEED)
    out = [("hand", t) for t in HAND]
    words = []
    for pool in [ZH, YUE, ZH_TW, JA, KANJI_ONLY, KO, EN, OTHER_LATIN, OTHER_SCRIPT, DIGIT, PUNCT, EMOJI, SPACE]:
        words.extend(pool)
    out += [("word", t) for t in words]
    out += [("bulk", t) for t in _bulk(rng)]
    out += [("corpus", t) for t in corpus_texts()]
    return out

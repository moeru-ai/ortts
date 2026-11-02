"""Shared setup for the cleaner, text_preprocessor and e2e fixture generators.

- Imports the upstream GPT-SoVITS text frontend through `gpt_sovits_env`.
- Points fast-langdetect at `<cache>/fast_langdetect/lid.176.bin`,
  the file upstream loads from `pretrained_models/fast_langdetect`.
- Replaces `clean_text` (in `text.cleaner` and in `TextPreprocessor`) with a
  wrapper that raises `NotPortedError` where upstream would import a module
  that the Rust port does not have (`text.korean` for v2 `ko`, `text.chinese`
  for v1 `zh`). The Rust port returns `cleaner::Error::NotPorted` there, whose
  name is also `NotPortedError`.
- `FakeBert`: a deterministic stand-in for `chinese-roberta-wwm-ext-large`,
  implemented identically in `crates/ports/gpt_sovits_text/tests/common/e2e_common.rs`.
"""

import contextlib
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gpt_sovits_env  # noqa: E402

FIXTURES = gpt_sovits_env.FIXTURES
CORPUS_DIR = gpt_sovits_env.CORPUS_DIR

gpt_sovits_env.setup()

with contextlib.redirect_stdout(sys.stderr):
    import fast_langdetect  # noqa: E402
    import torch  # noqa: E402
    from fast_langdetect.infer import LangDetectConfig, LangDetector  # noqa: E402
    from transformers import AutoTokenizer  # noqa: E402

    import text.cleaner as cleaner  # noqa: E402
    import TTS_infer_pack.TextPreprocessor as TPM  # noqa: E402
    from text import symbols2  # noqa: E402

fast_langdetect.infer._default_detector = LangDetector(
    LangDetectConfig(cache_dir=gpt_sovits_env.LID_MODEL.parent)
)


class NotPortedError(Exception):
    """Raised where upstream imports `text.korean` or `text.chinese`."""


_orig_clean_text = cleaner.clean_text


def clean_text(text, language, version=None):
    v = version if version is not None else os.environ.get("version", "v2")
    if (v == "v1" and language == "zh") or (v != "v1" and language == "ko"):
        raise NotPortedError(f"text.{'chinese' if v == 'v1' else 'korean'}")
    return _orig_clean_text(text, language, version)


cleaner.clean_text = clean_text
TPM.clean_text = clean_text

HIDDEN = 1024
N_HIDDEN_STATES = 25


class FakeBert:
    """`bert_model(**inputs, output_hidden_states=True)` with 25 hidden states.

    hidden_states[-3][0, t, d] = ((id_t * 1009 + t * 31 + d * 17) % 65521) / 64,
    exact in float32. The other hidden states are constants, so a wrong slice
    gives different features.
    """

    def __call__(self, input_ids, token_type_ids=None, attention_mask=None, output_hidden_states=False):
        ids = input_ids[0].to(torch.int64)
        t = torch.arange(ids.shape[0], dtype=torch.int64)
        d = torch.arange(HIDDEN, dtype=torch.int64)
        val = (ids[:, None] * 1009 + t[:, None] * 31 + d[None, :] * 17) % 65521
        h = (val.to(torch.float32) / 64.0)[None]
        states = [torch.full((1, ids.shape[0], HIDDEN), float(k) + 0.5) for k in range(N_HIDDEN_STATES)]
        states[-3] = h
        return {"hidden_states": tuple(states)}


def tokenizer():
    return AutoTokenizer.from_pretrained("GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large")


def text_preprocessor():
    return TPM.TextPreprocessor(FakeBert(), tokenizer(), torch.device("cpu"))


def bert_summary(bert):
    """`[shape, checksum]` of a `[1024, n]` feature.

    `checksum` is `sum(bert[i, j] * 64 * (1 + (i * n + j) % 997))` over all
    entries. Every value of the fake model times 64 is an integer below
    65521, so the sum is exact; a feature taken from the wrong token, layer
    or position, or repeated the wrong number of times, changes it.
    """
    k = (bert * 64).to(torch.int64)
    assert torch.equal(k.to(torch.float32), bert * 64)
    rows, cols = k.shape
    idx = torch.arange(rows * cols, dtype=torch.int64).reshape(rows, cols)
    w = 1 + idx % 997
    return [[rows, cols], int((k * w).sum())]


def dumps(rec):
    return json.dumps(rec, ensure_ascii=False, separators=(",", ":"))


def err(e):
    return {"error": type(e).__name__}


def quiet(f, *args, **kwargs):
    with contextlib.redirect_stdout(sys.stderr):
        return f(*args, **kwargs)


def symbols():
    return symbols2.symbols


# ---------------------------------------------------------------------------
# Inputs

def load_corpus(name):
    return json.loads((CORPUS_DIR / name).read_text())


KO = [
    "안녕하세요.",
    "감사합니다!",
    "오늘 날씨가 좋네요.",
    "저는 학생입니다.",
]

EDGE_PARAGRAPHS = [
    "",
    "\n",
    "\n\n\n",
    " ",
    "。",
    "。。。",
    "……",
    "!?",
    "a",
    "你",
    "你好",
    "Hi.",
    "OK",
    "好",
    "嗯。",
    "123",
    "3.14",
    "价格￥100元。",
    "^_^ 你好^啊。",
    "￥",
    "^",
    "你好！！！？？？……世界。。。",
    "Hello!!! World??? ...",
    "你好  世界   hello    world",
    "第一句。\n\n\n第二句。\n\n第三句。",
    "短。短。短。短。短。短。短。短。",
    "一，二，三，四，五，六，七，八，九，十。",
    "a, b, c, d, e, f, g.",
    "今天是2026年11月2日，温度是-3.5℃。",
    "我的电话是13800138000。",
    "这是一个很长的句子" * 60 + "。",
    "これは長い文です" * 70,
    "This is a long sentence without any break " * 15,
    "你好hello世界こんにちは안녕하세요。",
    "안녕하세요 你好 hello こんにちは",
    "我今天去了Starbucks，点了一杯latte。然后去了公园。",
    "@#$%^&*()",
    "@@@",
    "※",
    "＃＃",
    "你好@@@",
    "———",
    "-- hello --",
    "《三体》是一部科幻小说。",
    "“你好”，他说。",
    "他说：“我们走吧！”",
    "今日は良い天気ですね。散歩に行きましょう。",
    "呢度好靚。我哋去飲茶啦。",
]


def paragraphs(n, seed):
    """`n` random multi-sentence paragraphs and the language modes to run them in."""
    zh_en_yue = load_corpus("gpt_sovits_zh_en_yue.json")
    ja = [x["text"] for x in load_corpus("gpt_sovits_ja.json")]
    zh = [x["text"] for x in zh_en_yue if x["cat"].startswith("zh_") or x["cat"] == "mixed"]
    en = [x["text"] for x in zh_en_yue if x["cat"].startswith("en_")]
    yue = [x["text"] for x in zh_en_yue if x["cat"] == "yue"]
    pools = {"zh": zh, "ja": ja, "en": en, "yue": yue, "ko": KO}
    styles = [
        ("zh", ["zh"], ["zh", "auto"]),
        ("ja", ["ja"], ["ja", "auto"]),
        ("en", ["en"], ["en", "auto"]),
        ("yue", ["yue"], ["yue", "auto_yue"]),
        ("zh+en", ["zh", "en"], ["zh", "auto"]),
        ("ja+en", ["ja", "en"], ["ja", "auto"]),
        ("yue+en", ["yue", "en"], ["yue", "auto_yue"]),
        ("zh+ja", ["zh", "ja"], ["auto", "all_zh"]),
        ("mixed", ["zh", "ja", "en", "yue"], ["auto", "zh"]),
        ("ko", ["ko", "zh", "en"], ["auto", "zh"]),
    ]
    weights = [16, 16, 12, 8, 14, 10, 6, 8, 8, 2]
    seps = ["", "", " ", "\n", "\n\n", "。", "！", "？", "，", "...", "…", "~", " \n"]
    rng = random.Random(seed)
    out = []
    for i in range(n):
        style, langs, modes = rng.choices(styles, weights)[0]
        parts = []
        for _ in range(rng.randint(2, 8)):
            parts.append(rng.choice(pools[rng.choice(langs)]))
            parts.append(rng.choice(seps))
        text = "".join(parts)
        r = rng.random()
        if r < 0.05:
            text = text.replace("，", "，，，")
        elif r < 0.10:
            text = text.replace(" ", "  ")
        elif r < 0.13:
            text = "\n\n" + text + "\n\n"
        elif r < 0.15:
            text = text * 4
        out.append({"cat": f"para_{style}", "text": text, "modes": modes})
    return out

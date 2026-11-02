"""Generate golden fixtures for the `gpt_sovits_text::english` port.

    pixi run --manifest-path crates/ports/gpt_sovits_text/pixi.toml fixtures english

Runs upstream GPT_SoVITS/text/english.py (04a44eb) and writes
crates/ports/gpt_sovits_text/tests/fixtures/english/:

- dict.json: the dictionaries. For `get_dict()`, `en_G2p().cmu` and
  `get_namedict()`: the entry count, an FNV-1a 64 digest over the sorted
  entries (see `digest`), and a seeded sample of entries. Also the result of
  `read_dict()` (it raises) and the `read` / `complex` homograph overrides.
- texts.jsonl: one record per input text `i`:
  `norm` = text_normalize(i), `g2p` = g2p(norm) (what cleaner.clean_text does
  for language "en"), and for the hand-written and single-token inputs also
  `call` = _g2p(norm) (the raw `en_G2p.__call__` output) and `raw` =
  g2p(i) (unnormalized text, which reaches the KeyError branch and the
  non-letter tokens). `g2p` is null when text_normalize raises.
- qryword.jsonl: `_g2p.qryword(i)` for single words.
- funcs.jsonl: `replace_consecutive_punctuation` and `replace_phs` on
  hand-written and seeded inputs.

Each result is {"o": output} or, when Python raises,
{"e": [exception type, str(exception.args[0])]}.
"""

import contextlib
import io
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gpt_sovits_env  # noqa: E402

gpt_sovits_env.setup()

with contextlib.redirect_stdout(sys.stderr):
    from text import english  # noqa: E402
    from text.LangSegmenter import langsegmenter as ls  # noqa: E402

OUT = gpt_sovits_env.FIXTURES / "english"
CORPUS = gpt_sovits_env.CORPUS_DIR / "gpt_sovits_zh_en_yue.json"
SEED = 20241102

G = english._g2p
CMU_WORDS = sorted(k for k in G.cmu if re.fullmatch(r"[a-z]+", k))
CMU_APOS = sorted(k for k in G.cmu if re.fullmatch(r"[a-z]+'[a-z]+", k))
NAMES_ONLY = sorted(k for k in G.namedict if k not in G.cmu)
HOMOGRAPHS = sorted(G.homograph2features)


def run(fn, *args):
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            return {"o": fn(*args)}
    except Exception as e:  # noqa: BLE001 - every exception is part of the contract
        return {"e": [type(e).__name__, str(e.args[0]) if e.args else ""]}


# ---------------------------------------------------------------------------
# dictionaries

FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3


def digest(d):
    """FNV-1a 64 over the UTF-8 of the entries sorted by key. Each entry is
    key, 0x1D, the pronunciations separated by 0x1E (phones separated by
    0x1F), then 0x1C."""
    h = FNV_OFFSET
    for key in sorted(d):
        s = key + "\x1d" + "\x1e".join("\x1f".join(p) for p in d[key]) + "\x1c"
        for b in s.encode("utf-8"):
            h ^= b
            h = (h * FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


def summary(d, r, n):
    keys = sorted(d)
    sample = sorted(r.sample(keys, min(n, len(keys))))
    special = [k for k in keys if not re.fullmatch(r"[a-z]+", k)][:200]
    pick = sorted(set(sample) | set(special) | {k for k in ["a", "z", "read", "chatgpt", "json", "conda", "ae", "os"] if k in d})
    return {"len": len(d), "digest": digest(d), "sample": {k: d[k] for k in pick}}


def gen_dict():
    r = random.Random(SEED)
    with contextlib.redirect_stdout(sys.stderr):
        get_dict = english.get_dict()
        read_new = english.read_dict_new()
        hot = english.hot_reload_hot({})
        namedict = english.get_namedict()
    return {
        "read_dict": run(english.read_dict),
        "read_dict_new": summary(read_new, r, 300),
        "hot_reload_hot_empty": hot,
        "get_dict": summary(get_dict, r, 300),
        "cmu": summary(G.cmu, r, 300),
        "namedict": summary(namedict, r, 300),
        "homograph2features": {k: list(G.homograph2features[k]) for k in ["read", "complex", "live", "record", "object", "refuse", "present", "wind"] if k in G.homograph2features},
        "homograph2features_len": len(G.homograph2features),
    }


# ---------------------------------------------------------------------------
# texts

HAND = [
    "",
    " ",
    "\n",
    "hello",
    "Hello world.",
    "e.g. I used openai's AI tool to draw a picture.",
    "In this; paper, we propose 1 DSPGAN, a GAN-based universal vocoder.",
    # rep_map: the keys are re.escape'd, so only the literal strings match
    "a[;:：，；]b",
    'a["’]b',
    "a;b:c：d，e；f",
    'say "hi" and ’bye’',
    "end。next！really？",
    "[;:：，；][;:：，；]",
    '["’]["’]',
    # consecutive punctuation
    "Wait... what?!",
    "No!!! Yes??? Maybe...",
    "Hmm,,, okay.. fine!?",
    "a - - b",
    "a -- b --- c",
    "…… ok …",
    ". . . .",
    "?!?!?!",
    "Really?! Yes!",
    "one, two, three.",
    "!!!hello",
    "hello!!!",
    "  , , ,  ",
    # sentences
    "The quick brown fox jumps over the lazy dog.",
    "She sells sea shells by the sea shore.",
    "How are you doing today?",
    "I can't believe it's already November!",
    "We'll see what they've done, won't we?",
    "You're right, I'd rather not.",
    "Don't, shouldn't, couldn't, wouldn't, isn't, aren't, wasn't.",
    "It's John's book, not Mary's.",
    "The dogs' bowls and the cat's toy.",
    "The boss's office and James's car.",
    "Mike's, Jack's, Jeff's, Smith's, Ruth's, Bob's, Ted's, Meg's, Dave's, Tom's, Ben's, Bill's, Karl's, Joe's.",
    "Rex's, Liz's, Josh's, Mitch's, George's, Bush's.",
    "openai's, nvidia's, google's, microsoft's, tiktok's, youtube's, zoom's.",
    "xyzzyqwop's plan",
    # homographs
    "I read books every day. Yesterday I read a novel.",
    "I will read it. I have read it. Read it now. Reading is fun.",
    "They live here. The show is live.",
    "Record the record.",
    "I refuse to collect the refuse.",
    "I object to that object.",
    "The complex problem is complex. A complex of buildings.",
    "The wind will wind the clock.",
    "Close the door, it's close.",
    "They lead the way with lead pipes.",
    "Please present the present.",
    "The bass player caught a bass.",
    "I wound the wound.",
    "Does he like does?",
    "Tear up and shed a tear.",
    "Minute details took a minute.",
    "We must polish the Polish furniture.",
    "Desert the desert.",
    "Produce more produce.",
    "Permit me to get a permit.",
    "The content made him content.",
    "Use it; what is the use?",
    "read", "Read", "READ", "live", "Live", "record", "refuse", "object", "complex",
    # single letters
    "A", "a", "I", "i", "B b C c", "A B C D E F G", "a b c d e f g h i j k l m n o p q r s t u v w x y z",
    "Plan A and plan B.", "Grade A", "a A a A",
    # hot dict and deleted abbreviations
    "ChatGPT", "chatgpt", "JSON", "json", "conda", "Conda env with JSON and ChatGPT",
    "AE AI AR IOS HUD OS", "ae ai ar ios hud os", "Ai Ae Ar Ios Hud Os", "iOS macOS AI",
    "The AI on my iOS device.",
    # names (title case uses namedict)
    "Aadi Aadil Aaditya Aakash", "aadi aadil", "AADI AADIL",
    "Hello Aakriti, meet Aaditya.",
    # OOV words and compounds (wordsegment)
    "smartwatch", "keyboardwarrior", "thequickbrownfox", "superduperlongword",
    "notebooklm", "chatbot", "datacenter", "workflowautomation", "unbelievableness",
    "Supercalifragilisticexpialidocious", "antidisestablishmentarianism",
    "pneumonoultramicroscopicsilicovolcanoconiosis",
    "covid", "blockchain", "cryptocurrency", "selfie", "hashtag", "emoji", "podcast",
    # predict path
    "xyzzy", "qwrtp", "zxcvbnm", "bloop", "flurble", "snarfle", "grobnitz", "kwyjibo",
    # short OOV (letter by letter)
    "xq", "zzq", "qx", "aa", "aaa", "abc", "xyz", "Xq", "XQ", "a1", "1a", "b2b", "x-y",
    # acronyms
    "NASA", "FBI", "CPU", "GPU", "HTTP", "URL", "API", "USB", "DSPGAN", "GPT", "TTS",
    "The NASA and FBI used HTTP APIs.", "iPhone", "eBay", "YouTube", "OpenAI", "PhD", "TV",
    "U.S.A.", "i.e. that", "E.G. this", "a.m. p.m.",
    # numbers, currency, dates
    "1", "42", "2024", "1,000,000", "3.14", "-5", "1st 2nd 3rd 4th 21st 22nd 101st",
    "$5", "$5.50", "$0.01", "£10", "£1.5", "€20", "10%", "3:30 pm", "12:00", "1/2", "3/4",
    "1 + 2 = 3", "5km", "100 km/h", "23°C", "on 2024-11-02", "Nov 2, 2024", "in the 1990s",
    "call 555-1234", "Room 101", "version 2.0.1",
    # text_normalize raises (int(",5") and inflect NumOutOfRangeError)
    "$,5", "costs $,5 now", "1" * 40,
    # raw-only branches (normalize removes these)
    "café", "naïve", "résumé", "über", "straße", "aß", "éa", "ña", "Ça", "İstanbul", "İ",
    "Kelvin", "K", "ǅemal", "ǈ", "Ǉ", "ʼa", "aʼ", "áb", "ﬁx", "ﬁ",
    "日本 hello 中文", "你好world", "123abc", "abc123def", "a_b", "_x_", "x2", "2x", "a'b",
    "'s", "x's", "ab's", "abc's", "Bob's", "'tis", "'twas", "rock'n'roll", "o'neil", "O'Neil",
    "y'all", "e-mail", "co-op", "state-of-the-art", "well-known", "--", "-", "'", "''", "'''",
    # tweet tokenizer specifics
    ":-) :) :( <3", "@user hello", "#hashtag", "http://example.com/path?x=1", "www.example.com",
    "email@example.com", "+1 (555) 123-4567", "loooooool", "soooo goood", "&amp; &lt;",
    "...", "…", "a…b", "Hello...World", "....", "a.b.c", "U.S.", "Mr. Smith", "Dr. Who",
    "tab\tseparated\twords", "new\nline", "multiple   spaces",
    # unk / filtered
    "UW", "uw", "<pad>", "<unk>", "</s>", "<s>",
    # dictionary entries with an empty phone or "#", "FOREIGN", "FRENCH" (dropped by replace_phs)
    "agronomy", "HTTP and http", "Danglars met Porthos.", "danglar", "taxonomy of squamous metabolites",
]


WORDS = (
    "the a an of to and in is it you that he was for on are with as I his they be at one have "
    "this from or had by hot word but what some we can out other were all there when up use your "
    "how said each she which do their time if will way about many then them write would like so "
    "these her long make thing see him two has look more day could go come did number sound no "
    "most people my over know water than call first who may down side been now find"
).split()
CONTRACTIONS = ["can't", "won't", "don't", "it's", "I'm", "you're", "they've", "we'd", "she'll", "isn't", "ain't", "y'all", "o'clock", "let's"]
ACRONYMS = ["NASA", "API", "GPU", "CPU", "macOS", "iPhone", "iOS", "USB", "HTTP", "JSON", "OpenAI", "YouTube", "eBay", "PhD", "TV", "AI", "OS", "HUD"]
PUNCT = [",", ".", "!", "?", ";", ":", "...", " -", "'", '"', "(", ")", "—", "。", "！", "？", "，", "：", "；", "’", "!!", "?!", "..", "……", "--"]
SPACES = [" ", " ", " ", " ", "  ", "\t", "\n"]
LETTERS = "abcdefghijklmnopqrstuvwxyz"


def rand_letters(r, lo, hi):
    return "".join(r.choice(LETTERS) for _ in range(r.randint(lo, hi)))


def case(r, w):
    k = r.random()
    if k < 0.6:
        return w
    if k < 0.85:
        return w.capitalize()
    if k < 0.95:
        return w.upper()
    return "".join(c.upper() if r.random() < 0.5 else c for c in w)


def number(r):
    k = r.randrange(8)
    if k == 0:
        return str(r.randint(0, 100))
    if k == 1:
        return f"{r.randint(1000, 10**7):,}"
    if k == 2:
        return f"{r.randint(0, 99)}.{r.randint(0, 99)}"
    if k == 3:
        return r.choice(["$", "£"]) + str(r.randint(0, 999)) + r.choice(["", f".{r.randint(0, 99):02d}"])
    if k == 4:
        return str(r.randint(1, 100)) + r.choice(["st", "nd", "rd", "th"])
    if k == 5:
        return f"{r.randint(1, 12)}:{r.randint(0, 59):02d}" + r.choice(["", " am", " pm"])
    if k == 6:
        return str(r.randint(1900, 2100))
    return f"{r.randint(1, 9)}/{r.randint(2, 9)}"


def token(r):
    k = r.randrange(20)
    if k <= 4:
        return case(r, r.choice(WORDS))
    if k <= 7:
        return case(r, r.choice(CMU_WORDS))
    if k == 8:
        return r.choice(NAMES_ONLY).capitalize()
    if k == 9:
        return case(r, r.choice(HOMOGRAPHS))
    if k == 10:
        return r.choice(CONTRACTIONS)
    if k == 11:
        return case(r, r.choice(CMU_APOS))
    if k == 12:
        return r.choice(CMU_WORDS + NAMES_ONLY) + "'s"
    if k == 13:
        return r.choice(CMU_WORDS) + r.choice(CMU_WORDS)
    if k == 14:
        return rand_letters(r, 1, 10)
    if k == 15:
        return rand_letters(r, 2, 6).upper()
    if k == 16:
        return r.choice(ACRONYMS)
    if k == 17:
        return number(r)
    if k == 18:
        return r.choice(PUNCT)
    return r.choice(CMU_WORDS) + "-" + r.choice(CMU_WORDS)


def sentence(r):
    parts = []
    for _ in range(r.randint(1, 14)):
        parts.append(token(r))
        if r.random() < 0.25:
            parts[-1] += r.choice(PUNCT)
    out = parts[0]
    for p in parts[1:]:
        out += r.choice(SPACES) + p
    return out


def bulk_sentences():
    r = random.Random(SEED)
    return [sentence(r) for _ in range(3800)]


def bulk_tokens():
    r = random.Random(SEED + 1)
    out = [token(r) for _ in range(1600)]
    # names in every casing: only title case reaches namedict
    for name in r.sample(NAMES_ONLY, 150):
        out += [name, name.capitalize(), name.upper()]
    # random letter strings for the predict path
    out += [rand_letters(r, 4, 14) for _ in range(300)]
    return out


def corpus_inputs():
    corpus = json.loads(CORPUS.read_text())
    out = []
    span_re = re.compile(r"[^　-〿぀-ヿ㐀-鿿＀-￯]+")
    for item in corpus:
        text = item["text"]
        out.append(text)
        for span in span_re.findall(text):
            if re.search(r"[A-Za-z]", span):
                out.append(span)
                out.append(span.strip())
        for lang in ["zh", "auto"]:
            try:
                with contextlib.redirect_stdout(sys.stderr):
                    segs = ls.LangSegmenter.getTexts(text, lang) if lang != "auto" else ls.LangSegmenter.getTexts(text)
            except Exception:  # noqa: BLE001
                continue
            for seg in segs:
                if seg["lang"] == "en":
                    out.append(seg["text"])
    return out


def text_record(text, full):
    rec = {"i": text, "norm": run(english.text_normalize, text)}
    if "o" in rec["norm"]:
        norm = rec["norm"]["o"]
        rec["g2p"] = run(english.g2p, norm)
        if full:
            rec["call"] = run(G, norm)
    else:
        rec["g2p"] = None
    if full:
        rec["raw"] = run(english.g2p, text)
    return rec


# ---------------------------------------------------------------------------
# qryword and helpers


def qryword_inputs():
    r = random.Random(SEED + 2)
    out = ["", "a", "A", "b", "ab", "AB", "Ab", "aB", "xq", "xqz", "Xqz", "XQZ", "ae", "ai", "os", "ios", "hud",
           "chatgpt", "ChatGPT", "json", "read", "complex", "'s", "a's", "ab's", "abc's", "abcd's", "dog's", "Dog's",
           "DOG'S", "dogs's", "cat's", "fish's", "judge's", "church's", "box's", "bob's", "zzzz's", "cafe's",
           "smartwatch", "thequickbrownfox", "abc123", "123", "12", "1234567", "a1b2", "é", "aé", "aß", "ñ", "ña",
           "Kaadi", "İ", "İa", "ǅemal", "ǅ", "x\n", "ab's\n", "abcd's\n", "a-b", "a'b", "o'neil", "rock'n'roll",
           "e.g.", "u.s.a.", "___", "a_b", "x__y", "aaaaaaaaaaaaaaaaaaaaaaaa", "qwertyuiopasdfghjklzxcvbnm"]
    out += r.sample(CMU_WORDS, 500)
    out += [w.upper() for w in r.sample(CMU_WORDS, 100)]
    out += r.sample(CMU_APOS, 100)
    for name in r.sample(NAMES_ONLY, 250):
        out += [name, name.capitalize(), name.upper(), "K" + name[1:] if name.startswith("k") else name.title() + "'s"]
    out += [w + "'s" for w in r.sample(CMU_WORDS, 300)]
    out += [rand_letters(r, 1, 3) for _ in range(200)]
    out += [rand_letters(r, 4, 12) + "'s" for _ in range(100)]
    out += [r.choice(CMU_WORDS) + r.choice(CMU_WORDS) for _ in range(250)]
    out += [rand_letters(r, 4, 16) for _ in range(250)]
    return out


def funcs_inputs():
    r = random.Random(SEED + 3)
    rcp = ["", "a", "!!", "a!!", "a!!b", " !!", " ! ! !", "..", "...", "…", "……", "-", "--", "a--b", ",,,", "?!", "!?.,-…",
           "\t!!", "\n..", "　,,", "\x1c..", "\xa0!!", "a .", "a  ..", "x?!?!y", "end.", "end..", "!a!", "! a !",
           "a;;b", "a::b", "''", "'!!"]
    chars = "ab !?…,.-\t\n;:'　"
    rcp += ["".join(r.choice(chars) for _ in range(r.randint(1, 20))) for _ in range(400)]
    phs = [[], ["'"], ["AH0", "'", "B"], [" "], ["UNK"], ["<unk>"], ["SP"], ["é", "AH0"], ["", "AA1"], ["!", "?", "…", ",", ".", "-"],
           ["a", "ㄱ", "[", "]", "Yi1", "AA", "ER", "IH", "UW"], ["\"", ";", ":", "..."]]
    pool = ["AH0", "'", "-", ",", ".", "!", "?", "...", "ER", "IH", "UW", "UW1", "", " ", "x", "a", "UNK", "SP", "_", "[", "ㄱ"]
    phs += [[r.choice(pool) for _ in range(r.randint(0, 10))] for _ in range(200)]
    out = [{"f": "replace_consecutive_punctuation", "i": t, **run(english.replace_consecutive_punctuation, t)} for t in rcp]
    out += [{"f": "replace_phs", "i": p, **run(english.replace_phs, p)} for p in phs]
    return out


def dedup(xs):
    seen = set()
    return [x for x in xs if not (x in seen or seen.add(x))]


def write_jsonl(name, records):
    path = OUT / name
    with path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"wrote {path} ({len(records)} cases)")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "dict.json").write_text(json.dumps(gen_dict(), ensure_ascii=False, separators=(",", ":")) + "\n")
    print("wrote dict.json")

    hand = dedup(HAND)
    corpus = [t for t in dedup(corpus_inputs()) if t not in set(hand)]
    tokens = [t for t in dedup(bulk_tokens()) if t not in set(hand) | set(corpus)]
    sents = [t for t in dedup(bulk_sentences()) if t not in set(hand) | set(corpus) | set(tokens)]
    records = []
    for src, texts, full in [("hand", hand, True), ("corpus", corpus, True), ("tokens", tokens, True), ("sentences", sents, False)]:
        for t in texts:
            rec = text_record(t, full)
            rec["src"] = src
            records.append(rec)
    write_jsonl("texts.jsonl", records)

    words = dedup(qryword_inputs())
    write_jsonl("qryword.jsonl", [{"i": w, **run(G.qryword, w)} for w in words])
    write_jsonl("funcs.jsonl", funcs_inputs())


if __name__ == "__main__":
    main()

"""Generate golden fixtures for the wordsegment port.

Run: pixi run --manifest-path crates/ports/wordsegment/pixi.toml fixtures

Writes to tests/fixtures/ of this crate:
- meta.json:       sizes and samples of UNIGRAMS / BIGRAMS / WORDS after load()
- segment.jsonl:   {"text", "clean", "segment"} per input
- score.jsonl:     {"word", "previous", "bits"} per (word, previous) pair;
                   "bits" is the f64 bit pattern as 16 hex digits so the Rust
                   test compares exactly without depending on decimal parsing
- divide.jsonl:    {"text", "limit", "divide"} per input
"""

import json
import random
import string
import struct
from pathlib import Path

import wordsegment
from wordsegment import Segmenter

assert wordsegment.__version__ == "1.3.1", wordsegment.__version__

CRATE = Path(__file__).resolve().parents[1]
OUT = CRATE / "tests" / "fixtures"
OUT.mkdir(parents=True, exist_ok=True)

wordsegment.load()
UNI = wordsegment.UNIGRAMS
BI = wordsegment.BIGRAMS
WORDS = wordsegment.WORDS

rng = random.Random(20261102)


def bits(x: float) -> str:
    return struct.pack(">d", x).hex()


def dump_jsonl(name, rows):
    with open(OUT / name, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"{name}: {len(rows)} rows")


# ---------------------------------------------------------------- inputs
uni_by_freq = list(UNI)  # file order is descending frequency
common = uni_by_freq[:20000]
rare = uni_by_freq[20000:]
alpha_words = [w for w in WORDS if w.isalpha()]

HAND = [
    "", " ", "a", "I", "x", "zz", "qq", "0", "9", "42", "1984", "2024", "007",
    "thisisatest", "choosespain", "choosespain.com", "wheninthecourseofhumanevents",
    "itwasabrightcolddayinaprilandtheclockswerestrikingthirteen",
    "inaholeinthegroundtherelivedahobbitnotanastydirtywetholefilledwiththeendsofwormsandanoozysmell",
    "Hello, World!", "HELLO WORLD", "don't", "it's", "o'clock", "rock'n'roll",
    "can't-stop", "e-mail", "a.b.c", "U.S.A.", "mp3player", "windows10",
    "iphone15promax", "covid19", "web2.0", "3d printing", "4k60fps", "1password",
    "abc123def456", "123abc", "a1b2c3d4", "---", "!!!", "\t\n", "日本語", "naïve",
    "café", "résumé", "straße", "İstanbul", "Kelvin", "ﬁnance", "Ångström",
    "xyzzy", "qwertyuiop", "asdfghjkl", "zxcvbnm", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "pneumonoultramicroscopicsilicovolcanoconiosis", "supercalifragilisticexpialidocious",
    "antidisestablishmentarianism", "floccinaucinihilipilification",
    "thequickbrownfoxjumpsoverthelazydog", "nowisthetimeforallgoodmentocometotheaidoftheircountry",
    "openai", "chatgpt", "github", "youtube", "playstation", "macbookpro", "airpods",
    "tensorflow", "pytorch", "javascript", "typescript", "postgresql", "kubernetes",
    "huggingface", "stackoverflow", "wordpress", "linkedin", "whatsapp", "photoshop",
    "nintendoswitch", "xboxone", "powerpoint", "onedrive", "dropbox", "spotify",
    "gptsovits", "ortts", "moeru", "airi", "neko", "ayaka",
]

CAMEL = [
    "iPhone", "iPad", "MacBook", "PlayStation", "YouTube", "GitHub", "LinkedIn",
    "PowerPoint", "OneDrive", "WhatsApp", "JavaScript", "TypeScript", "PostgreSQL",
    "OpenAI", "ChatGPT", "DeepMind", "TikTok", "SoundCloud", "FaceTime", "AirDrop",
    "HomePod", "SpaceX", "StarLink", "DoorDash", "SnapChat", "WordPress", "DropBox",
    "VSCode", "IntelliJ", "PyCharm", "NodeJS", "ReactNative", "TensorFlow", "PyTorch",
    "HuggingFace", "StackOverflow", "MySpace", "NetFlix", "BitCoin", "EtherScan",
]


def concat_words(pool, lo, hi):
    return "".join(rng.choice(pool) for _ in range(rng.randint(lo, hi)))


def camel_name():
    parts = [rng.choice(common).capitalize() for _ in range(rng.randint(2, 4))]
    if rng.random() < 0.3:
        parts.append(str(rng.randint(0, 2030)))
    if rng.random() < 0.3:
        parts[0] = parts[0].lower()
    return "".join(parts).lower()


SYLL_ON = ["", "b", "br", "ch", "d", "dr", "f", "g", "gr", "h", "j", "k", "kr", "l",
           "m", "n", "p", "ph", "qu", "r", "s", "sh", "st", "t", "th", "tr", "v", "w",
           "x", "y", "z", "zh"]
SYLL_NU = ["a", "e", "i", "o", "u", "ai", "ei", "ou", "ia", "ae", "y", "oo"]
SYLL_CO = ["", "", "n", "ng", "r", "s", "x", "k", "l", "m", "th"]


def oov_name():
    return "".join(rng.choice(SYLL_ON) + rng.choice(SYLL_NU) + rng.choice(SYLL_CO)
                   for _ in range(rng.randint(1, 5)))


def with_digits_apostrophes():
    parts = []
    for _ in range(rng.randint(2, 5)):
        r = rng.random()
        if r < 0.5:
            parts.append(rng.choice(common))
        elif r < 0.7:
            parts.append(str(rng.randint(0, 99999)))
        elif r < 0.85:
            parts.append(rng.choice(common) + "'" + rng.choice(["s", "t", "re", "ll", "d", "ve"]))
        else:
            parts.append(rng.choice(["-", " ", ".", "_", "'", ",", "&", "/"]))
    return "".join(parts)


def random_chars(n, alphabet):
    return "".join(rng.choice(alphabet) for _ in range(n))


inputs = list(HAND)
inputs += [c.lower() for c in CAMEL]
inputs += CAMEL
for _ in range(1500):
    inputs.append(concat_words(common, 2, 5))
for _ in range(500):
    inputs.append(concat_words(rare, 2, 5))
for _ in range(500):
    inputs.append(concat_words(alpha_words, 2, 5))
for _ in range(700):
    inputs.append(camel_name())
for _ in range(700):
    inputs.append(oov_name())
for _ in range(500):
    inputs.append(with_digits_apostrophes())
for _ in range(300):
    inputs.append(random_chars(rng.randint(1, 3), string.ascii_lowercase + string.digits))
for _ in range(200):
    inputs.append(random_chars(rng.randint(4, 30), string.ascii_lowercase))
for _ in range(500):
    s = concat_words(common, 8, 16)
    while len(s) < 40:
        s += rng.choice(common)
    inputs.append(s)
for _ in range(100):
    inputs.append(random_chars(rng.randint(40, 80), string.ascii_lowercase + string.digits))
# Single dictionary words (exercise the in-unigram / OOV score paths directly).
inputs += rng.sample(common, 150) + rng.sample(rare, 150) + rng.sample(alpha_words, 150)
# Inputs longer than one 250-character chunk (chunk + 5-word carry-over).
for n in (249, 250, 251, 255, 300, 499, 500, 501, 760):
    s = ""
    while len(s) < n:
        s += rng.choice(common)
    inputs.append(s[:n])
for _ in range(10):
    s = ""
    while len(s) < 600:
        s += rng.choice(common) + rng.choice(["", " ", "'", ", ", "1"])
    inputs.append(s)
inputs.append("a" * 260)
inputs.append(random_chars(270, string.ascii_lowercase))

seen = set()
segment_rows = []
for text in inputs:
    if text in seen:
        continue
    seen.add(text)
    segment_rows.append({
        "text": text,
        "clean": wordsegment.clean(text),
        "segment": wordsegment.segment(text),
    })
dump_jsonl("segment.jsonl", segment_rows)

# ---------------------------------------------------------------- score
seg = wordsegment._segmenter
score_pairs = []
bigram_keys = list(BI)
for key in rng.sample(bigram_keys, 1500):
    prev, _, word = key.partition(" ")
    score_pairs.append((word, prev))
# Duplicated bigram keys in bigrams.txt (dict keeps the last value).
dup_seen, dups = set(), []
with open(Segmenter.BIGRAMS_FILENAME, encoding="utf-8") as f:
    for line in f:
        k = line.split("\t")[0]
        if k in dup_seen:
            dups.append(k)
        dup_seen.add(k)
for key in rng.sample(dups, 200):
    prev, _, word = key.partition(" ")
    score_pairs.append((word, prev))
for w in rng.sample(uni_by_freq, 800):
    score_pairs.append((w, None))
    score_pairs.append((w, rng.choice(common)))
    score_pairs.append((w, "<s>"))
for n in list(range(0, 31)) + [50, 100, 200, 308]:
    w = random_chars(n, string.ascii_lowercase)
    score_pairs.append((w, None))
    score_pairs.append((w, rng.choice(common)))
for _ in range(300):
    score_pairs.append((oov_name(), rng.choice([None, "<s>", oov_name(), rng.choice(common)])))
for w in ["the", "a", "i", "0", "", "日本", "naïve"]:
    score_pairs.append((w, None))
    score_pairs.append((w, "the"))
    score_pairs.append((w, "日本"))

score_rows = []
for word, prev in score_pairs:
    score_rows.append({"word": word, "previous": prev, "bits": bits(seg.score(word, prev))})
dump_jsonl("score.jsonl", score_rows)

# ---------------------------------------------------------------- divide
divide_rows = []
for text in ["", "a", "ab", "abc", "日本語テキスト", "naïve", "x" * 30,
             "abcdefghijklmnopqrstuvwxyz0123"]:
    for limit in (0, 1, 3, 24, 100):
        s = Segmenter()
        s.limit = limit
        divide_rows.append({"text": text, "limit": limit,
                            "divide": [list(p) for p in s.divide(text)]})
dump_jsonl("divide.jsonl", divide_rows)

# ---------------------------------------------------------------- meta
meta = {
    "unigrams_len": len(UNI),
    "bigrams_len": len(BI),
    "words_len": len(WORDS),
    "words_head": WORDS[:20],
    "words_tail": WORDS[-20:],
    "words_sample": sorted(rng.sample(range(len(WORDS)), 200)),
    "total_bits": bits(seg.total),
    "limit": seg.limit,
}
meta["words_sample"] = [[i, WORDS[i]] for i in meta["words_sample"]]
with open(OUT / "meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=1)
print("meta.json written")

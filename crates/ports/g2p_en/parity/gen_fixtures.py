"""Generate golden fixtures for the `g2p_en` port crate.

Run from the crate directory with:

    pixi run fixtures

The task first fetches GPT-SoVITS and the nltk data into parity/.cache (see
gpt_sovits_env.py for the environment overrides). Writes to tests/fixtures/:

- predict.jsonl: one {"src", "word", "phones"} object per line, where phones is
  `G2p().predict(word)` from the pinned g2p_en 2.1.0.
- homograph2features.json: `G2p().homograph2features` as the parent
  `G2p.__init__` builds it (before the en_G2p overrides in GPT-SoVITS), as
  [[headword, pron1, pron2, pos1], ...] in dict insertion order.
- vocab.json: the graphemes and phonemes lists.

Inputs (deduplicated, the first source wins):

- edge: hand-written edge cases (empty string, single letters, characters
  outside a-z that map to <unk>, words long enough to hit the 20-step cap).
- oov / names / products: hand-written words that are not in CMUdict.
- gpt_sovits: every word that GPT-SoVITS `text.english.g2p` sends to
  `predict` for the parity corpus (parity/corpus/gpt_sovits_zh_en_yue.json)
  and a set of English sentences.
- cmudict: seeded sample of the GPT-SoVITS cmudict.rep / cmudict-fast.rep words.
- namedict: seeded sample of the GPT-SoVITS namedict_cache.pickle words.
- random_az: seeded random [a-z] strings of length 1..20.
- random_mixed: seeded random strings over a-z plus apostrophe, hyphen, dot,
  digits, uppercase, space and non-ASCII letters, length 0..25.
"""

import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "tests" / "fixtures"
sys.path.insert(0, str(HERE))

import gpt_sovits_env  # noqa: E402  (sys.path for GPT-SoVITS, NLTK_DATA)
from g2p_en import G2p  # noqa: E402
from text import english  # noqa: E402

SEED = 20240521

EDGE = [
    "", "a", "b", "x", "z", "aa", "ab", "zz", "'", "-", "''", "'s", "o'", "1",
    "42", "A", "ABC", "Hello", "é", "café", "café", "naïve", "über",
    "straße", "ﬁ", "你好", "日本語", "\U0001f600", "a\U0001f600b", " ",
    "a b", "\t", "<pad>", "<unk>", "</s>", "<s>",
    "abcdefghijklmnopqrstuvwxyz", "zyxwvutsrqponmlkjihgfedcba",
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "zzzzzzzzzzzzzzzzzzzzzzzzzzzzzz",
    "supercalifragilisticexpialidocious", "antidisestablishmentarianism",
    "pneumonoultramicroscopicsilicovolcanoconiosis",
    "honorificabilitudinitatibus",
    "llanfairpwllgwyngyllgogerychwyrndrobwllllantysiliogogogoch",
    "qwertyuiopasdfghjklzxcvbnm", "xxxxxxxxxx", "ababababababababababab",
    "mississippi", "e.g.", "i.e.", "u.s.a.", "rock'n'roll", "o'neil", "don't",
    "can't", "won't", "y'all", "e-mail", "co-op", "x-ray", "state-of-the-art",
    "abc123", "mp3", "4k",
]

OOV = [
    "activationist", "unputdownable", "hangry", "doomscrolling", "cromulent",
    "embiggen", "flibbertigibbet", "snollygoster", "bamboozlement", "glamping",
    "phubbing", "staycation", "mansplain", "brexiteer", "covidiot", "vaxxer",
    "selfie", "photobomb", "unfriend", "hashtagging", "blockchainify",
    "metaverse", "deepfake", "cryptojacking", "microservices", "serverless",
    "containerization", "kubernetes", "dockerfile", "webassembly",
    "typescript", "javascript", "postgresql", "nginx", "rustacean", "pythonic",
    "tokenizer", "vocoder", "phoneme", "grapheme", "transformerless",
    "hyperparameter", "finetuning", "dataloader", "backpropagation",
    "overfitting", "multilingual", "zeroshot", "upsampling", "spectrogram",
    "mel", "onnx", "ortts", "sovits", "dspgan", "zxqvbn", "qxz", "bcdfg",
    "rhythmless", "strengths", "twelfths", "queueing", "aeiou", "ouija",
    "eunoia", "syzygy", "crwth", "cwm", "tsktsk", "brrr", "hmmmm", "shhhh",
    "zzz", "ugh", "pfft", "psst",
]

NAMES = [
    "nguyen", "xiaoming", "zhang", "wang", "liu", "chen", "yang", "huang",
    "zhao", "zhou", "takahashi", "watanabe", "yamamoto", "nakamura",
    "kobayashi", "kyubyong", "jongseok", "siobhan", "niamh", "saoirse",
    "aoife", "tadhg", "dvorak", "przybylski", "szczepanski", "wojciechowski",
    "brzezinski", "tchaikovsky", "dostoevsky", "nietzsche", "schwarzenegger",
    "goethe", "beethoven", "zelenskyy", "oyelaran", "chukwuemeka", "olusegun",
    "aaliyah", "jaxon", "brayden", "khaleesi", "daenerys", "hermione",
    "voldemort", "frodo", "gandalf", "legolas", "ayaka", "neko", "moeru",
    "airi",
]

PRODUCTS = [
    "chatgpt", "openai", "anthropic", "deepseek", "qwen", "llama", "mistral",
    "huggingface", "pytorch", "tensorflow", "numpy", "scipy", "matplotlib",
    "jupyter", "github", "gitlab", "bitbucket", "youtube", "tiktok",
    "instagram", "whatsapp", "wechat", "weibo", "bilibili", "alibaba",
    "taobao", "xiaomi", "huawei", "oneplus", "iphone", "ipad", "macbook",
    "airpods", "playstation", "xbox", "nintendo", "pokemon", "minecraft",
    "fortnite", "genshin", "hoyoverse", "spotify", "netflix", "airbnb", "uber",
    "lyft", "doordash", "instacart", "salesforce", "snowflake", "databricks",
    "cloudflare", "vercel", "netlify", "supabase", "firebase", "mongodb",
    "redis", "kafka", "elasticsearch", "grafana", "prometheus", "terraform",
    "ansible", "jenkins", "nvidia", "geforce", "radeon", "ryzen",
    "threadripper", "snapdragon", "exynos", "tesla", "cybertruck",
    "lamborghini", "volkswagen", "hyundai", "peugeot", "ikea", "lego",
    "starbucks", "mcdonald's", "frappuccino", "skittles", "doritos",
]

# English sentences sent through GPT-SoVITS text.english.g2p to collect the
# words it really passes to predict (OOV after CMUdict, namedict, possessive
# split and wordsegment).
SENTENCES = [
    "e.g. I used openai's AI tool to draw a picture.",
    "In this; paper, we propose 1 DSPGAN, a GAN-based universal vocoder.",
    "I'm an activationist, and Zxqvbn's GPT-SoVITS rocks.",
    "Kubernetes, Dockerfile and WebAssembly walked into a bar.",
    "Flibbertigibbet snollygoster cromulent embiggen bamboozlement.",
    "Doomscrolling hangry vaxxers photobomb the staycation.",
    "Przybylski, Szczepanski and Wojciechowski met Chukwuemeka.",
    "The rhythmless crwth and cwm syzygy; tsktsk, brrr, pfft, psst.",
    "Ayaka and Neko from Moeru AI built AIRI with ORTTS.",
    "My favourite colour is grey, said the neighbour's labourer.",
    "Xiaoming, Zhang Wei and Takahashi visited Huawei and Xiaomi.",
    "Qwen, DeepSeek, Llama and Mistral are LLMs; ChatGPT isn't.",
    "Rock'n'roll y'all, it's state-of-the-art e-mail co-op x-ray stuff!",
    "Zyxwvuts qponmlk jihgfedcba; qwertyuiop asdfghjkl zxcvbnm.",
    "Glorptastic blunderwhizz frimbulated the quaxomatic snorfle.",
    "Vrrrrm, kssshhh, thwomp, zaaap went the brzzt machine.",
    "Ouija eunoia aeiou queueing; strengths twelfths.",
    "Llanfairpwllgwyngyllgogerychwyrndrobwllllantysiliogogogoch is a town.",
    "Schwarzenegger's Lamborghini outran the Cybertruck's Threadripper.",
    "Brzęczyszczykiewicz, Grzegorz and Café Ünïcödé are naïve.",
]

AZ = "abcdefghijklmnopqrstuvwxyz"
ALPHA_MIXED = list(AZ) * 3 + list("'-.0123456789ABCXYZ ") + ["é", "ü", "ñ", "ß", "中", "ア"]


def cmudict_words():
    """Headwords of the dictionaries GPT-SoVITS read_dict_new() loads."""
    words = []
    seen = set()
    for name, start, sep in (("cmudict.rep", 57, "  "), ("cmudict-fast.rep", 1, " ")):
        path = gpt_sovits_env.UPSTREAM / "GPT_SoVITS" / "text" / name
        with open(path) as f:
            for idx, line in enumerate(f, 1):
                if idx < start:
                    continue
                w = line.strip().split(sep)[0].lower()
                if w and w not in seen:
                    seen.add(w)
                    words.append(w)
    return words


def main():
    g2p = G2p()
    rng = random.Random(SEED)
    items = []
    seen = set()

    def add(src, words):
        for w in words:
            if w not in seen:
                seen.add(w)
                items.append((src, w))

    add("edge", EDGE)
    add("oov", OOV)
    add("names", NAMES)
    add("products", PRODUCTS)

    # Words GPT-SoVITS passes to predict.
    captured = []
    eg2p = english._g2p
    orig = eg2p.predict

    def hook(word):
        captured.append(word)
        return orig(word)

    eg2p.predict = hook
    corpus = json.loads((HERE / "corpus" / "gpt_sovits_zh_en_yue.json").read_text())
    for text in [c["text"] for c in corpus] + SENTENCES:
        english.g2p(english.text_normalize(text))
    del eg2p.predict
    add("gpt_sovits", captured)

    add("cmudict", rng.sample(cmudict_words(), 2600))
    add("namedict", rng.sample(sorted(eg2p.namedict.keys()), 600))
    add("random_az", ["".join(rng.choice(AZ) for _ in range(rng.randint(1, 20))) for _ in range(2600)])
    add("random_mixed", ["".join(rng.choice(ALPHA_MIXED) for _ in range(rng.randint(0, 25))) for _ in range(600)])

    OUT.mkdir(parents=True, exist_ok=True)
    capped = 0
    with open(OUT / "predict.jsonl", "w", encoding="utf-8") as f:
        for src, w in items:
            phones = g2p.predict(w)
            capped += len(phones) == 20
            f.write(json.dumps({"src": src, "word": w, "phones": phones}, ensure_ascii=False, separators=(",", ":")) + "\n")

    homographs = [[k, p1, p2, pos] for k, (p1, p2, pos) in g2p.homograph2features.items()]
    (OUT / "homograph2features.json").write_text(json.dumps(homographs, separators=(",", ":")) + "\n")
    (OUT / "vocab.json").write_text(json.dumps({"graphemes": g2p.graphemes, "phonemes": g2p.phonemes}, separators=(",", ":")) + "\n")

    counts = {}
    for src, _ in items:
        counts[src] = counts.get(src, 0) + 1
    print(f"{len(items)} predict cases {counts}; {len(captured)} gpt_sovits calls; {capped} hit the 20-step cap; {len(homographs)} homographs", file=sys.stderr)


if __name__ == "__main__":
    main()

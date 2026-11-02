"""Generate the golden fixtures for the nltk port.

    pixi run --manifest-path crates/ports/nltk/pixi.toml fixtures

Writes to tests/fixtures/ of this crate:

- `regex_sources.json`: the regex sources and flags of nltk.tokenize.casual,
  so the Rust test checks that the copied constants are identical.
- `char_classes.json`: for every single-character construct used in the
  ported regexes, the code points the Python `regex` module matches (as
  ranges), with the flags nltk compiles them with.
- `python_lower.json`: `str.lower()` for every code point it changes.
- `tokenize.jsonl`: `TweetTokenizer().tokenize(text)` and
  `pos_tag(tokens)` for hand-written edge cases and generated inputs.
- `tokenize_options.jsonl`: `TweetTokenizer(**opts).tokenize(text)` for the
  non-default options.
- `pos_tag.jsonl`: `PerceptronTagger().tag(tokens, use_tagdict=...)` for
  token lists that do not come from the tokenizer.
- `corpus.jsonl`: every text that GPT-SoVITS `english.py` passes to
  `word_tokenize` while running the frontend over
  parity/corpus/gpt_sovits_zh_en_yue.json, with tokens and tags.

Python returns lone surrogates for `&#xD800;`..`&#xDFFF;`. JSON cannot carry
them to Rust, so they are replaced with U+FFFD and the record gets
`"surrogate": true` (see parity/README.md).
"""

import contextlib
import json
import random
import string
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import CORPUS, NLTK_DATA  # noqa: E402  (also sets NLTK_DATA)
from paths import FIXTURES as OUT  # noqa: E402

import nltk.data  # noqa: E402
import regex  # noqa: E402

nltk.data.path[:] = [str(NLTK_DATA)]
import gen_data  # noqa: E402
from nltk import pos_tag  # noqa: E402
from nltk.tag.perceptron import PerceptronTagger  # noqa: E402
from nltk.tokenize import TweetTokenizer  # noqa: E402
from nltk.tokenize import casual  # noqa: E402

TOKENIZER = TweetTokenizer()
TAGGER = PerceptronTagger()


def fix_surrogates(s):
    out = []
    bad = False
    for ch in s:
        if 0xD800 <= ord(ch) <= 0xDFFF:
            out.append("\ufffd")
            bad = True
        else:
            out.append(ch)
    return "".join(out), bad


def dump_jsonl(name, records):
    path = OUT / name
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"{name}: {len(records)} records, {path.stat().st_size} bytes", file=sys.stderr)


def dump_json(name, obj):
    path = OUT / name
    path.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8", newline="\n")
    print(f"{name}: {path.stat().st_size} bytes", file=sys.stderr)


def ranges(codepoints):
    out = []
    for c in codepoints:
        if out and out[-1][1] == c - 1:
            out[-1][1] = c
        else:
            out.append([c, c])
    return out


# ---------------------------------------------------------------------------
# regex sources and character classes


def regex_sources():
    p = casual.TweetTokenizer().PHONE_WORD_RE
    w = casual.TweetTokenizer().WORD_RE
    return {
        "EMOTICONS": casual.EMOTICONS,
        "URLS": casual.URLS,
        "FLAGS": casual.FLAGS,
        "PHONE_REGEX": casual.PHONE_REGEX,
        "REGEXPS": list(casual.REGEXPS),
        "REGEXPS_PHONE": list(casual.REGEXPS_PHONE),
        "WORD_RE": [w.pattern, w.flags],
        "PHONE_WORD_RE": [p.pattern, p.flags],
        "HANG_RE": [casual.HANG_RE.pattern, casual.HANG_RE.flags],
        "EMOTICON_RE": [casual.EMOTICON_RE.pattern, casual.EMOTICON_RE.flags],
        "ENT_RE": [casual.ENT_RE.pattern, casual.ENT_RE.flags],
        "HANDLES_RE": [casual.HANDLES_RE.pattern, casual.HANDLES_RE.flags],
        "flags": {"VERBOSE": int(regex.VERBOSE), "I": int(regex.I), "UNICODE": int(regex.UNICODE)},
        "atoms": {
            name: gen_data.atoms(
                gen_data.strip_verbose(src) if flags & regex.VERBOSE else src, bool(flags & regex.I)
            )
            for name, src, flags in gen_data.patterns()
        },
    }


# Probes: (pattern, verbose+ignorecase?, prefix). Every character-class atom
# of the ported regexes (see gen_data.atoms), probed with the flags of the
# regex it occurs in, plus `\\b` after a word character. A probe without a
# prefix records the code points the pattern matches on its own; a probe with
# a prefix records the code points `c` for which the pattern matches at the
# start of `prefix + c`.
def class_probes():
    probes = []
    for _, src, flags in gen_data.patterns():
        vi = bool(flags & regex.I)
        if flags & regex.VERBOSE:
            src = gen_data.strip_verbose(src)
        for atom in gen_data.atoms(src, vi):
            if (atom, vi, "") not in probes:
                probes.append((atom, vi, ""))
    probes.append((r"\w\b", True, "a"))
    probes.append((r"\w\b", True, "_"))
    return probes


ALL_CHARS = "".join(chr(c) for c in range(0x110000) if not 0xD800 <= c <= 0xDFFF)


def char_classes():
    out = []
    for pat, vi, prefix in class_probes():
        flags = (regex.VERBOSE | regex.I) if vi else 0
        rx = regex.compile(pat, flags)
        if prefix:
            hits = [ord(ch) for ch in ALL_CHARS if rx.match(prefix + ch)]
        else:
            hits = []
            for m in rx.finditer(ALL_CHARS):
                assert m.end() - m.start() == 1, pat
                hits.append(ord(m.group()))
        out.append({"pattern": pat, "verbose_ignorecase": vi, "prefix": prefix, "ranges": ranges(hits)})
    return out


def python_lower():
    out = []
    for c in range(0x110000):
        if 0xD800 <= c <= 0xDFFF:
            continue
        lo = chr(c).lower()
        if lo != chr(c):
            out.append([c, lo])
    # Final_Sigma contexts: U+03A3 around cased, case-ignorable and other
    # characters.
    rng = random.Random(3)
    pool = ["\u03a3", "\u03a3", "A", "a", "'", ".", " ", "1", "\u0345", "\u00ad", "\u0300", "\u02b0",
            "\u2160", "\u24b6", "\u0130", "\u1e9e", "\U00010d50", "\u10a0", "\u03c3", "\u01c5", "\u3000", "\u200d"]
    strings = []
    for _ in range(3000):
        t = "".join(rng.choice(pool) for _ in range(rng.randint(1, 6)))
        strings.append([t, t.lower()])
    return {"chars": out, "strings": strings}


# ---------------------------------------------------------------------------
# inputs

HAND = [
    "This is a cooool #dummysmiley: :-) :-P <3 and some arrows < > -> <--",
    "@remy: This is waaaaayyyy too much for you!!!!!!",
    "@remy: This is waaaaayyyy too much for you!!!!!!!!!!",
    "",
    " ",
    "\t\n\r",
    "Hello, world.",
    "Hello world",
    "I can't believe it's not butter!",
    "John's big idea isn't all that bad.",
    "The dogs' bones were buried in the Jones's yard.",
    "rock 'n' roll, 'tis the season, y'all",
    "O'Neill and D'Angelo went to McDonald's.",
    "Don't, won't, shouldn't've, I'd've, ma'am",
    "state-of-the-art, well-known, e-mail, x-ray, re-enter",
    "--dash-- -leading trailing- a--b",
    "under_score _leading trailing_ a__b",
    "Visit https://www.example.com/path?query=1&b=2#frag now.",
    "http://example.com, https://example.com/. ftp://example.com",
    "http://t.co/abc123 http:/x https:///y http:a http:%41",
    "See www.example.org/page_(disambiguation) and en.wikipedia.org/wiki/Foo_(bar_(baz)).",
    "example.com example.co.uk foo.bar sub.example.museum a.b",
    "Mail me at john.doe+tag@example.co.uk or jane_doe@mail-server.org.",
    "foo@gmail.com foo@_gmail.com_ x@y.z bad@ @handle@x.com",
    "foo.na@example.com and foo.na",
    "@user @user_name @a @_ @1234 @verylonghandle1234567890 user@",
    "#hashtag #hash_tag #a #1 #hash-tag #don't ##double #it's_ok",
    "Call me at 555-123-4567 or (555) 123-4567 or +1 555 123 4567.",
    "Phone: 1-800-555-0199, 555.123.4567, 5551234567, 123-4567, 12345678901234567890",
    "+0 (123)) 456 *7890 and 0-123-456-7890",
    "Numbers: 1,000,000 3.14159 -42 +7 1/2 12:30 2020-01-01 1.5e10 $9.99 50% 3rd 21st 2nd 4th 100th",
    "In 1999, 2024 and 12345, we had 3 cats and 1,2,3.",
    "...and then... . . . wait.. ok.\n.\n.",
    "Wait.... what?!?! No!!!! Yes???? Hmm,,,, ;;;; ::::",
    "Emoticons: :) :( ;) :D :P :p :-D =) 8) :/ :\\ :| :} :{ :@ :] :[ <3 </3 >:( >:) (: ): d: D:",
    ":-))) :)) 8-) 8-D :'( :') :*( :o) :-o :o :O",
    "HTML <b>bold</b> <a href=x> <br/> <> < > <<>>",
    "Arrows -> --> ---> <- <-- <--- <-> =>",
    "Entities: &amp; &lt;tag&gt; &quot;quoted&quot; &copy; 2024 &nbsp;x &eacute;t&eacute;",
    "Numeric: &#65;&#66;&#x43;&#X44; &#x41;&#128;&#129;&#150;&#x9F;&#159;&#160;",
    "Bad: &#; &#x; &#xZZ; &unknown; &#99999999; &#-1; &#+65; &#0x41; &#x0x41; &#4_1; &#x_41;",
    "More: &#1114111; &#1114112; &#x10FFFF; &#x110000; &#0; &amp &amp;amp; &&amp;; &#\u0664\u0661;",
    "Quirks: &x41; &xamp; &#xd800; &#55296; &#xDFFF; &#57344; &AMP; &Amp;",
    "Unicode quotes: \u201cHello,\u201d she said. \u2018Hi\u2019 he replied. \u00abBonjour\u00bb",
    "Ellipsis \u2026 and em\u2014dash and en\u2013dash, \u00a0nbsp\u00a0, zero\u200bwidth",
    "Caf\u00e9 na\u00efve r\u00e9sum\u00e9 \u00fcber Stra\u00dfe \u0130stanbul \u0131 \u017fs",
    "Greek: \u039f\u0394\u03a5\u03a3\u03a3\u0395\u03a5\u03a3 \u03b1\u03b2\u03b3 \u03a3\u03b1\u03c2",
    "Mixed CaSe WoRdS ALLCAPS lowercase Title Case iPhone eBay McDonald",
    "Emoji: \U0001F600 \U0001F44D\U0001F3FD \U0001F468\u200d\U0001F469\u200d\U0001F467 \u2764\ufe0f \U0001F1FA\U0001F1F8 \U0001F1EF\U0001F1F5\U0001F1E9",
    "Flags: \U0001F3F4\U000E0067\U000E0062\U000E0065\U000E006e\U000E0067\U000E007F \U0001F3F4\U000E0067\U000E0062\U000E0073\U000E0063\U000E0074\U000E007F \U0001F3F4\U000E0067\U000E0062\U000E0077\U000E006C\U000E0073\U000E007F",
    "a\u200db \u200d \U0001F3FB alone x\U0001F3FF",
    "\u4f60\u597d world \u3053\u3093\u306b\u3061\u306f \uc548\ub155 \u0645\u0631\u062d\u0628\u0627 \u05e9\u05dc\u05d5\u05dd \u0928\u092e\u0938\u094d\u0924\u0947",
    "Fullwidth \uff21\uff22\uff23 \uff11\uff12\uff13 \uff08x\uff09",
    "Arabic digits \u0661\u0662\u0663 \u0664\u0665\u0666\u0667 and superscripts \u00b2\u00b3 \u2460\u2461 \u00bd",
    "Tabs\tand\nnewlines\r\nand  multiple   spaces",
    "$$$$ #### @@@@ !!!!!!!! ???? ---- ____ ==== ++++ **** .... ,,,, ;;;; ~~~~",
    "aaaaaaa bbbbbb 1111111 \u00e9\u00e9\u00e9\u00e9\u00e9 \U0001F600\U0001F600\U0001F600\U0001F600\U0001F600",
    "The quick brown fox jumps over the lazy dog.",
    "I read the book yesterday and I will read it again tomorrow.",
    "He refused to pay the refuse collection fee.",
    "The complex is a complex structure.",
    "Time flies like an arrow; fruit flies like a banana.",
    "Buffalo buffalo Buffalo buffalo buffalo buffalo Buffalo buffalo.",
    "AI, OS, iOS, HUD, AE, AR, GPU, CPU, NASA, FBI, U.S.A., e.g., i.e., etc.",
    "Mr. Smith, Dr. Jones and Mrs. Brown met at 5 p.m. on Jan. 3rd.",
    "a b c d e f g h i j k l m n o p q r s t u v w x y z A I",
    "1 2 3 4 5 6 7 8 9 0 10 100 1000 10000 2023 1984 0000 9999",
    "-1 -2023 +2023 2023- 20-23 2,023 20.23 1984th",
    "x-1 1-x -x x- -- - --- 1-2-3",
    "It costs \u00a3100, \u20ac50, \u00a510 or $20.",
    "COVID-19 SARS-CoV-2 H1N1 MP3 4K 3D B2B 24/7 9/11",
    "C++ C# F# .NET node.js Vue.js",
    "<3<3<3 :):):) ;-);-)",
    "a.b.c.d e,f,g h;i;j k:l:m",
    "1.2.3 1,2,3 1/2/3 1:2:3 1-2-3 +1-2- -1+",
    "who's what's where's there's here's",
    "I'M SHOUTING DON'T YOU SEE",
    "naive Naive NAIVE na\u00efve",
    "'quoted' \"double\" `backtick` ''two'' ``latex''",
    "(parenthesized) [bracketed] {braced} <angled>",
    "a/b a\\b a|b a~b a^b a=b a+b a*b",
    "100% 50\u00b0C 6'2\" 3.5mm 10km/h",
    "http://a.b/c(d)e http://x.com/(a(b)c) https://x.org/a)b",
    "goo.gl/abc bit.ly/xyz t.co/q",
    "Hi.My name is Bob.Nice to meet you!How are you?",
    "\u0130I\u0131i \u1e9e\u00df \u212a k \u2126 \u03c9",
    "\u01c4 \u01c5 \u01c6 \u1f88 \u1f80",
]

WORDS_SIMPLE = (
    "the of and to a in is it you that he was for on are with as I his they be at one have this from "
    "or had by hot word but what some we can out other were all there when up use your how said an each "
    "she which do their time if will way about many then them write would like so these her long make "
    "thing see him two has look more day could go come did number sound no most people my over know "
    "water than call first who may down side been now find any new work part take get place made live "
    "where after back little only round man year came show every good me give our under name very through "
    "just form sentence great think say help low line differ turn cause much mean before move right boy old "
    "too same tell does set three want air well also play small end put home read hand port large spell add "
    "even land here must big high such follow act why ask men change went light kind off need house picture "
    "try us again animal point mother world near build self earth father head stand own page should country "
    "found answer school grow study still learn plant cover food sun four between state keep eye never last "
    "let thought city tree cross farm hard start might story saw far sea draw left late run while press close "
    "night real life few north open seem together next white children begin got walk example ease paper group "
    "always music those both mark often letter until mile river car feet care second book carry took science "
    "eat room friend began idea fish mountain stop once base hear horse cut sure watch color face wood main"
).split()

CONTRACTIONS = [
    "can't", "won't", "don't", "isn't", "aren't", "wasn't", "weren't", "I'm", "you're", "we're", "they're",
    "it's", "he's", "she's", "I've", "we've", "I'll", "you'll", "I'd", "you'd", "shouldn't've", "y'all",
    "ma'am", "o'clock", "'cause", "'em", "rock'n'roll", "John's", "dogs'", "James's", "children's",
]
URLS = [
    "https://www.example.com", "http://t.co/AbC123", "https://github.com/moeru-ai/ortts/pull/27",
    "www.google.com/search?q=rust", "example.org", "sub.domain.co.uk/path/to/page.html",
    "https://en.wikipedia.org/wiki/Python_(programming_language)", "http://x.io/a(b)c", "ftp.example.com",
    "HTTPS://EXAMPLE.COM/UPPER", "http://localhost:8080/api", "bit.ly/3xYz", "foo.museum",
]
EMAILS = ["john@example.com", "jane.doe+news@mail.example.org", "a_b@c-d.ef", "x@y.co", "First.Last@Company.COM"]
EMOTICONS = [":)", ":-)", ":(", ":-(", ";)", ";-)", ":D", ":-D", ":P", ":p", "XD", "<3", "</3", ":'(", "=)",
             "8)", ":/", ":|", ">:(", "(:", "D:", ":]", ":[", ":}", ":{", ":@", ":*", ":o", "o_O"]
HASHTAGS = ["#rust", "#NLP", "#tts_rocks", "#100DaysOfCode", "#it's", "#a-b", "##double", "#_x_", "#1"]
HANDLES = ["@user", "@User_Name", "@a", "@moeru_ai", "@verylonghandle_1234567", "@1", "@_"]
PHONES = ["555-123-4567", "(555) 123-4567", "+1 555 123 4567", "1-800-555-0199", "555.123.4567",
          "5551234567", "123-4567", "+0 123 456 7890", "(555)123-4567", "1 (555) 123 4567"]
NUMBERS = ["1", "42", "2024", "1999", "3.14", "-7", "+3", "1,000", "1,000,000", "1/2", "3/4", "12:30",
           "2020-01-01", "1st", "2nd", "3rd", "4th", "21st", "100th", "0.5", "$10", "50%", "1e5", "007",
           "\u0661\u0662\u0663\u0664", "\u00b2", "\u2460"]
PUNCT = [".", ",", "!", "?", ";", ":", "...", "!!!", "?!", "--", "-", "(", ")", "\"", "'", "\u201c", "\u201d",
         "\u2018", "\u2019", "\u2026", "\u2014", "\u2013", "[", "]", "&", "*", "/", "\\", "......", ". . ."]
ENTITIES = ["&amp;", "&lt;", "&gt;", "&quot;", "&#39;", "&#x27;", "&nbsp;", "&copy;", "&eacute;", "&#128512;",
            "&#150;", "&bogus;", "&#xD83D;", "&#1;", "&#x1F600;"]
EMOJI = ["\U0001F600", "\U0001F44D\U0001F3FB", "\U0001F469\u200d\U0001F4BB", "\U0001F1EC\U0001F1E7",
         "\u2764\ufe0f", "\U0001F525", "\u2728", "\U0001F64F\U0001F3FE"]
UNICODE_WORDS = ["café", "naïve", "über", "Straße", "你好", "こんにちは",
                 "\u041f\u0440\u0438\u0432\u0435\u0442", "\u0130stanbul", "\u03a3\u039f\u03a6\u0399\u0391", "r\u00e9sum\u00e9"]


def vocab_from_model():
    """Words the tagger knows: tagdict keys and `i word` features."""
    words = set(TAGGER.tagdict)
    for feat in TAGGER.model.weights:
        if feat.startswith("i word "):
            words.add(feat[len("i word "):])
    return sorted(words)


def case_variant(rng, w):
    r = rng.random()
    if r < 0.1:
        return w.upper()
    if r < 0.25:
        return w.capitalize()
    if r < 0.28:
        return "".join(c.upper() if rng.random() < 0.5 else c for c in w)
    return w


def gen_sentence(rng, vocab):
    n = rng.randint(1, 18)
    parts = []
    for _ in range(n):
        r = rng.random()
        if r < 0.45:
            parts.append(case_variant(rng, rng.choice(vocab)))
        elif r < 0.6:
            parts.append(case_variant(rng, rng.choice(WORDS_SIMPLE)))
        elif r < 0.65:
            parts.append(rng.choice(CONTRACTIONS))
        elif r < 0.68:
            parts.append(rng.choice(URLS))
        elif r < 0.70:
            parts.append(rng.choice(EMAILS))
        elif r < 0.73:
            parts.append(rng.choice(EMOTICONS))
        elif r < 0.75:
            parts.append(rng.choice(HASHTAGS))
        elif r < 0.77:
            parts.append(rng.choice(HANDLES))
        elif r < 0.79:
            parts.append(rng.choice(PHONES))
        elif r < 0.85:
            parts.append(rng.choice(NUMBERS))
        elif r < 0.93:
            parts.append(rng.choice(PUNCT))
        elif r < 0.95:
            parts.append(rng.choice(ENTITIES))
        elif r < 0.97:
            parts.append(rng.choice(EMOJI))
        else:
            parts.append(rng.choice(UNICODE_WORDS))
    out = []
    for i, p in enumerate(parts):
        if i:
            r = rng.random()
            out.append("" if r < 0.12 else ("  " if r < 0.15 else ("\n" if r < 0.17 else " ")))
        out.append(p)
    s = "".join(out)
    if rng.random() < 0.5:
        s += rng.choice([".", "!", "?", "...", "!!", ""])
    return s


# Characters for random noise strings: ASCII, Latin-1, and code points from
# many scripts and categories, to stress the character classes.
NOISE_POOL = (
    string.ascii_letters + string.digits + string.punctuation + " \t\n" * 3
    + "".join(chr(c) for c in range(0xA0, 0x250))
    + "\u0300\u0301\u0345\u0370\u0391\u03a3\u03c2\u03c3\u0400\u0430\u0531\u05d0\u0620\u0660\u0669\u06f0"
    + "\u0900\u0966\u0e01\u0e50\u1000\u1040\u10a0\u1100\u1e00\u1e9e\u1f00\u1f88\u2000\u200b\u200c\u200d"
    + "\u2028\u2029\u202f\u205f\u2070\u00b9\u2150\u2160\u2460\u24b6\u2c00\u3000\u3007\u3041\u30a1\u4e00"
    + "\uac00\ua640\ua8d0\ufb01\ufe0f\uff10\uff21\uff41\U00010400\U00010428\U0001d400\U0001d7ce\U0001f3fb"
    + "\U0001f1e6\U0001f1ff\U0001f600\U000e0067\U0001e900\U0001e922\U00010d50\U00016e40\U0001e030\U00011f50"
    + "\u212a\u2126\u0130\u0131\u017f\u1e9b\u1fbe\u01c5\u2c2f\u2c5f\ua7c0\ua7d0\U00010570\U00010597"
)


def gen_noise(rng):
    n = rng.randint(1, 40)
    return "".join(rng.choice(NOISE_POOL) for _ in range(n))


def gen_inputs():
    rng = random.Random(20261102)
    vocab = vocab_from_model()
    inputs = list(HAND)
    # Every emoticon / url / email / ... alone and inside a sentence.
    for pool in (CONTRACTIONS, URLS, EMAILS, EMOTICONS, HASHTAGS, HANDLES, PHONES, NUMBERS, PUNCT, ENTITIES,
                 EMOJI, UNICODE_WORDS):
        for item in pool:
            inputs.append(item)
            inputs.append(f"Look at {item} here")
            inputs.append(f"x{item}y")
    for _ in range(5000):
        inputs.append(gen_sentence(rng, vocab))
    for _ in range(1500):
        inputs.append(gen_noise(rng))
    # Entities and numeric references with random bodies.
    for _ in range(300):
        body = "".join(rng.choice("#xX0123456789abcdefABCDEF_+- \u0661\u0662" ) for _ in range(rng.randint(1, 8)))
        inputs.append(f"a&{body};b")
    # Phone-like and digit runs.
    for _ in range(300):
        inputs.append("".join(rng.choice("0123456789 -.()*+") for _ in range(rng.randint(3, 25))))
    seen = set()
    out = []
    for s in inputs:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def tokenize_record(text):
    tokens = TOKENIZER.tokenize(text)
    tags = [t for _, t in pos_tag(tokens)]
    rec = {"text": text}
    toks = []
    bad = False
    for t in tokens:
        t2, b = fix_surrogates(t)
        bad |= b
        toks.append(t2)
    rec["tokens"] = toks
    rec["tags"] = tags
    if bad:
        rec["surrogate"] = True
    return rec


OPTION_SETS = [
    {"preserve_case": False},
    {"reduce_len": True},
    {"strip_handles": True},
    {"match_phone_numbers": False},
    {"preserve_case": False, "reduce_len": True, "strip_handles": True, "match_phone_numbers": False},
]


def gen_option_records(inputs):
    recs = []
    sample = inputs[: len(HAND)] + inputs[len(HAND) :: 20]
    for opts in OPTION_SETS:
        tk = TweetTokenizer(**opts)
        for text in sample:
            toks = []
            bad = False
            for t in tk.tokenize(text):
                t2, b = fix_surrogates(t)
                bad |= b
                toks.append(t2)
            rec = {"opts": opts, "text": text, "tokens": toks}
            if bad:
                rec["surrogate"] = True
            recs.append(rec)
    return recs


def gen_pos_tag_records():
    rng = random.Random(7)
    vocab = vocab_from_model()
    lists = [
        [],
        [""],
        ["", "", ""],
        ["-"],
        ["-x", "x-", "a-b", "--"],
        ["1999", "199", "19999", "\u0661\u0662\u0663\u0664", "\u00b2\u00b2\u00b2\u00b2", "1a", "a1", "\u2460"],
        ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"],
        ["read", "complex", "refuse", "record", "present", "lead"],
        ["\u03a3\u039f\u03a6\u0399\u0391\u03a3", "\u0130", "A", "I", "a"],
        ["-START-", "-END-", "!YEAR", "!DIGITS", "!HYPHEN"],
    ]
    for _ in range(600):
        n = rng.randint(1, 12)
        lists.append([case_variant(rng, rng.choice(vocab)) for _ in range(n)])
    recs = []
    for toks in lists:
        for use_tagdict in (True, False):
            tags = [t for _, t in TAGGER.tag(toks, use_tagdict=use_tagdict)]
            recs.append({"tokens": toks, "use_tagdict": use_tagdict, "tags": tags})
    # pos_tag itself on the hand lists
    for toks in lists[:10]:
        assert [t for _, t in pos_tag(toks)] == [t for _, t in TAGGER.tag(toks)]
    return recs


def gen_corpus_records():
    import gpt_sovits_env  # noqa: F401  (changes the working directory)

    with contextlib.redirect_stdout(sys.stderr):
        import torch
        from text import english
        from TTS_infer_pack.TextPreprocessor import TextPreprocessor

        class NoBertPreprocessor(TextPreprocessor):
            def get_bert_inf(self, phones, word2ph, norm_text, language):
                return torch.zeros((1024, len(phones)), dtype=torch.float32)

        tp = NoBertPreprocessor(None, None, None)

    captured = []
    orig = english.word_tokenize

    def recorder(text):
        captured.append(text)
        return orig(text)

    english.word_tokenize = recorder
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    for item in corpus:
        text = item["text"]
        langs = {item["lang"], "en", "auto"} if item["lang"] != "yue" else {"yue", "auto_yue"}
        for lang in sorted(langs):
            with contextlib.redirect_stdout(sys.stderr):
                try:
                    tp.get_phones_and_bert(text, lang, "v2")
                except Exception as e:  # noqa: BLE001
                    print("error", lang, repr(text), repr(e), file=sys.stderr)
                try:
                    tp.preprocess(text, lang, "cut0", "v2")
                except Exception as e:  # noqa: BLE001
                    print("error", lang, repr(text), repr(e), file=sys.stderr)
    english.word_tokenize = orig
    seen = set()
    recs = []
    for text in captured:
        if text in seen:
            continue
        seen.add(text)
        recs.append(tokenize_record(text))
    return recs


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dump_json("regex_sources.json", regex_sources())
    dump_json("char_classes.json", char_classes())
    dump_json("python_lower.json", python_lower())
    inputs = gen_inputs()
    dump_jsonl("tokenize.jsonl", [tokenize_record(t) for t in inputs])
    dump_jsonl("tokenize_options.jsonl", gen_option_records(inputs))
    dump_jsonl("pos_tag.jsonl", gen_pos_tag_records())
    dump_jsonl("corpus.jsonl", gen_corpus_records())


if __name__ == "__main__":
    main()

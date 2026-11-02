"""Generate golden fixtures for the `gpt_sovits_text::japanese` port.

    pixi run --manifest-path crates/ports/gpt_sovits_text/pixi.toml fixtures japanese

Writes `crates/ports/gpt_sovits_text/tests/fixtures/japanese/*.jsonl`.

User dictionary: upstream `text/japanese.py` compiles `ja_userdic/userdict.csv`
into `ja_userdic/user.dict` next to its own source on first import and then
loads it with `pyopenjtalk.update_global_jtalk_with_user_dict`. To leave the
upstream checkout untouched, this script compiles the same CSV into a temporary
directory, loads it into the global pyopenjtalk instance, and makes the
import-time `mecab_dict_index` call fail, which upstream swallows (`except
Exception: pass`). The global instance then has the user dictionary loaded,
exactly as after a normal first import.

Each record has every layer of the pipeline so mismatches can be localized:
`njd` (`pyopenjtalk.run_frontend`, one string per node in jpreprocess layout),
`labels` (`pyopenjtalk.make_label`), `g2p_plain` (`pyopenjtalk.g2p`),
`prosody` (`pyopenjtalk_g2p_prosody`), and the japanese.py outputs.
"""

import contextlib
import hashlib
import json
import random
import re
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import gpt_sovits_env  # noqa: E402

REPO = gpt_sovits_env.CRATE_DIR
CORPUS = gpt_sovits_env.CORPUS_DIR / "gpt_sovits_ja.json"
OUT = gpt_sovits_env.FIXTURES / "japanese"
USERDIC = gpt_sovits_env.DATA / "ja_userdic/userdict.csv"

gpt_sovits_env.setup()

import pyopenjtalk  # noqa: E402

_tmp = tempfile.TemporaryDirectory()
_user_dict = str(Path(_tmp.name) / "user.dict")
with contextlib.redirect_stdout(sys.stderr):
    pyopenjtalk.mecab_dict_index(str(USERDIC), _user_dict)
pyopenjtalk.update_global_jtalk_with_user_dict(_user_dict)


def _no_compile(*_args, **_kwargs):
    raise RuntimeError("gen_fixtures.py: user dictionary is already loaded")


pyopenjtalk.mecab_dict_index = _no_compile

from text import japanese as ja  # noqa: E402

UPSTREAM_CSV = gpt_sovits_env.UPSTREAM / "GPT_SoVITS/text/ja_userdic/userdict.csv"
assert hashlib.md5(UPSTREAM_CSV.read_bytes()).hexdigest() == hashlib.md5(USERDIC.read_bytes()).hexdigest(), (
    "data/ja_userdic/userdict.csv differs from upstream"
)


def njd_str(f):
    return ",".join(
        str(x)
        for x in [
            f["string"],
            f["pos"],
            f["pos_group1"],
            f["pos_group2"],
            f["pos_group3"],
            f["ctype"],
            f["cform"],
            f["orig"],
            f["read"],
            f["pron"],
            f"{f['acc']}/{f['mora_size']}",
            f["chain_rule"],
            f["chain_flag"],
        ]
    )


def try_(fn, *args):
    try:
        return fn(*args)
    except Exception as e:  # noqa: BLE001
        return {"error": type(e).__name__}


def layers(text, with_labels, with_njd=True):
    njd = pyopenjtalk.run_frontend(text)
    rec = {"text": text}
    if with_njd:
        rec["njd"] = [njd_str(f) for f in njd]
    if with_labels:
        rec["labels"] = pyopenjtalk.make_label(njd)
    rec["g2p_plain"] = pyopenjtalk.g2p(text)
    rec["prosody"] = try_(ja.pyopenjtalk_g2p_prosody, text)
    return rec


def segments(norm):
    """The strings preprocess_jap hands to pyopenjtalk."""
    text = ja.symbols_to_japanese(norm).lower()
    return [s for s in re.split(ja._japanese_marks, text) if re.match(ja._japanese_characters, s)]


def frontend_record(text, with_labels, with_njd=True):
    norm = ja.text_normalize(text)
    return {
        "text": text,
        "norm": norm,
        "g2p": try_(ja.g2p, norm, True),
        "g2p_noprosody": try_(ja.g2p, norm, False),
        "segs": [layers(s, with_labels, with_njd) for s in segments(norm)],
    }


def write_jsonl(name, records):
    path = OUT / name
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"{path.relative_to(REPO)}: {len(records)} records, {path.stat().st_size} bytes", file=sys.stderr)


def fullwidth_to_ascii(s):
    return "".join(chr(ord(c) - 0xFEE0) if 0xFF01 <= ord(c) <= 0xFF5E else c for c in s)


KATAKANA_PRON = re.compile(r"^[ァ-ヴー’]+$")


def userdict_inputs(rng):
    rows = [line.split(",") for line in USERDIC.read_text(encoding="utf-8").splitlines() if line]
    surfaces = sorted({r[0] for r in rows})
    # Rows whose pronunciation is not plain katakana (jpreprocess cannot store
    # some of them as is), all of them.
    odd = sorted({r[0] for r in rows if len(r) != 15 or not KATAKANA_PRON.match(r[12])})
    picked = rng.sample(surfaces, 900)
    out = []
    for s in sorted(set(picked) | set(odd)):
        out.append(fullwidth_to_ascii(s))
    # Some in fullwidth form and inside Japanese context.
    for s in rng.sample(surfaces, 100):
        out.append(s)
    for s in rng.sample(surfaces, 100):
        out.append(f"これは{fullwidth_to_ascii(s)}です")
    return out


RANDOM_ALPHABET = list(
    "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをんがぎぐげござじずぜぞだぢづでど"
    "ばびぶべぼぱぴぷぺぽぁぃぅぇぉっゃゅょゎゔー"
    "アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲンヴッャュョァィゥェォヵヶ"
    "日本語東京大学生先人今年月時分秒円上下中山川田本気電車会社行来見食飲話読書言思々"
    "0123456789０１２３４５６７８９abcxyzABCXYZａｂｃＡＢＣ"
    "!?…,.-！？。、，：；·「」『』（）・〜～％%＆&/\\@#$^*_=+<>|~`'\" 　\n\t"
)


def random_inputs(rng):
    out = []
    for _ in range(1000):
        n = rng.randint(0, 12)
        out.append("".join(rng.choice(RANDOM_ALPHABET) for _ in range(n)))
    return out


def phrase_inputs(rng, corpus_texts):
    # Recombine corpus fragments into new sentences.
    frags = sorted({f for t in corpus_texts for f in re.split(r"[、。！？!?\s]+", t) if 1 <= len(f) <= 20})
    seps = ["", "、", "。", "！", "？", "…", " ", "，", "!!", "?!", "……"]
    out = []
    for _ in range(600):
        k = rng.randint(1, 4)
        parts = [rng.choice(frags) for _ in range(k)]
        out.append("".join(p + rng.choice(seps) for p in parts))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20240315)
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    with contextlib.redirect_stdout(sys.stderr), contextlib.redirect_stderr(open("/dev/null", "w")):
        recs = []
        for item in corpus:
            r = frontend_record(item["text"], True)
            r["cat"] = item["cat"]
            recs.append(r)
        bulk_user = [layers(t, False) | {"g2p": try_(ja.g2p, t, True)} for t in userdict_inputs(rng)]
        # Without the NJD layer to keep the fixtures small.
        bulk_random = [frontend_record(t, False, False) for t in random_inputs(rng)]
        bulk_phrase = [
            frontend_record(t, False, False) for t in phrase_inputs(rng, [i["text"] for i in corpus])
        ]
        punct = [
            {"text": t, "norm": ja.text_normalize(t), "rcp": ja.replace_consecutive_punctuation(t),
             "stj": ja.symbols_to_japanese(t)}
            for t in random_inputs(rng)
        ]
        post = [{"ph": p, "out": ja.post_replace_ph(p)} for p in
                ["：", "；", "，", "。", "！", "？", "\n", "·", "、", "...", "..", ".", "a", "", "ー", " "]]
    write_jsonl("corpus.jsonl", recs)
    write_jsonl("bulk_userdict.jsonl", bulk_user)
    write_jsonl("bulk_random.jsonl", bulk_random)
    write_jsonl("bulk_phrase.jsonl", bulk_phrase)
    write_jsonl("text_normalize.jsonl", punct)
    write_jsonl("post_replace_ph.jsonl", post)


if __name__ == "__main__":
    main()

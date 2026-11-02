"""Write the Python-runtime data tables that the nltk port embeds.

    pixi run --manifest-path crates/ports/nltk/pixi.toml data

nltk itself ships no tables for the ported code, but the ported code calls
CPython builtins whose behaviour depends on CPython's Unicode database
(Python 3.11, Unicode 14.0.0) and on the stdlib `html.entities` module:

- `html_name2codepoint.json`: `html.entities.name2codepoint`, used by
  `nltk.tokenize.casual._replace_html_entities`.
- `python_str.json`: code point sets for `str.isdigit` (used by
  `PerceptronTagger.normalize`), `str.isspace` and the decimal digit blocks
  (both used by `int(text, base)` in `_replace_html_entities`), and the
  `str.lower` mapping with the Case_Ignorable / Cased properties of its
  Final_Sigma rule.
- `regex_sets.json`: for every character-class atom of the nltk.tokenize.casual
  regexes, the code points the `regex` module matches for it with the flags
  nltk compiles that regex with. The Rust port substitutes these sets so that
  it does not depend on the Unicode tables or case folding of the Rust regex
  engine.

The perceptron model files are copied verbatim from nltk_data
(`taggers/averaged_perceptron_tagger_eng/*.json`); this script only checks
that the committed copies match the installed ones.
"""

import hashlib
import html.entities
import json
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import DATA, NLTK_DATA  # noqa: E402  (also sets NLTK_DATA)


def write(path, text):
    # Fixed encoding and line endings, so the output is the same on Windows.
    path.write_text(text, encoding="utf-8", newline="\n")


def ranges(codepoints):
    out = []
    for c in codepoints:
        if out and out[-1][1] == c - 1:
            out[-1][1] = c
        else:
            out.append([c, c])
    return out


def strip_verbose(src):
    """Same as `nltk::tokenize::casual::strip_verbose` in Rust: drop
    whitespace and `#` comments outside sets (regex.VERBOSE)."""
    out = []
    i = 0
    n = len(src)
    while i < n:
        c = src[i]
        if c == "\\":
            out.append(src[i : i + 2])
            i += 2
        elif c == "[":
            j = set_end(src, i)
            out.append(src[i:j])
            i = j
        elif c == "#":
            j = src.find("\n", i)
            i = n if j < 0 else j + 1
        elif c.isspace():
            i += 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


def set_end(src, i):
    """Index just past the set starting at src[i] == '['."""
    j = i + 1
    if j < len(src) and src[j] == "^":
        j += 1
    if j < len(src) and src[j] == "]":
        j += 1
    while j < len(src):
        c = src[j]
        j += 1
        if c == "\\":
            j += 1
        elif c == "]":
            break
    return j


def atoms(src, ignorecase):
    """Same as the atom scanner in `nltk::tokenize::casual::compile`: the
    single-character constructs of a (non-verbose) pattern whose meaning
    depends on the Unicode database or on case folding."""
    out = []
    i = 0
    n = len(src)
    while i < n:
        c = src[i]
        if c == "\\":
            e = src[i + 1]
            if e in "wWdDsS":
                out.append(src[i : i + 2])
                i += 2
            elif e in "pP" and src[i + 2] == "{":
                j = src.index("}", i) + 1
                out.append(src[i:j])
                i = j
            elif e == "b":
                out.append("\\w")
                i += 2
            elif ignorecase and e in "Uux":
                k = {"U": 10, "u": 6, "x": 4}[e]
                out.append(src[i : i + k])
                i += k
            else:
                i += 2
        elif c == "[":
            j = set_end(src, i)
            out.append(src[i:j])
            i = j
        elif c == ".":
            out.append(c)
            i += 1
        elif ignorecase and (c.isalpha() or ord(c) > 127):
            out.append(c)
            i += 1
        else:
            i += 1
    return out


def patterns():
    """(name, source, flags) of every regex the port compiles."""
    import regex
    from nltk.tokenize import casual

    vi = regex.VERBOSE | regex.I
    return [
        ("WORD_RE", "(" + "|".join(casual.REGEXPS) + ")", vi),
        ("PHONE_WORD_RE", "(" + "|".join(casual.REGEXPS_PHONE) + ")", vi),
        ("EMOTICON_RE", casual.EMOTICONS, vi),
        ("HANG_RE", casual.HANG_RE.pattern, 0),
        ("ENT_RE", casual.ENT_RE.pattern, 0),
        ("HANDLES_RE", casual.HANDLES_RE.pattern, 0),
        ("reduce_lengthening", r"(.)\1{2,}", 0),
    ]


ALL_CHARS = None


def probe(atom, flags):
    """Code points (as ranges) that `atom` matches with `flags`."""
    import regex

    global ALL_CHARS
    if ALL_CHARS is None:
        ALL_CHARS = "".join(chr(c) for c in range(0x110000) if not 0xD800 <= c <= 0xDFFF)
    rx = regex.compile(atom, flags)
    hits = []
    for m in rx.finditer(ALL_CHARS):
        assert m.end() - m.start() == 1, atom
        hits.append(ord(m.group()))
    return ranges(hits)


def regex_sets():
    import regex

    tables = {"default": {}, "ignorecase": {}}
    for _, src, flags in patterns():
        ic = bool(flags & regex.I)
        if flags & regex.VERBOSE:
            src = strip_verbose(src)
        table = tables["ignorecase" if ic else "default"]
        for atom in atoms(src, ic):
            if atom not in table:
                table[atom] = probe(atom, flags)
    return {
        "regex_version": regex.__version__,
        "default": dict(sorted(tables["default"].items())),
        "ignorecase": dict(sorted(tables["ignorecase"].items())),
    }


def main():
    assert sys.version_info[:2] == (3, 11), sys.version
    assert unicodedata.unidata_version == "14.0.0", unicodedata.unidata_version

    write(DATA / "regex_sets.json", json.dumps(regex_sets(), separators=(",", ":")) + "\n")

    write(
        DATA / "html_name2codepoint.json",
        json.dumps(html.entities.name2codepoint, sort_keys=True, separators=(",", ":")) + "\n"
    )

    isdigit = [c for c in range(0x110000) if chr(c).isdigit()]
    isspace = [c for c in range(0x110000) if chr(c).isspace()]
    decimal = [c for c in range(0x110000) if unicodedata.decimal(chr(c), -1) >= 0]
    zeros = sorted({c - unicodedata.decimal(chr(c)) for c in decimal})
    # Every decimal digit belongs to a full 0..9 block that starts at a zero.
    assert sorted(z + i for z in zeros for i in range(10)) == decimal

    # str.lower: the full lowercase mapping of every code point it changes,
    # and the Case_Ignorable / Cased properties that the Final_Sigma rule
    # (handle_capital_sigma in unicodeobject.c) reads. CPython does not expose
    # them, so they are measured through lower() itself:
    #   (c + "Σ").lower() ends in "ς"  <=>  not CI(c) and Cased(c)
    #   ("AΣ" + c).lower() has "ς"      <=>  CI(c) or not Cased(c)
    # Cased is only read for characters that are not case-ignorable.
    lower = []
    case_ignorable = []
    cased = []
    for c in range(0x110000):
        if 0xD800 <= c <= 0xDFFF:
            continue
        ch = chr(c)
        if ch.lower() != ch:
            lower.append([c, ch.lower()])
        t1 = (ch + "Σ").lower().endswith("ς")
        t3 = "ς" in ("AΣ" + ch).lower()
        if t1:
            cased.append(c)
        elif t3 and not ("A" + ch + "Σ").lower().endswith("ς"):
            pass  # not CI, not cased
        elif t3:
            case_ignorable.append(c)
    write(
        DATA / "python_str.json",
        json.dumps(
            {
                "unidata_version": unicodedata.unidata_version,
                "isdigit": ranges(isdigit),
                "isspace": ranges(isspace),
                "decimal_zeros": zeros,
                "lower": lower,
                "case_ignorable": ranges(case_ignorable),
                "cased": ranges(cased),
            },
            separators=(",", ":"),
        )
        + "\n"
    )

    import nltk.data

    nltk.data.path[:] = [str(NLTK_DATA)]
    src = Path(nltk.data.find("taggers/averaged_perceptron_tagger_eng/"))
    for name in ("weights", "tagdict", "classes"):
        f = f"averaged_perceptron_tagger_eng.{name}.json"
        a = hashlib.sha256((src / f).read_bytes()).hexdigest()
        b = hashlib.sha256((DATA / f).read_bytes()).hexdigest()
        assert a == b, (f, a, b)
        print(f, a)


if __name__ == "__main__":
    main()

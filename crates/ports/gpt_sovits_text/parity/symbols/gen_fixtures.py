"""Golden fixtures for text/symbols.py, text/symbols2.py and text/__init__.py.

Run:

    pixi run --manifest-path crates/ports/gpt_sovits_text/pixi.toml fixtures symbols

Writes crates/ports/gpt_sovits_text/tests/fixtures/symbols/symbols.json.
"""

import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gpt_sovits_env  # noqa: E402

gpt_sovits_env.setup()

import text as text_init  # noqa: E402  (text/__init__.py)
from text import symbols as symbols_v1  # noqa: E402
from text import symbols2 as symbols_v2  # noqa: E402

OUT = gpt_sovits_env.FIXTURES / "symbols/symbols.json"
SEED = 20240604


def module_constants(mod, extra):
    out = {
        "punctuation": mod.punctuation,
        "pu_symbols": mod.pu_symbols,
        "pad": mod.pad,
        "c": mod.c,
        "v": mod.v,
        "v_without_tone": mod.v_without_tone,
        "ja_symbols": mod.ja_symbols,
        # `arpa` is a set; its iteration order depends on the hash seed.
        "arpa_sorted": sorted(mod.arpa),
        "symbols": mod.symbols,
    }
    for name in extra:
        value = getattr(mod, name)
        out[name] = sorted(value) if isinstance(value, set) else value
    return out


def run_sequence(cleaned_text, version):
    try:
        return {"ids": text_init.cleaned_text_to_sequence(cleaned_text, version)}
    except KeyError as e:
        return {"error": "KeyError", "key": e.args[0]}


def main():
    # `version=None` reads os.environ["version"]; pin it to the default.
    os.environ.pop("version", None)
    rng = random.Random(SEED)
    v1 = symbols_v1.symbols
    v2 = symbols_v2.symbols
    v2_only = [s for s in v2 if s not in set(v1)]
    unknown = ["", "zz", "a0", " ", "SP4", "ㄱㄱ", "Ya1x", "\n", "AH3", "@"]

    cases = []
    for version in ["v1", "v2", None, "v3", "v2Pro", ""]:
        table = v1 if version == "v1" else v2
        cases.append({"cleaned_text": [], "version": version})
        cases.append({"cleaned_text": list(table), "version": version})
        for _ in range(60):
            n = rng.randint(1, 40)
            cases.append({"cleaned_text": [rng.choice(table) for _ in range(n)], "version": version})
        for bad in unknown + rng.sample(v2_only, 5):
            seq = [rng.choice(table) for _ in range(rng.randint(0, 5))]
            seq.insert(rng.randint(0, len(seq)), bad)
            cases.append({"cleaned_text": seq, "version": version})
    for case in cases:
        case.update(run_sequence(case["cleaned_text"], case["version"]))

    data = {
        "symbols": module_constants(symbols_v1, []),
        "symbols2": module_constants(symbols_v2, ["ko_symbols", "yue_symbols"]),
        # Dict items in insertion order.
        "symbol_to_id_v1": list(text_init._symbol_to_id_v1.items()),
        "symbol_to_id_v2": list(text_init._symbol_to_id_v2.items()),
        "cleaned_text_to_sequence": cases,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
        f.write("\n")
    print(f"wrote {OUT} ({len(cases)} sequence cases, {len(v1)} v1 / {len(v2)} v2 symbols)")


if __name__ == "__main__":
    main()

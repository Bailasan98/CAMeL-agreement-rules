from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Tuple, Optional, List


# ----------------------------
# Hard-set project paths
# ----------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

CONLLU_IN = DATA_DIR / "e100.conllu"
MAGOLD_IN = DATA_DIR / "e100.magold"
CONLLU_OUT = DATA_DIR / "e100.SYNC.conllu"


# ----------------------------
# CoNLL-U helpers
# ----------------------------
def parse_feats(feats: str) -> Dict[str, str]:
    feats = (feats or "").strip()
    if feats in ("", "_"):
        return {}
    d = {}
    for part in feats.split("|"):
        if "=" in part:
            k, v = part.split("=", 1)
            d[k] = v
    return d


def format_feats(d: Dict[str, str]) -> str:
    if not d:
        return "_"
    # stable order helps diffing
    return "|".join(f"{k}={d[k]}" for k in sorted(d.keys()))


def parse_misc(misc: str) -> Dict[str, str]:
    misc = (misc or "").strip()
    if misc in ("", "_"):
        return {}
    d = {}
    for part in misc.split("|"):
        if "=" in part:
            k, v = part.split("=", 1)
            d[k] = v
    return d


def read_conllu_lines(path: Path) -> List[str]:
    return path.read_text(encoding="utf-8").splitlines()


# ----------------------------
# MAGOLD parsing (GLOBAL lookup)
# ----------------------------
# We only care about the analysis lines that start with "*"
# Example:
# *1.0000000 ... diac:ma$Akila ... gen:f ... num:p ... rat:i ...
MAGOLD_ANALYSIS_RE = re.compile(r"^\*")


def extract_field(line: str, key: str) -> Optional[str]:
    # finds patterns like "gen:f" or "num:p" or "diac:ma$Akila"
    m = re.search(rf"\b{re.escape(key)}:([^\s]+)", line)
    return m.group(1) if m else None


def build_magold_lookup(magold_path: Path) -> Dict[str, Tuple[str, str, str]]:
    """
    Returns dict:
      key (string) -> (gen, num, rat)
    Keys we store:
      - diac
      - bw token (before "/") if present
    Values we store:
      - gen / num / rat (functional), NOT form_gen / form_num
    """
    lookup: Dict[str, Tuple[str, str, str]] = {}

    for line in magold_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or not MAGOLD_ANALYSIS_RE.match(line):
            continue

        gen = extract_field(line, "gen")
        num = extract_field(line, "num")
        rat = extract_field(line, "rat")
        diac = extract_field(line, "diac")
        bw_full = extract_field(line, "bw")  # e.g. ma$Akil/NOUN+a/...
        bw_tok = None
        if bw_full:
            bw_tok = bw_full.split("/", 1)[0].lstrip("+")  # remove leading '+'

        # Keep only usable values
        if gen in (None, "na") or num in (None, "na") or rat in (None, "na"):
            continue

        val = (gen, num, rat)

        # store diac key
        if diac:
            lookup[diac] = val

        # store bw token key too
        if bw_tok:
            lookup[bw_tok] = val

    return lookup


# ----------------------------
# Sync logic
# ----------------------------
def choose_conllu_key(misc: Dict[str, str]) -> Optional[str]:
    # Best key first
    if "surface_plus_bw" in misc:
        return misc["surface_plus_bw"]
    if "surface_form_bw" in misc:
        return misc["surface_form_bw"]
    # sometimes people store "bw" under other keys; add here if needed
    return None


def sync_conllu_with_magold(conllu_lines: List[str], mag_lookup: Dict[str, Tuple[str, str, str]]):
    out: List[str] = []
    matched = 0
    updated = 0

    # For debugging a specific token
    debug_targets = {"m$Akl", "ma$Akila", "ma$Akil"}

    debug_hits = []

    for ln in conllu_lines:
        if not ln.strip() or ln.startswith("#"):
            out.append(ln)
            continue

        cols = ln.split("\t")
        if len(cols) < 10:
            out.append(ln)
            continue

        feats_str = cols[5]
        misc_str = cols[9]

        feats = parse_feats(feats_str)
        misc = parse_misc(misc_str)

        key = choose_conllu_key(misc)
        if key and key in debug_targets:
            debug_hits.append(f"[CONLLU] key={key} feats_before={feats_str} misc={misc_str}")

        if not key:
            out.append(ln)
            continue

        mag_val = mag_lookup.get(key)
        if not mag_val:
            out.append(ln)
            continue

        matched += 1
        new_gen, new_num, new_rat = mag_val

        # update only if changed (keeps counts meaningful)
        changed = False
        if feats.get("gen") != new_gen:
            feats["gen"] = new_gen
            changed = True
        if feats.get("num") != new_num:
            feats["num"] = new_num
            changed = True
        if feats.get("rat") != new_rat:
            feats["rat"] = new_rat
            changed = True

        if changed:
            updated += 1
            cols[5] = format_feats(feats)
            new_ln = "\t".join(cols)
            out.append(new_ln)

            if key in debug_targets:
                debug_hits.append(f"[UPDATED] key={key} feats_after={cols[5]}")
        else:
            out.append(ln)

    return out, matched, updated, debug_hits


def main() -> None:
    for p in (CONLLU_IN, MAGOLD_IN):
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")

    print(f"BASE_DIR   : {BASE_DIR}")
    print(f"CONLLU_IN  : {CONLLU_IN}")
    print(f"MAGOLD_IN  : {MAGOLD_IN}")
    print(f"CONLLU_OUT : {CONLLU_OUT}")

    conllu_lines = read_conllu_lines(CONLLU_IN)
    mag_lookup = build_magold_lookup(MAGOLD_IN)

    print(f"MAGOLD lookup size (keys): {len(mag_lookup)}")

    out_lines, matched, updated, debug_hits = sync_conllu_with_magold(conllu_lines, mag_lookup)

    CONLLU_OUT.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    print("Wrote:", CONLLU_OUT)
    print("Tokens matched to MAGOLD (by key):", matched)
    print("Tokens actually updated (gen/num/rat changed):", updated)

    # Debug output if we touched your example forms
    if debug_hits:
        print("\n--- DEBUG (target tokens) ---")
        for x in debug_hits:
            print(x)


if __name__ == "__main__":
    main()

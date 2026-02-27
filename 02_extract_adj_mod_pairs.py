
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# ---------- Paths ----------
BASE_DIR = Path("~/Desktop/ArabicAgreementSync").expanduser()
DATA_DIR = BASE_DIR / "data"

CONLLU_IN = DATA_DIR / "e100.SYNC.conllu"   # <- input
CSV_OUT   = DATA_DIR / "adj_mod_pairs.csv"  # <- output


# ---------- Helpers ----------
def parse_misc(misc: str) -> Dict[str, str]:
    misc = (misc or "").strip()
    if misc in ("", "_"):
        return {}
    out: Dict[str, str] = {}
    for part in misc.split("|"):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k] = v
    return out

def parse_feats(feats: str) -> Dict[str, str]:
    feats = (feats or "").strip()
    if feats in ("", "_"):
        return {}
    out: Dict[str, str] = {}
    for part in feats.split("|"):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k] = v
    return out

def is_adj_token(upos: str, xpos: str, misc: Dict[str, str]) -> bool:
    # Your data: UPOS/XPOS often "NOM", so we rely on MISC (bw/kulick/mada)
    bw = misc.get("bw", "")
    mada = misc.get("mada", "")
    kulick = misc.get("kulick", "")

    if bw.startswith("ADJ"):
        return True
    if mada == "adj":
        return True
    if kulick == "JJ":
        return True

    # fallback: sometimes ADJ can appear in bw somewhere
    if "ADJ" in bw.split("+"):
        return True

    return False

def is_noun_like(misc: Dict[str, str], upos: str, xpos: str) -> bool:
    bw = misc.get("bw", "")
    mada = misc.get("mada", "")
    if bw.startswith("NOUN"):
        return True
    if mada == "noun":
        return True
    # If your scheme uses NOM for nouns too, keep this permissive:
    # (But still prefer bw/mada.)
    return False

def read_conllu_sentences(path: Path) -> List[List[str]]:
    sentences: List[List[str]] = []
    cur: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() == "":
            if cur:
                sentences.append(cur)
                cur = []
            continue
        cur.append(line)
    if cur:
        sentences.append(cur)
    return sentences

def parse_token_line(line: str) -> Optional[Dict]:
    if line.startswith("#"):
        return None
    cols = line.split("\t")
    if len(cols) < 10:
        return None

    tok_id = cols[0]
    # skip multiword tokens / empty nodes
    if "-" in tok_id or "." in tok_id:
        return None

    return {
        "id": int(tok_id),
        "form": cols[1],
        "lemma": cols[2],
        "upos": cols[3],
        "xpos": cols[4],
        "feats_raw": cols[5],
        "head": int(cols[6]) if cols[6].isdigit() else 0,
        "deprel": cols[7],
        "deps": cols[8],
        "misc_raw": cols[9],
    }


def main() -> None:
    if not CONLLU_IN.exists():
        raise FileNotFoundError(f"Missing input: {CONLLU_IN}")

    sents = read_conllu_sentences(CONLLU_IN)

    rows: List[Dict[str, str]] = []

    for sent_i, sent_lines in enumerate(sents, start=1):
        tokens: Dict[int, Dict] = {}
        for ln in sent_lines:
            t = parse_token_line(ln)
            if not t:
                continue
            t["feats"] = parse_feats(t["feats_raw"])
            t["misc"] = parse_misc(t["misc_raw"])
            tokens[t["id"]] = t

        for dep_id, dep in tokens.items():
            if dep["deprel"] != "MOD":
                continue

            if not is_adj_token(dep["upos"], dep["xpos"], dep["misc"]):
                # Not an adjective modifier (e.g., NOUN->NOUN, NUM, etc.)
                continue

            head_id = dep["head"]
            head = tokens.get(head_id)
            if not head:
                continue

            # optionally require noun head:
            if not is_noun_like(head["misc"], head["upos"], head["xpos"]):
                continue

            rows.append({
                "sent_idx": str(sent_i),
                "adj_id": str(dep_id),
                "adj_form": dep["form"],
                "adj_lemma": dep["lemma"],
                "adj_feats": dep["feats_raw"],
                "adj_misc": dep["misc_raw"],
                "head_id": str(head_id),
                "head_form": head["form"],
                "head_lemma": head["lemma"],
                "head_feats": head["feats_raw"],
                "head_misc": head["misc_raw"],
                "deprel": dep["deprel"],
            })

    # write CSV
    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else [
        "sent_idx","adj_id","adj_form","adj_lemma","adj_feats","adj_misc",
        "head_id","head_form","head_lemma","head_feats","head_misc","deprel"
    ]
    with CSV_OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Read sentences: {len(sents)}")
    print(f"Extracted ADJ→NOUN MOD pairs: {len(rows)}")
    print(f"Wrote: {CSV_OUT}")


if __name__ == "__main__":
    main()

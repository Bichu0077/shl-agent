"""Generate a larger ground-truth JSONL file from `data/catalog.json`.

Heuristics:
- For items with test_types containing 'K' (knowledge), generate developer/data role queries.
- For 'P' (personality), generate manager/sales role queries.
- For 'A' (ability), generate graduate/analyst queries.

This creates synthetic query->ground_truth pairs for evaluation.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List


def load_catalog(path: str) -> List[dict]:
    with open(path) as f:
        return json.load(f)


def pick_role_for_item(item: dict) -> str:
    name = item.get("name", "").lower()
    types = item.get("test_types", [])
    # heuristics
    if "k" in [t.lower() for t in types]:
        if any(tok in name for tok in ["java", "python", "sql", "javascript", ".net"]):
            return "software developer"
        if any(tok in name for tok in ["data", "machine"]):
            return "data analyst"
        return "engineer"
    if "p" in [t.lower() for t in types]:
        return "sales manager"
    if "a" in [t.lower() for t in types]:
        return "graduate analyst"
    return "hiring manager"


def make_query(item: dict) -> str:
    role = pick_role_for_item(item)
    levels = item.get("job_levels", [])
    level = random.choice(levels) if levels else random.choice(["Junior", "Mid-level", "Senior"])
    return f"Hiring a {level.lower()} {role} — what assessments should I use?"


def generate(catalog_path: str, out_path: str, n: int):
    catalog = load_catalog(catalog_path)
    entries = []
    candidates = [c for c in catalog if c.get("name")]
    if not candidates:
        raise RuntimeError("Catalog appears empty or malformed")

    for _ in range(n):
        item = random.choice(candidates)
        q = make_query(item)
        # include the sampled item as the ground truth; also try to include a related item if available
        gt = [item["name"]]
        # try to add one more related item of same test_type
        same_type = [c for c in candidates if set(c.get("test_types", [])) & set(item.get("test_types", [])) and c["name"] != item["name"]]
        if same_type:
            gt.append(random.choice(same_type)["name"])

        entries.append({"query": q, "ground_truth": gt})

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default="data/catalog.json")
    ap.add_argument("--out", default="eval/ground_truth_large.jsonl")
    ap.add_argument("--n", type=int, default=150)
    args = ap.parse_args()
    random.seed(42)
    generate(args.catalog, args.out, args.n)
    print(f"Wrote {args.n} entries to {args.out}")

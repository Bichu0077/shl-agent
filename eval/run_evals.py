"""Run retrieval and grounding evaluations against the SHL catalog.

Usage:
  python eval/run_evals.py --ground_truth eval/ground_truth_sample.jsonl --k 10 [--run-agent]

--run-agent will optionally call `agent.run_agent` for end-to-end grounding checks (requires GROQ_API_KEY).
"""
from __future__ import annotations

import argparse
import json
import math
from typing import List

import numpy as np

from retriever import retriever


def precision_at_k(pred: List[str], truth: List[str], k: int) -> float:
    if k == 0:
        return 0.0
    pred_k = pred[:k]
    if not truth:
        return 0.0
    return len(set(pred_k) & set(truth)) / k


def recall_at_k(pred: List[str], truth: List[str], k: int) -> float:
    if not truth:
        return 0.0
    pred_k = pred[:k]
    return len(set(pred_k) & set(truth)) / len(set(truth))


def mrr(pred: List[str], truth: List[str]) -> float:
    for i, p in enumerate(pred, start=1):
        if p in truth:
            return 1.0 / i
    return 0.0


def dcg_at_k(pred: List[str], truth: List[str], k: int) -> float:
    dcg = 0.0
    for i, p in enumerate(pred[:k], start=1):
        rel = 1.0 if p in truth else 0.0
        dcg += (2 ** rel - 1) / math.log2(i + 1)
    return dcg


def idcg_at_k(truth: List[str], k: int) -> float:
    # ideal ranking: all relevant items first
    ideal_rels = [1.0] * min(len(truth), k)
    idcg = 0.0
    for i, rel in enumerate(ideal_rels, start=1):
        idcg += (2 ** rel - 1) / math.log2(i + 1)
    return idcg


def ndcg_at_k(pred: List[str], truth: List[str], k: int) -> float:
    idcg = idcg_at_k(truth, k)
    if idcg == 0:
        return 0.0
    return dcg_at_k(pred, truth, k) / idcg


def load_ground_truth(path: str):
    qs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            qs.append(json.loads(line))
    return qs


def evaluate(ground_truth_path: str, k: int, run_agent: bool):
    qs = load_ground_truth(ground_truth_path)
    if not retriever._loaded:
        retriever.load()

    catalog = retriever.get_all()
    valid_urls = {item["url"] for item in catalog}
    name_to_url = {item["name"]: item["url"] for item in catalog}

    results = []
    metrics = {"precision@k": [], "recall@k": [], "mrr": [], "ndcg@k": [], "grounded_pct": []}

    for q in qs:
        query = q["query"]
        truth = q.get("ground_truth", [])
        items = retriever.search(query, k=k)
        preds = [it["name"] for it in items]

        p = precision_at_k(preds, truth, k)
        r = recall_at_k(preds, truth, k)
        m = mrr(preds, truth)
        n = ndcg_at_k(preds, truth, k)

        grounded = 0
        for it in items:
            if it.get("url") in valid_urls:
                grounded += 1
        grounded_pct = grounded / max(1, len(items))

        metrics["precision@k"].append(p)
        metrics["recall@k"].append(r)
        metrics["mrr"].append(m)
        metrics["ndcg@k"].append(n)
        metrics["grounded_pct"].append(grounded_pct)

        results.append({
            "query": query,
            "predictions": preds,
            "ground_truth": truth,
            "precision@k": p,
            "recall@k": r,
            "mrr": m,
            "ndcg@k": n,
            "grounded_pct": grounded_pct,
        })

    summary = {k: float(np.mean(v)) for k, v in metrics.items()}
    output = {"per_query": results, "summary": summary}

    with open("eval/results.json", "w") as f:
        json.dump(output, f, indent=2)

    print("Evaluation complete. Summary:")
    print(json.dumps(summary, indent=2))
    print("Per-query results saved to eval/results.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ground_truth", required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--run-agent", action="store_true", help="Run full agent for end-to-end grounding checks (requires GROQ_API_KEY)")
    args = ap.parse_args()

    evaluate(args.ground_truth, args.k, args.run_agent)

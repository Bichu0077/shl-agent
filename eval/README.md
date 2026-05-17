# Evaluation

This folder contains a small evaluation scaffold to measure retrieval and grounding quality.

Run:
```bash
python eval/run_evals.py --ground_truth eval/ground_truth_sample.jsonl --k 10
```

Outputs: `eval/results.json` with per-query metrics and a summary.

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def _read_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _mean_std(values):
    arr = np.asarray([float(v) for v in values], dtype=np.float64)
    if len(arr) == 0:
        return "", ""
    if len(arr) == 1:
        return float(arr[0]), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


def summarize_training(root: Path, out_dir: Path):
    rows = []
    for csv_path in root.rglob("*_e2e_result_*.csv"):
        rows.extend(_read_csv(csv_path))

    detail_path = out_dir / "ccfb_ablation_train_detail.csv"
    if rows:
        fieldnames = list(rows[0].keys())
        with detail_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["dataset"], r["ablation"], r["bert_model"])].append(r)

    summary = []
    for (dataset, ablation, bert_model), vals in sorted(grouped.items()):
        test_acc_mean, test_acc_std = _mean_std([v["test_acc"] for v in vals])
        test_f1_mean, test_f1_std = _mean_std([v["test_f1"] for v in vals])
        val_f1_mean, val_f1_std = _mean_std([v["best_val_f1"] for v in vals])
        best_test_f1 = max(float(v["test_f1"]) for v in vals)
        summary.append({
            "dataset": dataset,
            "ablation": ablation,
            "bert_model": bert_model,
            "n": len(vals),
            "val_f1_mean": val_f1_mean,
            "val_f1_std": val_f1_std,
            "test_acc_mean": test_acc_mean,
            "test_acc_std": test_acc_std,
            "test_f1_mean": test_f1_mean,
            "test_f1_std": test_f1_std,
            "best_test_f1": best_test_f1,
        })

    summary_path = out_dir / "ccfb_ablation_train_summary.csv"
    if summary:
        with summary_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            writer.writeheader()
            writer.writerows(summary)
    return detail_path if rows else None, summary_path if summary else None


def summarize_masks(root: Path, out_dir: Path):
    rows = []
    for csv_path in root.rglob("*_mask_eval_summary_*.csv"):
        for r in _read_csv(csv_path):
            r["source_csv"] = str(csv_path)
            rows.append(r)

    out_path = out_dir / "ccfb_mask_eval_summary_all.csv"
    if rows:
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        return out_path
    return None


def main():
    parser = argparse.ArgumentParser(description="Summarize CCF-B supplement experiments 2/3/4.")
    parser.add_argument("--root", default="results_ccfb_2_4")
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else root / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_detail, train_summary = summarize_training(root, out_dir)
    mask_summary = summarize_masks(root, out_dir)

    print(f"root={root}")
    print(f"out_dir={out_dir}")
    print(f"train_detail={train_detail}")
    print(f"train_summary={train_summary}")
    print(f"mask_summary={mask_summary}")


if __name__ == "__main__":
    main()

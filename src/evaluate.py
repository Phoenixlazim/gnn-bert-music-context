"""
evaluate.py
===========
Aggregate repeated-seed MTT experiments from results/metrics.json and create
report-ready summary tables/plots.

Outputs
-------
results/evaluation/mtt_summary.csv
results/evaluation/mtt_seed_results.csv
results/evaluation/mtt_summary.md
results/evaluation/mtt_macro_f1.png
results/evaluation/mtt_auc_pr.png
results/evaluation/mtt_micro_f1.png
results/evaluation/mtt_auc_roc.png

Usage
-----
python src/evaluate.py
python src/evaluate.py --metrics results/metrics.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

MODEL_ORDER = ["B1", "B4", "BERT", "A3", "C1", "B2", "A5", "A4"]
MODEL_LABELS = {
    "B1": "B1 prior",
    "B4": "B4 PCA+MLP",
    "BERT": "BERT only",
    "A3": "A3 relation-GNN",
    "C1": "C1 shuffled-GNN",
    "B2": "B2 log-mel CNN",
    "A5": "A5 cross-attention",
    "A4": "A4 concat",
}
METRICS = ["macro_f1", "micro_f1", "auc_pr", "auc_roc"]

RUN_RE = re.compile(
    r"MTT_(BERT|A3|A4|A5|C1|B1|B2|B4)_s(\d+)$",
    re.IGNORECASE,
)


def parse_runs(metrics: dict) -> list[dict]:
    rows = []
    for key, rec in metrics.items():
        run_name = key.split("/")[-1]
        m = RUN_RE.match(run_name)
        if not m:
            continue
        model = m.group(1).upper()
        if model == "BERT":
            model = "BERT"
        seed = int(m.group(2))
        test = rec.get("test", {})
        if not all(k in test for k in METRICS):
            continue
        rows.append({
            "model": model,
            "seed": seed,
            **{k: float(test[k]) for k in METRICS},
        })
    return rows


def mean_sd(vals: list[float]) -> tuple[float, float]:
    a = np.asarray(vals, dtype=float)
    mean = float(a.mean())
    sd = float(a.std(ddof=1)) if len(a) > 1 else 0.0
    return mean, sd


def aggregate(rows: list[dict]) -> list[dict]:
    out = []
    for model in MODEL_ORDER:
        rr = [r for r in rows if r["model"] == model]
        if not rr:
            continue
        rec = {
            "model": model,
            "label": MODEL_LABELS[model],
            "n_seeds": len(rr),
        }
        for metric in METRICS:
            mean, sd = mean_sd([r[metric] for r in rr])
            rec[f"{metric}_mean"] = mean
            rec[f"{metric}_sd"] = sd
        out.append(rec)
    return out


def fmt(mean: float, sd: float, n: int) -> str:
    return f"{mean:.4f} ± {sd:.4f}" if n > 1 else f"{mean:.4f}"


def write_csvs(out_dir: Path, seed_rows: list[dict], summary: list[dict]):
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "mtt_seed_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "seed", *METRICS])
        w.writeheader()
        w.writerows(sorted(seed_rows, key=lambda r: (MODEL_ORDER.index(r["model"]), r["seed"])))

    fields = ["model", "label", "n_seeds"]
    for metric in METRICS:
        fields += [f"{metric}_mean", f"{metric}_sd"]
    with open(out_dir / "mtt_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(summary)


def write_markdown(out_dir: Path, summary: list[dict]):
    lines = [
        "| Model | Seeds | Macro-F1 | Micro-F1 | AUC-PR | AUC-ROC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in summary:
        n = r["n_seeds"]
        vals = [
            fmt(r[f"{m}_mean"], r[f"{m}_sd"], n)
            for m in METRICS
        ]
        lines.append(
            f"| {r['label']} | {n} | {vals[0]} | {vals[1]} | {vals[2]} | {vals[3]} |"
        )

    with open(out_dir / "mtt_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines))


def plot_metric(out_dir: Path, summary: list[dict], metric: str, title: str, ylabel: str):
    labels = [r["label"] for r in summary]
    means = [r[f"{metric}_mean"] for r in summary]
    sds = [r[f"{metric}_sd"] if r["n_seeds"] > 1 else 0.0 for r in summary]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=sds, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / f"mtt_{metric}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="results/metrics.json")
    ap.add_argument("--out-dir", default="results/evaluation")
    args = ap.parse_args()

    mpath = Path(args.metrics)
    if not mpath.exists():
        raise SystemExit(f"[fatal] missing {mpath}")

    metrics = json.load(open(mpath, encoding="utf-8"))
    rows = parse_runs(metrics)
    if not rows:
        raise SystemExit("[fatal] no MTT_*_s<seed> runs found in metrics.json")

    summary = aggregate(rows)
    out_dir = Path(args.out_dir)
    write_csvs(out_dir, rows, summary)
    write_markdown(out_dir, summary)

    plot_metric(out_dir, summary, "macro_f1", "MTT model comparison — Macro-F1", "Macro-F1")
    plot_metric(out_dir, summary, "micro_f1", "MTT model comparison — Micro-F1", "Micro-F1")
    plot_metric(out_dir, summary, "auc_pr", "MTT model comparison — AUC-PR", "AUC-PR")
    plot_metric(out_dir, summary, "auc_roc", "MTT model comparison — AUC-ROC", "AUC-ROC")

    print(f"\n[done] {len(rows)} seed-level runs")
    print(f"[save] {out_dir / 'mtt_summary.csv'}")
    print(f"[save] {out_dir / 'mtt_summary.md'}")
    print(f"[save] 4 comparison plots in {out_dir}")


if __name__ == "__main__":
    main()

"""
final_analysis.py
=================
Final report-analysis pass for the CSE425 GNN-BERT music-context project.

Generates:
    results/evaluation/training_curves.png
    results/evaluation/tsne_fusion.png
    results/evaluation/case_studies.json
    results/evaluation/case_studies.png

Also generates:
    results/evaluation/bert_examples.json
    results/evaluation/gtzan_ablation.csv

Run from the project root:
    python src/final_analysis.py

The script uses only held-out test examples for t-SNE/case studies.
It does not train or tune any model.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


def find_root() -> Path:
    cwd = Path.cwd().resolve()
    for p in (cwd, cwd.parent):
        if (p / "src" / "gnn_model.py").exists() and (p / "results" / "metrics.json").exists():
            return p
    raise SystemExit(
        f"[fatal] could not locate project root from {cwd}\n"
        "Run this command from G:\\CSE425_Project"
    )


ROOT = find_root()
sys.path.insert(0, str(ROOT / "src"))
from gnn_model import ContextModel, ModelCfg  # noqa: E402

REL_NAMES = {0: "temporal", 1: "harmonic", 2: "timbral"}


def load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def locate_run(metrics: dict, run_name: str) -> dict:
    for key, rec in metrics.items():
        if key.split("/")[-1] == run_name:
            return rec
    raise KeyError(run_name)


def sample_f1(y: np.ndarray, pred: np.ndarray) -> float:
    y = y.astype(bool)
    pred = pred.astype(bool)
    tp = np.logical_and(y, pred).sum()
    fp = np.logical_and(~y, pred).sum()
    fn = np.logical_and(y, ~pred).sum()
    den = 2 * tp + fp + fn
    return float(2 * tp / den) if den else 1.0


def make_data(graph_rec: dict, bert: dict, bert_idx: dict[str, int]) -> Data:
    tid = graph_rec["track_id"]
    j = bert_idx[tid]
    d = Data(
        x=graph_rec["x"].float(),
        edge_index=graph_rec["edge_index"].long(),
        edge_type=graph_rec["edge_type"].long(),
    )
    d.y = graph_rec["y"].float().unsqueeze(0)
    d.text_cls = bert["cls"][j].float().unsqueeze(0)
    d.text_tokens = bert["tokens"][j].float().unsqueeze(0)
    d.text_mask = bert["mask"][j].float().unsqueeze(0)
    d.track_id = tid
    d.text_context = graph_rec.get("text", "")
    return d


def load_checkpoint_model(path: Path, device: str):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ModelCfg(**ckpt["cfg"])
    model = ContextModel(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device).eval()
    thresholds = np.asarray(ckpt["thresholds"], dtype=float)
    tags = list(ckpt["target_tags"])
    return model, thresholds, tags


def training_curves(metrics: dict, out_dir: Path, seed: int):
    wanted = [
        ("BERT", f"MTT_BERT_s{seed}"),
        ("A3 GNN", f"MTT_A3_s{seed}"),
        ("A4 concat", f"MTT_A4_s{seed}"),
        ("A5 cross-attn", f"MTT_A5_s{seed}"),
        ("B2 CNN", f"MTT_B2_s{seed}"),
    ]
    curves = []
    for label, run in wanted:
        try:
            hist = locate_run(metrics, run).get("history", [])
            if hist:
                curves.append((label, hist))
        except KeyError:
            print(f"[warn] missing history: {run}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for label, hist in curves:
        ep = [h["epoch"] for h in hist]
        axes[0].plot(
            ep, [h.get("macro_f1", np.nan) for h in hist],
            marker="o", markersize=2.5, linewidth=1.5, label=label
        )
        axes[1].plot(
            ep, [h.get("auc_pr", np.nan) for h in hist],
            marker="o", markersize=2.5, linewidth=1.5, label=label
        )
    axes[0].set_title(f"Validation Macro-F1 (seed {seed})")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Macro-F1")
    axes[0].grid(alpha=0.25)

    axes[1].set_title(f"Validation AUC-PR (seed {seed})")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("AUC-PR")
    axes[1].grid(alpha=0.25)

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(5, len(labels)), frameon=False)
    fig.suptitle("MagnaTagATune training dynamics", y=1.02)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    path = out_dir / "training_curves.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {path}")


def load_test_data(root: Path):
    graphs_dir = root / "data" / "processed" / "graphs" / "magnatagatune"
    bert_path = root / "data" / "processed" / "bert_cache_magnatagatune.pt"
    labels_path = root / "data" / "splits" / "magnatagatune_labels.json"

    for p in (graphs_dir, bert_path, labels_path):
        if not p.exists():
            raise SystemExit(f"[fatal] missing required artifact: {p}")

    bert = torch.load(bert_path, map_location="cpu", weights_only=False)
    bert_idx = {tid: i for i, tid in enumerate(bert["ids"])}
    labels = load_json(labels_path)

    data_list = []
    graph_paths = {}
    skipped = 0
    for f in sorted(graphs_dir.glob("*.pt")):
        r = torch.load(f, map_location="cpu", weights_only=False)
        if r.get("split") != "test":
            continue
        if r["track_id"] not in bert_idx:
            skipped += 1
            continue
        data_list.append(make_data(r, bert, bert_idx))
        graph_paths[r["track_id"]] = f

    if not data_list:
        raise SystemExit("[fatal] no held-out MTT test graphs found")

    print(f"[data] held-out test graphs: {len(data_list)} (skipped={skipped})")
    return data_list, graph_paths, bert, bert_idx, labels


@torch.no_grad()
def collect_outputs(model, data_list, device: str, batch_size: int = 64):
    loader = DataLoader(data_list, batch_size=batch_size, shuffle=False)
    Z, P, Y, IDS, TEXTS = [], [], [], [], []

    for b in loader:
        ids = list(b.track_id)
        texts = list(b.text_context)
        b = b.to(device)
        out = model(b)
        Z.append(out["z"].detach().cpu().numpy())
        P.append(torch.sigmoid(out["logits"]).detach().cpu().numpy())
        Y.append(b.y.detach().cpu().numpy())
        IDS.extend(ids)
        TEXTS.extend(texts)

    return (
        np.concatenate(Z),
        np.concatenate(P),
        np.concatenate(Y),
        IDS,
        TEXTS,
    )


def tsne_fusion(z: np.ndarray, y: np.ndarray, tags: list[str], out_dir: Path, seed: int):
    n = len(z)
    zs = StandardScaler().fit_transform(z)
    pca_dim = min(50, zs.shape[1], n - 1)
    zp = PCA(n_components=pca_dim, random_state=seed).fit_transform(zs)

    emb = TSNE(
        n_components=2,
        perplexity=min(30.0, max(5.0, (n - 1) / 3.0)),
        init="pca",
        learning_rate="auto",
        max_iter=1500,
        random_state=seed,
    ).fit_transform(zp)

    freq = y.sum(axis=0)
    top = np.argsort(-freq)[:7]
    group = []
    for row in y:
        label = "other"
        for k in top:
            if row[k] > 0.5:
                label = tags[k]
                break
        group.append(label)
    group = np.asarray(group)

    fig, ax = plt.subplots(figsize=(9, 7))
    for name in [tags[k] for k in top] + ["other"]:
        m = group == name
        if np.any(m):
            ax.scatter(
                emb[m, 0], emb[m, 1],
                s=18 if name != "other" else 10,
                alpha=0.72 if name != "other" else 0.25,
                label=f"{name} (n={int(m.sum())})",
            )
    ax.set_title("A4 fusion representation on held-out MTT test set")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.15)
    fig.tight_layout()

    path = out_dir / "tsne_fusion.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {path}")


@torch.no_grad()
def one_a5_case(
    model,
    graph_path: Path,
    bert: dict,
    bert_idx: dict[str, int],
    thresholds: np.ndarray,
    tags: list[str],
    device: str,
):
    r = torch.load(graph_path, map_location="cpu", weights_only=False)
    d = make_data(r, bert, bert_idx)
    d.batch = torch.zeros(d.x.shape[0], dtype=torch.long)

    out = model(d.to(device), want_attn=True)
    probs = torch.sigmoid(out["logits"]).squeeze(0).cpu().numpy()
    y = r["y"].cpu().numpy().astype(int)
    pred = (probs >= thresholds).astype(int)
    top_idx = np.argsort(-probs)[:8]

    xw = out.get("xattn_weights")
    if xw is not None:
        xw = xw.squeeze(0).detach().cpu().numpy()
        mask = d.text_mask.squeeze(0).cpu().numpy().astype(bool)
        valid = xw[mask] if mask.any() else xw
        seg = valid.mean(axis=0)
        if seg.sum() > 0:
            seg = seg / seg.sum()
    else:
        seg = np.zeros(r["x"].shape[0], dtype=float)

    rel_values = {name: [] for name in REL_NAMES.values()}
    for layer in out.get("rel_attn", []):
        for rid, pair in layer.items():
            _, alpha = pair
            if alpha.numel():
                rel_values[REL_NAMES[int(rid)]].append(
                    float(alpha.float().mean().cpu())
                )
    rel_mean = {
        k: float(np.mean(v)) if v else 0.0
        for k, v in rel_values.items()
    }

    return {
        "track_id": r["track_id"],
        "text": r.get("text", ""),
        "sample_f1": sample_f1(y, pred),
        "true_tags": [t for t, q in zip(tags, y) if q == 1],
        "predicted_tags": [t for t, q in zip(tags, pred) if q == 1],
        "top_predictions": [
            {
                "tag": tags[int(k)],
                "probability": float(probs[k]),
                "threshold": float(thresholds[k]),
                "is_true": bool(y[k]),
                "is_predicted": bool(pred[k]),
            }
            for k in top_idx
        ],
        "num_nodes": int(r["x"].shape[0]),
        "num_edges": int(r["edge_index"].shape[1]),
        "edge_counts": {
            REL_NAMES[rid]: int((r["edge_type"] == rid).sum().item())
            for rid in REL_NAMES
        },
        "relation_attention_mean": rel_mean,
        "segment_attention": [float(x) for x in seg],
    }


def choose_cases(probs, y, texts, thresholds, ids):
    pred = (probs >= thresholds[None, :]).astype(int)
    scores = np.asarray([sample_f1(a, b) for a, b in zip(y, pred)])

    candidates = [
        i for i in range(len(ids))
        if y[i].sum() > 0
        and texts[i].strip()
        and texts[i].strip().lower() != "unlabelled audio"
    ]
    if len(candidates) < 3:
        candidates = [i for i in range(len(ids)) if y[i].sum() > 0]

    cs = np.asarray(candidates, dtype=int)
    vals = scores[cs]
    med = float(np.median(vals))

    best = int(cs[np.argmax(vals)])
    typical = next(
        int(i) for i in cs[np.argsort(np.abs(vals - med))]
        if int(i) != best
    )
    hard = next(
        int(i) for i in cs[np.argsort(vals)]
        if int(i) not in {best, typical}
    )

    return [("Best", best), ("Typical", typical), ("Challenging", hard)]


def case_studies(
    root, out_dir, graph_paths, bert, bert_idx,
    device, seed
):
    a5_path = root / "results" / "checkpoints" / f"magnatagatune_MTT_A5_s{seed}.pt"
    if not a5_path.exists():
        print(f"[warn] missing {a5_path}; skipping case studies")
        return

    model, thresholds, tags = load_checkpoint_model(a5_path, device)

    data_list = []
    for tid, path in graph_paths.items():
        r = torch.load(path, map_location="cpu", weights_only=False)
        data_list.append(make_data(r, bert, bert_idx))

    _, probs, y, ids, texts = collect_outputs(model, data_list, device)
    chosen = choose_cases(probs, y, texts, thresholds, ids)

    records = []
    for label, idx in chosen:
        rec = one_a5_case(
            model, graph_paths[ids[idx]], bert, bert_idx,
            thresholds, tags, device
        )
        rec["case_type"] = label
        records.append(rec)

    json_path = out_dir / "case_studies.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"[save] {json_path}")

    fig, axes = plt.subplots(3, 2, figsize=(13, 11))
    for row, rec in enumerate(records):
        top = rec["top_predictions"][::-1]
        ylabels = [
            ("★ " if x["is_true"] else "") + x["tag"]
            for x in top
        ]
        axes[row, 0].barh(
            ylabels, [x["probability"] for x in top]
        )
        axes[row, 0].set_xlim(0, 1)
        axes[row, 0].set_xlabel("Predicted probability")
        axes[row, 0].set_title(
            f"{rec['case_type']} — F1={rec['sample_f1']:.2f}\n"
            f"text: {rec['text']}"
        )
        axes[row, 0].grid(axis="x", alpha=0.2)

        seg = np.asarray(rec["segment_attention"])
        axes[row, 1].plot(
            np.arange(len(seg)), seg, marker="o", markersize=3
        )
        axes[row, 1].set_xlabel("Audio segment (1 s windows)")
        axes[row, 1].set_ylabel("Normalized attention")
        axes[row, 1].set_title(
            "A5 token→segment grounding\n"
            + ", ".join(
                f"{k}={v:.3f}"
                for k, v in rec["relation_attention_mean"].items()
            )
        )
        axes[row, 1].grid(alpha=0.2)

    fig.suptitle(
        "A5 qualitative held-out case studies (★ = ground-truth tag)",
        y=1.01
    )
    fig.tight_layout()
    path = out_dir / "case_studies.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {path}")


def bert_examples(root, out_dir, data_list, device, seed):
    ckpt_path = root / "results" / "checkpoints" / f"magnatagatune_MTT_BERT_s{seed}.pt"
    if not ckpt_path.exists():
        print(f"[warn] missing {ckpt_path}; skipping BERT examples")
        return

    model, thresholds, tags = load_checkpoint_model(ckpt_path, device)
    _, probs, y, ids, texts = collect_outputs(model, data_list, device)

    candidates = [
        i for i, txt in enumerate(texts)
        if txt.strip()
        and txt.strip().lower() != "unlabelled audio"
        and y[i].sum() > 0
    ]
    if len(candidates) < 5:
        candidates = [i for i in range(len(ids)) if y[i].sum() > 0]

    pos = np.linspace(0, len(candidates) - 1, 5).round().astype(int)
    chosen = [candidates[int(p)] for p in pos]

    rows = []
    for i in chosen:
        pred = probs[i] >= thresholds
        top = np.argsort(-probs[i])[:5]
        rows.append({
            "track_id": ids[i],
            "text": texts[i],
            "true_tags": [t for t, q in zip(tags, y[i]) if q > 0.5],
            "predicted_tags": [t for t, q in zip(tags, pred) if q],
            "top5": [
                {"tag": tags[int(k)], "probability": float(probs[i, k])}
                for k in top
            ],
        })

    path = out_dir / "bert_examples.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"[save] {path}")


def gtzan_csv(metrics: dict, out_dir: Path):
    specs = [
        ("A0 no message passing", ["A0_s425", "A0_s426", "A0_s427"]),
        ("A1 temporal GraphSAGE", ["A1"]),
        ("A2 all-rel GraphSAGE", ["A2"]),
        ("A2 capacity-matched", ["A2cap"]),
        ("A3 relation-GAT", ["A3_s425", "A3_s426", "A3_s427"]),
        ("C1 shuffled relation-GAT", ["C1_s425", "C1_s426", "C1_s427"]),
    ]

    rows = []
    for label, names in specs:
        acc, f1 = [], []
        for name in names:
            found = None
            for key, rec in metrics.items():
                if not key.startswith("gtzan"):
                    continue
                if key.split("/")[-1] != name:
                    continue
                if name.startswith("C1_s") and rec.get("graphs_suffix") != "_shuffled":
                    continue
                found = rec
                break
            if found:
                acc.append(float(found["test"]["accuracy"]))
                f1.append(float(found["test"]["macro_f1"]))

        if acc:
            rows.append({
                "model": label,
                "n_seeds": len(acc),
                "accuracy_mean": float(np.mean(acc)),
                "accuracy_sd": float(np.std(acc, ddof=1)) if len(acc) > 1 else 0.0,
                "macro_f1_mean": float(np.mean(f1)),
                "macro_f1_sd": float(np.std(f1, ddof=1)) if len(f1) > 1 else 0.0,
            })

    path = out_dir / "gtzan_ablation.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "model", "n_seeds",
                "accuracy_mean", "accuracy_sd",
                "macro_f1_mean", "macro_f1_sd",
            ],
        )
        w.writeheader()
        w.writerows(rows)
    print(f"[save] {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=425)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    out_dir = ROOT / "results" / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = load_json(ROOT / "results" / "metrics.json")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[root] {ROOT}")
    print(f"[device] {device}")
    print("[note] analysis only — no training or parameter tuning")

    training_curves(metrics, out_dir, args.seed)
    gtzan_csv(metrics, out_dir)

    data_list, graph_paths, bert, bert_idx, labels = load_test_data(ROOT)

    a4_path = ROOT / "results" / "checkpoints" / f"magnatagatune_MTT_A4_s{args.seed}.pt"
    if not a4_path.exists():
        raise SystemExit(f"[fatal] missing A4 checkpoint: {a4_path}")

    a4, thresholds, tags = load_checkpoint_model(a4_path, device)
    z, probs, y, ids, texts = collect_outputs(
        a4, data_list, device, batch_size=args.batch_size
    )
    print(f"[A4] z={z.shape} probs={probs.shape} y={y.shape}")

    tsne_fusion(z, y, tags, out_dir, args.seed)
    case_studies(
        ROOT, out_dir, graph_paths, bert, bert_idx,
        device, args.seed
    )
    bert_examples(ROOT, out_dir, data_list, device, args.seed)

    print("\n=== FINAL ANALYSIS COMPLETE ===")
    print(f"Open: {out_dir}")
    print("  training_curves.png")
    print("  tsne_fusion.png")
    print("  case_studies.png")
    print("  case_studies.json")
    print("  bert_examples.json")
    print("  gtzan_ablation.csv")


if __name__ == "__main__":
    main()

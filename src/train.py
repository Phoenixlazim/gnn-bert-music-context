"""
train.py
========
Trains one ablation preset and appends its metrics to results/metrics.json.

    python src\\train.py --dataset gtzan        --preset A3 --epochs 40
    python src\\train.py --dataset magnatagatune --preset A5 --epochs 30
    python src\\train.py --dataset magnatagatune --preset C1   # auto-uses _shuffled graphs
    python src\\train.py --dataset magnatagatune --preset A2 --hidden 152 --run-name A2cap

Notes that matter for the report:
  * Per-tag thresholds are tuned on VALIDATION, never fixed at 0.5. On an
    imbalanced tag set a flat threshold understates macro-F1 badly.
  * Model selection is on validation mean AUC-PR (threshold-free), then the
    selected checkpoint is evaluated once on test.
  * --hidden lets you capacity-match A2 to A3 so a win for relation typing
    cannot be dismissed as extra parameters.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).parent))
from gnn_model import ABLATIONS, ContextModel, info_nce, make_cfg  # noqa: E402


# ----------------------------------------------------------------------
def load_dataset(cfg, dataset, suffix, bert_cache):
    gdir = Path(cfg["paths"]["processed"]) / "graphs" / f"{dataset}{suffix}"
    files = sorted(gdir.glob("*.pt"))
    if not files:
        raise SystemExit(f"[fatal] no graphs in {gdir} — run graph_builder.py")

    cache = torch.load(bert_cache, map_location="cpu", weights_only=False)
    idx = {t: i for i, t in enumerate(cache["ids"])}

    splits = {"train": [], "val": [], "test": []}
    missing = 0
    for f in files:
        r = torch.load(f, map_location="cpu", weights_only=False)
        tid = r["track_id"]
        if tid not in idx:
            missing += 1
            continue
        j = idx[tid]
        d = Data(x=r["x"], edge_index=r["edge_index"], edge_type=r["edge_type"])
        d.y = r["y"].unsqueeze(0)
        d.text_cls = cache["cls"][j].float().unsqueeze(0)
        d.text_tokens = cache["tokens"][j].float().unsqueeze(0)
        d.text_mask = cache["mask"][j].float().unsqueeze(0)
        d.track_id = tid
        splits[r["split"]].append(d)

    if missing:
        print(f"[warn] {missing} graphs had no BERT cache entry and were skipped")
    for k, v in splits.items():
        print(f"[data] {k}: {len(v)}")
    if not splits["train"] or not splits["test"]:
        raise SystemExit("[fatal] empty train or test split")
    return splits


# ----------------------------------------------------------------------
@torch.no_grad()
def predict(model, loader, dev):
    model.eval()
    P, Y = [], []
    for b in loader:
        b = b.to(dev)
        P.append(torch.sigmoid(model(b)["logits"]).cpu().numpy())
        Y.append(b.y.cpu().numpy())
    return np.concatenate(P), np.concatenate(Y)


def tune_thresholds(p_val, y_val):
    """Per-tag threshold maximising F1 on validation."""
    grid = np.linspace(0.05, 0.95, 19)
    th = np.full(y_val.shape[1], 0.5)
    for k in range(y_val.shape[1]):
        if y_val[:, k].sum() == 0:
            continue
        best, best_t = -1.0, 0.5
        for t in grid:
            f = f1_score(y_val[:, k], (p_val[:, k] >= t).astype(int),
                         zero_division=0)
            if f > best:
                best, best_t = f, t
        th[k] = best_t
    return th


def evaluate(p, y, th, multiclass=False):
    m = {}
    if multiclass:
        m["accuracy"] = float((p.argmax(1) == y.argmax(1)).mean())
        m["macro_f1"] = float(f1_score(y.argmax(1), p.argmax(1),
                                       average="macro", zero_division=0))
        return m

    pred = (p >= th[None, :]).astype(int)
    m["macro_f1"] = float(f1_score(y, pred, average="macro", zero_division=0))
    m["micro_f1"] = float(f1_score(y, pred, average="micro", zero_division=0))

    keep = y.sum(0) > 0
    if keep.any():
        pr_scores = [
            average_precision_score(y[:, k], p[:, k])
            for k in range(y.shape[1]) if keep[k]
        ]
        if pr_scores:
            m["auc_pr"] = float(np.mean(pr_scores))

        roc_scores = []
        for k in range(y.shape[1]):
            # ROC-AUC is undefined if a tag has only one class in this split.
            if keep[k] and y[:, k].sum() < len(y):
                try:
                    roc_scores.append(roc_auc_score(y[:, k], p[:, k]))
                except ValueError:
                    pass
        m["auc_roc"] = float(np.mean(roc_scores)) if roc_scores else float("nan")

    m["n_eval_tags"] = int(keep.sum())
    return m


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["gtzan", "magnatagatune"])
    ap.add_argument("--preset", required=True, choices=list(ABLATIONS))
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--graphs-suffix", default="", help="e.g. _shuffled for C1")
    ap.add_argument("--labels", default=None)
    ap.add_argument("--bert-cache", default=None)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--contrastive", type=float, default=0.0,
                    help="lambda for the optional InfoNCE alignment term")
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    seed = args.seed if args.seed is not None else cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # C1 is specifically the degree-preserving shuffled-graph control.
    # Auto-select the shuffled graph directory so `--preset C1` cannot
    # accidentally evaluate the ordinary A3 graphs.
    if args.preset == "C1" and not args.graphs_suffix:
        args.graphs_suffix = "_shuffled"
        print("[auto] C1 selected: using graphs suffix '_shuffled'")

    lab_path = Path(args.labels) if args.labels else \
        Path(cfg["paths"]["splits"]) / f"{args.dataset}_labels.json"
    lab = json.load(open(lab_path))
    K = len(lab["target_tags"])
    multiclass = lab["task"] == "multiclass"

    bert_cache = Path(args.bert_cache) if args.bert_cache else \
        Path(cfg["paths"]["processed"]) / f"bert_cache_{args.dataset}.pt"
    if not bert_cache.exists():
        raise SystemExit(f"[fatal] {bert_cache} missing — run bert_encoder.py")

    run = args.run_name or args.preset
    print(f"=== {run} | {args.dataset}{args.graphs_suffix} | "
          f"{K} targets | device={dev} ===")

    sp = load_dataset(cfg, args.dataset, args.graphs_suffix, bert_cache)
    dl = {k: DataLoader(v, batch_size=args.batch_size, shuffle=(k == "train"))
          for k, v in sp.items()}

    mcfg = make_cfg(args.preset, n_classes=K, hidden=args.hidden,
                    in_dim=sp["train"][0].x.shape[1])
    model = ContextModel(mcfg).to(dev)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"[model] conv={mcfg.conv} relations={mcfg.relations} "
          f"fusion={mcfg.fusion} params={nparam/1e3:.1f}k")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    # positive weighting stabilises macro-F1 on imbalanced tag sets
    if not multiclass:
        Y = torch.cat([d.y for d in sp["train"]])
        pos = Y.sum(0).clamp(min=1)
        pw = ((len(Y) - pos) / pos).clamp(max=20.0).to(dev)
    else:
        pw = None

    best, best_state, best_th, best_epoch, bad, hist = -1.0, None, None, 0, 0, []
    t0 = time.time()

    for ep in range(1, args.epochs + 1):
        model.train()
        tot = 0.0
        for b in dl["train"]:
            b = b.to(dev)
            opt.zero_grad(set_to_none=True)
            out = model(b)
            if multiclass:
                loss = F.cross_entropy(out["logits"], b.y.argmax(1))
            else:
                loss = F.binary_cross_entropy_with_logits(
                    out["logits"], b.y, pos_weight=pw)
            if args.contrastive > 0:
                if "emb_g" not in out or "emb_t" not in out:
                    raise RuntimeError(
                        "--contrastive > 0 requires both 'emb_g' and 'emb_t' "
                        "from the selected model preset"
                    )
                loss = loss + args.contrastive * info_nce(out["emb_g"], out["emb_t"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.detach().item() * b.num_graphs
        sched.step()

        pv, yv = predict(model, dl["val"], dev)
        thv = np.full(K, 0.5) if multiclass else tune_thresholds(pv, yv)
        mv = evaluate(pv, yv, thv, multiclass)
        crit = mv.get("auc_pr", mv.get("accuracy", mv["macro_f1"]))
        if not np.isfinite(crit):
            # Degenerate validation tags can make threshold-free metrics undefined.
            # Fall back to macro-F1 rather than leaving best_state unset.
            crit = mv["macro_f1"]

        hist.append({"epoch": ep, "train_loss": tot / len(sp["train"]), **mv})
        print(f"  ep{ep:3d} loss={tot/len(sp['train']):.4f} "
              + " ".join(f"val_{k}={v:.4f}" for k, v in mv.items()
                         if isinstance(v, float)))

        if crit > best:
            best, bad = crit, 0
            best_epoch = ep
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            best_th = thv.copy()
        else:
            bad += 1
            if bad >= args.patience:
                print(f"  early stop at epoch {ep}")
                break

    if best_state is None or best_th is None:
        raise RuntimeError(
            "No valid checkpoint was selected. Check validation metrics for NaN/Inf."
        )

    print(f"[best] epoch={best_epoch}  val_selection_metric={best:.4f}")
    model.load_state_dict(best_state)
    pt, yt = predict(model, dl["test"], dev)
    mt = evaluate(pt, yt, best_th, multiclass)
    mins = (time.time() - t0) / 60
    print(f"\n[TEST] " + "  ".join(f"{k}={v:.4f}" for k, v in mt.items()
                                   if isinstance(v, float)))
    print(f"[time] {mins:.1f} min")

    res_dir = Path(cfg["paths"]["results"]); res_dir.mkdir(parents=True, exist_ok=True)
    mpath = res_dir / "metrics.json"
    allm = json.load(open(mpath)) if mpath.exists() else {}
    allm[f"{args.dataset}{args.graphs_suffix}/{run}"] = {
        "preset": args.preset, "dataset": args.dataset,
        "graphs_suffix": args.graphs_suffix, "tag_split_mode": lab["mode"],
        "params": nparam, "hidden": args.hidden, "seed": seed,
        "epochs_run": len(hist), "best_epoch": best_epoch,
        "minutes": round(mins, 2),
        "conv": mcfg.conv, "relations": mcfg.relations, "fusion": mcfg.fusion,
        "contrastive_lambda": args.contrastive,
        "n_train": len(sp["train"]), "n_val": len(sp["val"]),
        "n_test": len(sp["test"]),
        "val_best": best, "test": mt, "history": hist,
    }
    json.dump(allm, open(mpath, "w"), indent=1)

    ck = res_dir / "checkpoints"; ck.mkdir(exist_ok=True)
    torch.save({"state_dict": best_state, "cfg": mcfg.__dict__,
                "thresholds": best_th.tolist(),
                "target_tags": lab["target_tags"]},
               ck / f"{args.dataset}{args.graphs_suffix}_{run}.pt")
    print(f"[save] {mpath}  and  {ck}")


if __name__ == "__main__":
    main()

"""
baselines.py
============
Baselines for the CSE425 GNN-BERT music-context project.

B1  Frequency-prior baseline (deterministic sanity check)
B2  Small 2-D CNN over cached log-mel spectrograms
B4  PCA + MLP over pooled handcrafted 63-D segment features

The script uses the SAME graph-derived train/val/test split, validation-tuned
per-tag thresholds, and test metrics as train.py, and appends results to
results/metrics.json.

Examples
--------
python src/baselines.py --dataset magnatagatune --baseline B1 --seed 425 --run-name MTT_B1_s425
python src/baselines.py --dataset magnatagatune --baseline B2 --epochs 30 --seed 425 --run-name MTT_B2_s425
python src/baselines.py --dataset magnatagatune --baseline B4 --epochs 30 --seed 425 --run-name MTT_B4_s425
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, TensorDataset


# -----------------------------------------------------------------------------
# Metrics: intentionally mirrors train.py
# -----------------------------------------------------------------------------
def tune_thresholds(p_val: np.ndarray, y_val: np.ndarray) -> np.ndarray:
    grid = np.linspace(0.05, 0.95, 19)
    th = np.full(y_val.shape[1], 0.5, dtype=np.float32)
    for k in range(y_val.shape[1]):
        if y_val[:, k].sum() == 0:
            continue
        best, best_t = -1.0, 0.5
        for t in grid:
            f = f1_score(
                y_val[:, k], (p_val[:, k] >= t).astype(int),
                zero_division=0,
            )
            if f > best:
                best, best_t = f, float(t)
        th[k] = best_t
    return th


def evaluate(p: np.ndarray, y: np.ndarray, th: np.ndarray,
             multiclass: bool = False) -> dict:
    if multiclass:
        return {
            "accuracy": float((p.argmax(1) == y.argmax(1)).mean()),
            "macro_f1": float(
                f1_score(y.argmax(1), p.argmax(1), average="macro", zero_division=0)
            ),
        }

    pred = (p >= th[None, :]).astype(int)
    m = {
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y, pred, average="micro", zero_division=0)),
    }

    keep = y.sum(0) > 0
    if keep.any():
        m["auc_pr"] = float(np.mean([
            average_precision_score(y[:, k], p[:, k])
            for k in range(y.shape[1]) if keep[k]
        ]))
        roc_vals = []
        for k in range(y.shape[1]):
            if keep[k] and y[:, k].sum() < len(y):
                try:
                    roc_vals.append(roc_auc_score(y[:, k], p[:, k]))
                except ValueError:
                    pass
        m["auc_roc"] = float(np.mean(roc_vals)) if roc_vals else float("nan")
    m["n_eval_tags"] = int(keep.sum())
    return m


# -----------------------------------------------------------------------------
# Split/label discovery. Graph .pt files are the authoritative split source.
# -----------------------------------------------------------------------------
def load_split_records(cfg: dict, dataset: str) -> tuple[dict, dict[str, list[str]]]:
    lab_path = Path(cfg["paths"]["splits"]) / f"{dataset}_labels.json"
    if not lab_path.exists():
        raise SystemExit(f"[fatal] {lab_path} missing — run labels.py first")
    lab = json.load(open(lab_path, encoding="utf-8"))

    gdir = Path(cfg["paths"]["processed"]) / "graphs" / dataset
    files = sorted(gdir.glob("*.pt"))
    if not files:
        raise SystemExit(f"[fatal] no graphs in {gdir} — run graph_builder.py first")

    splits = {"train": [], "val": [], "test": []}
    for f in files:
        r = torch.load(f, map_location="cpu", weights_only=False)
        tid = r["track_id"]
        if tid in lab["tracks"] and r["split"] in splits:
            splits[r["split"]].append(tid)

    for s in ("train", "val", "test"):
        print(f"[data] {s}: {len(splits[s])}")
    if not splits["train"] or not splits["val"] or not splits["test"]:
        raise SystemExit("[fatal] empty train/val/test split")
    return lab, splits


def labels_for(ids: list[str], lab: dict) -> np.ndarray:
    return np.asarray([lab["tracks"][tid]["y"] for tid in ids], dtype=np.float32)


# -----------------------------------------------------------------------------
# B1: deterministic frequency-prior baseline
# -----------------------------------------------------------------------------
def run_b1(lab: dict, splits: dict[str, list[str]], multiclass: bool) -> tuple[dict, dict]:
    ytr = labels_for(splits["train"], lab)
    yv = labels_for(splits["val"], lab)
    yt = labels_for(splits["test"], lab)

    prior = ytr.mean(0)
    if multiclass:
        prior = prior / max(prior.sum(), 1e-8)
        pv = np.repeat(prior[None, :], len(yv), axis=0)
        pt = np.repeat(prior[None, :], len(yt), axis=0)
        th = np.full(prior.shape[0], 0.5, dtype=np.float32)
    else:
        pv = np.repeat(prior[None, :], len(yv), axis=0)
        pt = np.repeat(prior[None, :], len(yt), axis=0)
        th = tune_thresholds(pv, yv)

    mv = evaluate(pv, yv, th, multiclass)
    mt = evaluate(pt, yt, th, multiclass)
    return {"val": mv, "test": mt, "thresholds": th.tolist()}, {}


# -----------------------------------------------------------------------------
# B2: 2-D CNN over cached log-mel spectrograms
# -----------------------------------------------------------------------------
class MelDataset(Dataset):
    def __init__(self, ids: list[str], lab: dict, mel_dir: Path, frames: int):
        self.ids = ids
        self.lab = lab
        self.mel_dir = mel_dir
        self.frames = frames

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i: int):
        tid = self.ids[i]
        p = self.mel_dir / f"{tid}.npy"
        if not p.exists():
            raise FileNotFoundError(f"missing mel cache: {p}")
        x = np.load(p).astype(np.float32)  # [n_mels, T]
        # Per-track normalization. This keeps scale stable without leaking splits.
        x = (x - x.mean()) / (x.std() + 1e-6)
        if x.shape[1] < self.frames:
            x = np.pad(x, ((0, 0), (0, self.frames - x.shape[1])))
        else:
            x = x[:, :self.frames]
        y = np.asarray(self.lab["tracks"][tid]["y"], dtype=np.float32)
        return torch.from_numpy(x[None, :, :]), torch.from_numpy(y)


class MelCNN(nn.Module):
    def __init__(self, n_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=2, padding=2),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.25), nn.Linear(64, n_out)
        )

    def forward(self, x):
        return self.head(self.net(x))


# -----------------------------------------------------------------------------
# B4: PCA + MLP over mean/std pooled 63-D handcrafted segment features
# -----------------------------------------------------------------------------
def pooled_feature(npz_path: Path) -> np.ndarray:
    d = np.load(npz_path, allow_pickle=True)
    x = d["x"].astype(np.float32)
    return np.concatenate([x.mean(0), x.std(0)], axis=0).astype(np.float32)


class MLP(nn.Module):
    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 128), nn.ReLU(inplace=True), nn.Dropout(0.25),
            nn.Linear(128, 64), nn.ReLU(inplace=True), nn.Dropout(0.15),
            nn.Linear(64, n_out),
        )

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def predict_loader(model: nn.Module, loader: DataLoader, dev: str) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ps, ys = [], []
    for x, y in loader:
        x = x.to(dev, non_blocking=True)
        logits = model(x)
        ps.append(torch.sigmoid(logits).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ps), np.concatenate(ys)


def train_model(model: nn.Module, loaders: dict[str, DataLoader], y_train: np.ndarray,
                multiclass: bool, dev: str, epochs: int, lr: float,
                weight_decay: float, patience: int) -> tuple[dict, dict, dict]:
    model = model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))

    if multiclass:
        pos_weight = None
    else:
        pos = torch.tensor(y_train, dtype=torch.float32).sum(0).clamp(min=1)
        pos_weight = ((len(y_train) - pos) / pos).clamp(max=20.0).to(dev)

    best = -float("inf")
    best_state = None
    best_th = None
    best_epoch = None
    bad = 0
    hist = []

    for ep in range(1, epochs + 1):
        model.train()
        total, n = 0.0, 0
        for x, y in loaders["train"]:
            x = x.to(dev, non_blocking=True)
            y = y.to(dev, non_blocking=True)
            logits = model(x)
            if multiclass:
                loss = F.cross_entropy(logits, y.argmax(1))
            else:
                loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += loss.detach().item() * len(x)
            n += len(x)
        sched.step()

        pv, yv = predict_loader(model, loaders["val"], dev)
        th = np.full(yv.shape[1], 0.5, dtype=np.float32) if multiclass else tune_thresholds(pv, yv)
        mv = evaluate(pv, yv, th, multiclass)
        crit = mv.get("auc_pr", mv.get("accuracy", mv.get("macro_f1", float("nan"))))
        if not np.isfinite(crit):
            crit = mv.get("macro_f1", -float("inf"))
        hist.append({"epoch": ep, "train_loss": total / max(n, 1), **mv})
        print(
            f"  ep{ep:3d} loss={total/max(n,1):.4f} "
            + " ".join(f"val_{k}={v:.4f}" for k, v in mv.items() if isinstance(v, float))
        )

        if crit > best:
            best = float(crit)
            best_epoch = ep
            best_th = th.copy()
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"  early stop at epoch {ep}")
                break

    if best_state is None:
        raise RuntimeError("no valid checkpoint was selected")
    model.load_state_dict(best_state)
    pt, yt = predict_loader(model, loaders["test"], dev)
    mt = evaluate(pt, yt, best_th, multiclass)
    print(f"[best] epoch={best_epoch}  val_selection_metric={best:.4f}")
    return {"test": mt, "history": hist, "val_best": best,
            "best_epoch": best_epoch, "thresholds": best_th.tolist()}, best_state, {
                "params": sum(p.numel() for p in model.parameters())
            }


# -----------------------------------------------------------------------------
def append_metrics(cfg: dict, dataset: str, run: str, baseline: str, seed: int,
                   minutes: float, lab: dict, result: dict, extra: dict):
    rdir = Path(cfg["paths"]["results"])
    rdir.mkdir(parents=True, exist_ok=True)
    mpath = rdir / "metrics.json"
    allm = json.load(open(mpath, encoding="utf-8")) if mpath.exists() else {}

    allm[f"{dataset}/{run}"] = {
        "preset": baseline,
        "baseline": baseline,
        "dataset": dataset,
        "tag_split_mode": lab.get("mode"),
        "seed": seed,
        "minutes": round(minutes, 2),
        "n_train": extra.get("n_train"),
        "n_val": extra.get("n_val"),
        "n_test": extra.get("n_test"),
        "params": extra.get("params", 0),
        "best_epoch": result.get("best_epoch"),
        "val_best": result.get("val_best"),
        "test": result["test"],
        "history": result.get("history", []),
        "notes": extra.get("notes", ""),
    }
    json.dump(allm, open(mpath, "w", encoding="utf-8"), indent=1)
    return mpath


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["gtzan", "magnatagatune"])
    ap.add_argument("--baseline", required=True, choices=["B1", "B2", "B4"])
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--pca-dim", type=int, default=64)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 425))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    lab, splits = load_split_records(cfg, args.dataset)
    K = len(lab["target_tags"])
    multiclass = lab["task"] == "multiclass"
    run = args.run_name or args.baseline
    print(f"=== {run} | {args.dataset} | {K} targets | baseline={args.baseline} | device={dev} ===")

    t0 = time.time()
    state = None
    extra = {
        "n_train": len(splits["train"]),
        "n_val": len(splits["val"]),
        "n_test": len(splits["test"]),
    }

    if args.baseline == "B1":
        result, _ = run_b1(lab, splits, multiclass)
        extra["notes"] = "Deterministic train-frequency prior baseline"
        print("[model] deterministic train-frequency prior")

    elif args.baseline == "B2":
        mel_dir = Path(cfg["paths"]["processed"]) / "melspec" / args.dataset
        if not mel_dir.exists():
            raise SystemExit(f"[fatal] {mel_dir} missing — rerun audio_features.py with save_melspec: true")
        frames = int(round(30 * cfg["audio"]["sr"] / cfg["audio"]["hop_length"]))
        ds = {s: MelDataset(ids, lab, mel_dir, frames) for s, ids in splits.items()}
        loaders = {
            s: DataLoader(ds[s], batch_size=args.batch_size, shuffle=(s == "train"),
                          num_workers=0, pin_memory=torch.cuda.is_available())
            for s in ds
        }
        ytr = labels_for(splits["train"], lab)
        model = MelCNN(K)
        print(f"[model] 2-D log-mel CNN params={sum(p.numel() for p in model.parameters())/1e3:.1f}k")
        result, state, info = train_model(
            model, loaders, ytr, multiclass, dev,
            args.epochs, args.lr, args.weight_decay, args.patience,
        )
        extra.update(info)
        extra["notes"] = "2-D CNN over cached 128-bin log-mel spectrograms"

    else:  # B4
        feat_dir = Path(cfg["paths"]["processed"]) / "feats" / args.dataset
        X = {}
        Y = {}
        for s, ids in splits.items():
            X[s] = np.stack([pooled_feature(feat_dir / f"{tid}.npz") for tid in ids])
            Y[s] = labels_for(ids, lab)

        scaler = StandardScaler().fit(X["train"])
        Xs = {s: scaler.transform(X[s]).astype(np.float32) for s in X}
        pca_dim = min(args.pca_dim, Xs["train"].shape[1], Xs["train"].shape[0] - 1)
        pca = PCA(n_components=pca_dim, random_state=seed).fit(Xs["train"])
        Xp = {s: pca.transform(Xs[s]).astype(np.float32) for s in Xs}
        loaders = {
            s: DataLoader(
                TensorDataset(torch.from_numpy(Xp[s]), torch.from_numpy(Y[s])),
                batch_size=args.batch_size, shuffle=(s == "train"),
                num_workers=0, pin_memory=torch.cuda.is_available(),
            )
            for s in Xp
        }
        model = MLP(pca_dim, K)
        print(
            f"[model] pooled handcrafted -> StandardScaler -> PCA({pca_dim}) -> MLP "
            f"params={sum(p.numel() for p in model.parameters())/1e3:.1f}k "
            f"explained_var={pca.explained_variance_ratio_.sum():.3f}"
        )
        result, state, info = train_model(
            model, loaders, Y["train"], multiclass, dev,
            args.epochs, args.lr, args.weight_decay, args.patience,
        )
        extra.update(info)
        extra["notes"] = (
            f"Mean+std pooled 63-D segment features; StandardScaler; PCA({pca_dim}); MLP"
        )

    mins = (time.time() - t0) / 60
    mt = result["test"]
    print("\n[TEST] " + "  ".join(
        f"{k}={v:.4f}" for k, v in mt.items() if isinstance(v, float)
    ))
    print(f"[time] {mins:.1f} min")

    mpath = append_metrics(cfg, args.dataset, run, args.baseline, seed, mins, lab, result, extra)

    if state is not None:
        ck = Path(cfg["paths"]["results"]) / "checkpoints"
        ck.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": state,
            "baseline": args.baseline,
            "target_tags": lab["target_tags"],
            "thresholds": result.get("thresholds"),
            "seed": seed,
        }, ck / f"{args.dataset}_{run}.pt")
    print(f"[save] {mpath}" + ("  and  results\\checkpoints" if state is not None else ""))


if __name__ == "__main__":
    main()

"""
graph_builder.py
================
Builds one multi-relation graph per track from cached segment features.

Relations (edge_type ids):
    0  temporal  — adjacency at hops in graph.temporal_hops, bidirectional
    1  harmonic  — kNN on window-mean chroma (cosine)
    2  timbral   — kNN on window-mean MFCC   (cosine)

Relation-specific ids are what make the per-relation ablation (A1/A2/A3) and the
"attention mass per edge type" figure possible. Do not collapse them.

C1 control: --shuffle applies a degree-preserving double-edge swap within each
relation. Node features and degree sequence are untouched; only the wiring
changes. If performance survives, the graph is a fancy set encoder and we say so.

Output: {processed}/graphs/{dataset}{tag}/{track_id}.pt
    x           float32 [N, D]
    edge_index  int64   [2, E]
    edge_type   int64   [E]
    y           float32 [K]
    split       'train' | 'val' | 'test'
    text        str      (fed to BERT downstream)
    track_id    str

Usage
-----
    python src/graph_builder.py --dataset gtzan
    python src/graph_builder.py --dataset magnatagatune
    python src/graph_builder.py --dataset magnatagatune --shuffle     # C1
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml

REL = {"temporal": 0, "harmonic": 1, "timbral": 2}


# ----------------------------------------------------------------------
# Edge construction
# ----------------------------------------------------------------------
def temporal_edges(n: int, hops: list[int]) -> np.ndarray:
    src, dst = [], []
    for h in hops:
        for i in range(n - h):
            src += [i, i + h]
            dst += [i + h, i]
    return np.array([src, dst], dtype=np.int64) if src else np.zeros((2, 0), np.int64)


def knn_edges(feat: np.ndarray, k: int, tau: float,
              forbid: set[tuple[int, int]] | None = None) -> np.ndarray:
    """Symmetric cosine kNN. feat: [N, F]."""
    n = feat.shape[0]
    if n < 2:
        return np.zeros((2, 0), np.int64)
    f = feat - feat.mean(0, keepdims=True)
    norm = np.linalg.norm(f, axis=1, keepdims=True) + 1e-8
    sim = (f / norm) @ (f / norm).T
    np.fill_diagonal(sim, -np.inf)

    kk = min(k, n - 1)
    pairs = set()
    for i in range(n):
        for j in np.argpartition(-sim[i], kk - 1)[:kk]:
            j = int(j)
            if sim[i, j] <= tau:
                continue
            if forbid and (min(i, j), max(i, j)) in forbid:
                continue
            pairs.add((min(i, j), max(i, j)))

    if not pairs:
        return np.zeros((2, 0), np.int64)
    a = np.array(sorted(pairs), dtype=np.int64)
    return np.concatenate([a.T, a[:, ::-1].T], axis=1)  # both directions


def double_edge_swap(ei: np.ndarray, rng: np.random.Generator,
                     swaps_per_edge: int = 10) -> np.ndarray:
    """Degree-preserving rewiring. (u1,v1),(u2,v2) -> (u1,v2),(u2,v1)."""
    E = ei.shape[1]
    if E < 4:
        return ei
    e = ei.copy()
    existing = {(int(e[0, i]), int(e[1, i])) for i in range(E)}
    target, done, guard = swaps_per_edge * E, 0, 0
    while done < target and guard < target * 20:
        guard += 1
        i, j = rng.integers(0, E, 2)
        if i == j:
            continue
        u1, v1, u2, v2 = int(e[0, i]), int(e[1, i]), int(e[0, j]), int(e[1, j])
        if u1 == v2 or u2 == v1:                      # would self-loop
            continue
        if (u1, v2) in existing or (u2, v1) in existing:
            continue
        existing.discard((u1, v1)); existing.discard((u2, v2))
        existing.add((u1, v2)); existing.add((u2, v1))
        e[1, i], e[1, j] = v2, v1
        done += 1
    return e


def build_graph(npz_path: Path, gcfg: dict, shuffle: bool,
                rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = np.load(npz_path, allow_pickle=True)
    x, chroma, mfcc = d["x"], d["chroma"], d["mfcc"]
    n = x.shape[0]

    blocks, types = [], []
    forbid = None
    if gcfg["knn_exclude_adjacent"]:
        forbid = {(i, i + h) for h in gcfg["temporal_hops"] for i in range(n - h)}

    for name in gcfg["relations"]:
        if name == "temporal":
            e = temporal_edges(n, gcfg["temporal_hops"])
        elif name == "harmonic":
            e = knn_edges(chroma, gcfg["knn_k"], gcfg["knn_tau"], forbid)
        elif name == "timbral":
            e = knn_edges(mfcc, gcfg["knn_k"], gcfg["knn_tau"], forbid)
        else:
            raise ValueError(f"unknown relation: {name}")
        if shuffle and e.shape[1] > 0:
            e = double_edge_swap(e, rng, gcfg["shuffle_control"]["swaps_per_edge"])
        blocks.append(e)
        types.append(np.full(e.shape[1], REL[name], dtype=np.int64))

    ei = np.concatenate(blocks, axis=1) if blocks else np.zeros((2, 0), np.int64)
    et = np.concatenate(types) if types else np.zeros((0,), np.int64)

    if gcfg["add_self_loops"]:
        loops = np.stack([np.arange(n), np.arange(n)]).astype(np.int64)
        ei = np.concatenate([ei, loops], axis=1)
        et = np.concatenate([et, np.full(n, len(REL), dtype=np.int64)])

    return x.astype(np.float32), ei, et


# ----------------------------------------------------------------------
# Splits
# ----------------------------------------------------------------------
def assign_splits(dataset: str, tids: list[str], cfg: dict) -> dict[str, str]:
    if dataset == "magnatagatune":
        m = cfg["datasets"]["magnatagatune"]["split_by_folder"]
        folder2split = {f: s for s, fs in m.items() for f in fs}
        out, unknown = {}, 0
        for t in tids:
            f = t.split("__")[0]
            if f in folder2split:
                out[t] = folder2split[f]
            else:
                unknown += 1
        if unknown:
            print(f"[warn] {unknown} tracks had folder prefixes outside the "
                  f"official split map and were dropped")
        return out

    # GTZAN: stratified by genre, fixed seed. No official split exists;
    # note in the report that GTZAN is known to contain repeats/mislabels.
    sp = cfg["datasets"]["gtzan"]["split"]
    rng = np.random.default_rng(cfg["seed"])
    by_genre: dict[str, list[str]] = {}
    for t in tids:
        by_genre.setdefault(t.split("__")[0], []).append(t)
    out = {}
    for g, ts in by_genre.items():
        ts = sorted(ts)
        rng.shuffle(ts)
        n = len(ts)
        n_tr, n_va = int(sp["train"] * n), int(sp["val"] * n)
        for i, t in enumerate(ts):
            out[t] = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")
    return out


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["gtzan", "magnatagatune"])
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--shuffle", action="store_true",
                    help="C1 control: degree-preserving edge shuffle")
    ap.add_argument("--labels", default=None, help="override labels json path")
    ap.add_argument("--export-samples", type=int, default=20,
                    help="how many graphs to also dump as .json (deliverable)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    gcfg = cfg["graph"]
    proc = Path(cfg["paths"]["processed"])
    feats = proc / "feats" / args.dataset

    lab_path = Path(args.labels) if args.labels else \
        Path(cfg["paths"]["splits"]) / f"{args.dataset}_labels.json"
    if not lab_path.exists():
        raise SystemExit(f"[fatal] {lab_path} missing — run labels.py first")
    lab = json.load(open(lab_path))
    tracks = lab["tracks"]

    tag = "_shuffled" if args.shuffle else ""
    out_dir = proc / "graphs" / f"{args.dataset}{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = proc / "graph_samples" / f"{args.dataset}{tag}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    npzs = sorted(p for p in feats.glob("*.npz") if p.stem in tracks)
    if not npzs:
        raise SystemExit(f"[fatal] no features in {feats} matching {lab_path}")

    splits = assign_splits(args.dataset, [p.stem for p in npzs], cfg)
    npzs = [p for p in npzs if p.stem in splits]

    rng = np.random.default_rng(cfg["seed"])
    stats, n_exported = [], 0

    for p in npzs:
        tid = p.stem
        x, ei, et = build_graph(p, gcfg, args.shuffle, rng)
        rec = {
            "x": torch.from_numpy(x),
            "edge_index": torch.from_numpy(ei),
            "edge_type": torch.from_numpy(et),
            "y": torch.tensor(tracks[tid]["y"], dtype=torch.float32),
            "split": splits[tid],
            "text": tracks[tid]["text"],
            "track_id": tid,
        }
        torch.save(rec, out_dir / f"{tid}.pt")
        stats.append((x.shape[0], ei.shape[1], splits[tid]))

        if n_exported < args.export_samples:
            json.dump({
                "track_id": tid, "split": splits[tid], "text": tracks[tid]["text"],
                "num_nodes": int(x.shape[0]), "feature_dim": int(x.shape[1]),
                "edge_index": ei.tolist(), "edge_type": et.tolist(),
                "relations": {v: k for k, v in REL.items()},
                "y": tracks[tid]["y"], "target_tags": lab["target_tags"],
            }, open(sample_dir / f"{tid}.json", "w"), indent=1)
            n_exported += 1

    ns = np.array([s[0] for s in stats]); es = np.array([s[1] for s in stats])
    sc = Counter(s[2] for s in stats)
    print(f"[done] {len(stats)} graphs -> {out_dir}")
    print(f"[nodes] mean={ns.mean():.1f}  min={ns.min()}  max={ns.max()}")
    print(f"[edges] mean={es.mean():.1f}  mean degree={es.mean()/ns.mean():.2f}")
    print(f"[split] " + "  ".join(f"{k}={sc.get(k,0)}" for k in ("train", "val", "test")))
    print(f"[samples] {n_exported} json graphs -> {sample_dir}  (deliverable #2)")
    if sc.get("train", 0) == 0 or sc.get("test", 0) == 0:
        print("[warn] empty train or test split — check split config")


if __name__ == "__main__":
    main()

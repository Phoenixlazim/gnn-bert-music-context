"""
labels.py
=========
Label construction + the leakage-controlled vocabulary split.

This module is where our methodological contribution lives, so it prints a
full audit of every routing decision. Keep that output — it goes in the report.

GTZAN   -> single-label genre from the parent folder name.
MTT     -> multi-label from annotations_final.csv (tab-separated).
           Top-K tags by frequency are routed into:
               TEXT   tags  -> concatenated into a string, fed to BERT
               TARGET tags  -> the y vector the model must predict
           In `naive` mode ALL top-K tags go to BOTH sides. That is the leaky
           configuration we compare against.

Emits: data/splits/{dataset}_labels.json
    {
      "task": "multilabel",
      "target_tags": [...], "text_tags": [...], "dropped_tags": [...],
      "mode": "controlled",
      "tracks": { track_id: {"y": [0/1,...], "text": "guitar, drums, ..."} }
    }
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml


def _norm(t: str) -> str:
    return t.strip().lower().replace("_", " ")


def route_tags(tags: list[str], cfg: dict) -> tuple[list[str], list[str], list[str]]:
    """Split a tag vocabulary into (target, text, dropped).

    Rules, in order:
      1. matches a target keyword -> TARGET (conservative: target wins ties)
      2. matches a text keyword   -> TEXT
      3. otherwise                -> DROPPED (excluded from both sides)
    Matching is on whole normalised tag equality OR keyword-as-substring.
    """
    ts = cfg["tag_split"]
    tk = {_norm(k) for k in ts["target_keywords"]}
    xk = {_norm(k) for k in ts["text_keywords"]}
    target_exclude = {_norm(k) for k in ts.get("target_exclude", [])}

    target, text, dropped = [], [], []
    for t in tags:
        n = _norm(t)
        if n in target_exclude:
            dropped.append(t)
            continue
        if n in tk or any(k in n for k in tk):
            target.append(t)
        elif n in xk or any(k in n for k in xk):
            text.append(t)
        else:
            dropped.append(t)
    return target, text, dropped


# ----------------------------------------------------------------------
def build_gtzan(cfg: dict, feats_dir: Path) -> dict:
    genres = cfg["datasets"]["gtzan"]["genres"]
    g2i = {g: i for i, g in enumerate(genres)}
    tracks = {}
    for p in sorted(feats_dir.glob("*.npz")):
        genre = p.stem.split("__")[0]
        if genre not in g2i:
            continue
        y = [0] * len(genres)
        y[g2i[genre]] = 1
        # GTZAN has no text source. We use a neutral placeholder so the text
        # branch is structurally present but carries no label information —
        # this is exactly why GTZAN is used for Task 2 (GNN-only), not Task 3.
        tracks[p.stem] = {"y": y, "text": "a music recording"}
    return {"task": "multiclass", "target_tags": genres, "text_tags": [],
            "dropped_tags": [], "mode": "n/a", "tracks": tracks}


def build_mtt(cfg: dict, feats_dir: Path) -> dict:
    d = cfg["datasets"]["magnatagatune"]
    ann = Path(d["annotations"])
    if not ann.exists():
        sys.exit(f"[fatal] annotations not found: {ann.resolve()}")

    have = {p.stem for p in feats_dir.glob("*.npz")}
    if not have:
        sys.exit(f"[fatal] no extracted features in {feats_dir} — run audio_features.py first")

    with open(ann, newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if not rows:
        sys.exit("[fatal] annotations file parsed to 0 rows (wrong delimiter?)")

    tag_cols = [c for c in rows[0] if c not in ("clip_id", "mp3_path")]
    print(f"[mtt] {len(rows)} annotation rows, {len(tag_cols)} tag columns")

    # map annotation rows -> our track_id scheme (folder__stem)
    keep = {}
    for r in rows:
        mp3 = r.get("mp3_path", "").strip()
        if not mp3:
            continue
        parts = mp3.split("/")
        if len(parts) < 2:
            continue
        tid = f"{parts[0]}__{Path(parts[-1]).stem}"
        if tid in have:
            keep[tid] = r
    print(f"[mtt] matched {len(keep)}/{len(have)} extracted tracks to annotations")
    if len(keep) < 0.5 * len(have):
        print("[warn] low match rate — check that track_id scheme matches mp3_path")

    counts = Counter()
    for r in keep.values():
        for c in tag_cols:
            if r[c] == "1":
                counts[c] += 1

    mode = cfg["tag_split"]["mode"]
    ranked = [t for t, _ in counts.most_common() if counts[t] >= d["min_positives"]]
    print(f"[vocab] {len(ranked)}/{len(tag_cols)} tags clear "
          f"min_positives={d['min_positives']}")

    if mode == "naive":
        # The leaky baseline: one global top-K used as BOTH input and target.
        top = ranked[: d["n_target_tags"]]
        target, text, dropped = top, top, []
    else:
        # Route the FULL vocabulary, then take top-N of each pool separately.
        # A single global top-K starves the target pool — MTT's head is
        # overwhelmingly instrument/timbre tags. (Report this.)
        t_all, x_all, dropped = route_tags(ranked, cfg)
        target = t_all[: d["n_target_tags"]]
        text = x_all[: d["n_text_tags"]]
        dropped = dropped + t_all[d["n_target_tags"]:] + x_all[d["n_text_tags"]:]

    print(f"\n[split] mode={mode}")
    print(f"        TARGET ({len(target)}): {', '.join(target)}")
    print(f"        TEXT   ({len(text)}): {', '.join(text)}")
    print(f"        DROP   ({len(dropped)}): {', '.join(dropped) or '-'}")
    if mode == "controlled" and set(target) & set(text):
        print(f"[warn] overlap between TARGET and TEXT: {set(target) & set(text)}")
    if not target:
        sys.exit("[fatal] empty target vocabulary — widen target_keywords in config")
    if not text:
        sys.exit("[fatal] empty text vocabulary — widen text_keywords in config")

    tracks = {}
    for tid, r in keep.items():
        y = [1 if r[t] == "1" else 0 for t in target]
        pos_text = [t for t in text if r[t] == "1"]
        # empty text is legitimate (clip has no instrument tags) — keep it
        # neutral rather than dropping the track, so N stays comparable.
        tracks[tid] = {"y": y, "text": ", ".join(pos_text) if pos_text else "unlabelled audio"}

    pos = np.array([t["y"] for t in tracks.values()]).sum(0)
    print(f"\n[balance] positives per target tag: min={pos.min()} "
          f"median={int(np.median(pos))} max={pos.max()}  (n={len(tracks)})")
    if pos.min() < 20:
        print("[warn] some target tags have <20 positives — macro-F1 will be noisy. "
              "Consider lowering top_k_tags or raising subsample.")

    return {"task": "multilabel", "target_tags": target, "text_tags": text,
            "dropped_tags": dropped, "mode": mode, "tracks": tracks}


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["gtzan", "magnatagatune"])
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--mode", choices=["controlled", "naive"], default=None,
                    help="override tag_split.mode (use 'naive' for the leakage baseline)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.mode:
        cfg["tag_split"]["mode"] = args.mode

    feats = Path(cfg["paths"]["processed"]) / "feats" / args.dataset
    out_dir = Path(cfg["paths"]["splits"])
    out_dir.mkdir(parents=True, exist_ok=True)

    obj = build_gtzan(cfg, feats) if args.dataset == "gtzan" else build_mtt(cfg, feats)

    suffix = "" if obj["mode"] in ("controlled", "n/a") else f"_{obj['mode']}"
    out = out_dir / f"{args.dataset}{suffix}_labels.json"
    json.dump(obj, open(out, "w"), indent=1)
    print(f"\n[done] {len(obj['tracks'])} tracks -> {out}")


if __name__ == "__main__":
    main()

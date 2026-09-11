"""
audio_features.py
=================
Segment-level feature extraction for the GNN-BERT music context project.

Design notes (defend these in the report):
  * Fixed 1.0 s windows, NOT beat-synchronous. Beat tracking fails silently on
    ambient / rubato / spoken content and the fallback logic costs more than it
    buys at this scale. Fixed windows are deterministic and reproducible.
  * Frame-level features are computed ONCE per track, then aggregated into
    windows in frame-index space. Slicing the waveform per window and
    recomputing STFTs is ~5x slower for identical output.
  * chroma and mfcc are cached separately (in addition to being folded into the
    node feature matrix) because graph_builder needs them for the harmonic and
    timbral kNN relations.

Output per track: {processed}/feats/{dataset}/{track_id}.npz
    x       [N, D]  node feature matrix (z-scored per track)
    chroma  [N, 12] window-mean chroma   -> harmonic kNN
    mfcc    [N, 13] window-mean MFCC     -> timbral kNN
    track_id, sr, window_sec

Optionally: {processed}/melspec/{dataset}/{track_id}.npy  (CNN baseline B2)

Usage
-----
    python src/audio_features.py --dataset gtzan
    python src/audio_features.py --dataset magnatagatune --limit 4000
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import warnings
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import yaml

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# librosa is imported inside the worker to keep fork() cheap on Linux.


# ----------------------------------------------------------------------
# Feature layout — keep in sync with FEATURE_NAMES for the report figure
# ----------------------------------------------------------------------
FEATURE_BLOCKS = [
    ("chroma_mean", 12),
    ("chroma_std", 12),
    ("mfcc_mean", 13),
    ("mfcc_std", 13),
    ("contrast_mean", 7),
    ("rms_mean", 1),
    ("rms_std", 1),
    ("zcr_mean", 1),
    ("centroid_mean", 1),
    ("onset_density", 1),
    ("position", 1),
]
FEATURE_DIM = sum(n for _, n in FEATURE_BLOCKS)  # 63


@dataclass
class Cfg:
    sr: int
    hop_length: int
    n_fft: int
    n_mels: int
    n_mfcc: int
    n_chroma: int
    window_sec: float
    stride_sec: float
    min_nodes: int
    max_nodes: int
    min_file_bytes: int
    min_rms: float
    save_melspec: bool


def load_cfg(path: str) -> tuple[Cfg, dict]:
    with open(path) as f:
        raw = yaml.safe_load(f)
    a = raw["audio"]
    return Cfg(**{k: a[k] for k in Cfg.__dataclass_fields__}), raw


# ----------------------------------------------------------------------
# Core extraction (runs in worker processes)
# ----------------------------------------------------------------------
_CFG: Cfg | None = None
_OUT_FEATS: Path | None = None
_OUT_MEL: Path | None = None


def _init_worker(cfg: Cfg, out_feats: str, out_mel: str):
    global _CFG, _OUT_FEATS, _OUT_MEL
    _CFG = cfg
    _OUT_FEATS = Path(out_feats)
    _OUT_MEL = Path(out_mel)


def _window_bounds(n_frames: int, frames_per_win: int, frames_per_stride: int,
                   max_nodes: int) -> list[tuple[int, int]]:
    bounds = []
    start = 0
    while start + frames_per_win <= n_frames and len(bounds) < max_nodes:
        bounds.append((start, start + frames_per_win))
        start += frames_per_stride
    # keep a trailing partial window if it is at least half full and we have room
    if len(bounds) < max_nodes and start < n_frames:
        if (n_frames - start) >= frames_per_win // 2:
            bounds.append((start, n_frames))
    return bounds


def extract_one(job: tuple[str, str]) -> dict:
    """Returns a status dict. Never raises — failures are logged, not fatal."""
    path, track_id = job
    cfg = _CFG
    t0 = time.time()

    try:
        # --- cheap rejects before touching the decoder ---------------------
        try:
            size = os.path.getsize(path)
        except OSError as e:
            return {"track_id": track_id, "ok": False, "reason": f"stat:{e}"}
        if size < cfg.min_file_bytes:
            return {"track_id": track_id, "ok": False, "reason": f"too_small:{size}B"}

        import librosa  # noqa: WPS433 (deliberate: worker-local import)

        y, sr = librosa.load(path, sr=cfg.sr, mono=True)
        if y.size < cfg.sr:  # < 1 second of audio
            return {"track_id": track_id, "ok": False, "reason": "audio_too_short"}
        if float(np.sqrt(np.mean(y ** 2))) < cfg.min_rms:
            return {"track_id": track_id, "ok": False, "reason": "silent"}

        # --- frame-level features, computed once ---------------------------
        S = np.abs(librosa.stft(y, n_fft=cfg.n_fft, hop_length=cfg.hop_length))
        mel = librosa.feature.melspectrogram(
            S=S ** 2, sr=sr, n_mels=cfg.n_mels)
        logmel = librosa.power_to_db(mel, ref=np.max)

        chroma = librosa.feature.chroma_stft(
            S=S, sr=sr, n_chroma=cfg.n_chroma)                    # [12, F]
        mfcc = librosa.feature.mfcc(
            S=logmel, n_mfcc=cfg.n_mfcc)                          # [13, F]
        contrast = librosa.feature.spectral_contrast(
            S=S, sr=sr)                                           # [7, F]
        rms = librosa.feature.rms(S=S)[0]                         # [F]
        zcr = librosa.feature.zero_crossing_rate(
            y, frame_length=cfg.n_fft, hop_length=cfg.hop_length)[0]
        centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
        onset_env = librosa.onset.onset_strength(S=logmel, sr=sr)

        F = min(chroma.shape[1], mfcc.shape[1], contrast.shape[1],
                rms.shape[0], zcr.shape[0], centroid.shape[0], onset_env.shape[0])

        fps = sr / cfg.hop_length
        fpw = max(1, int(round(cfg.window_sec * fps)))
        fps_stride = max(1, int(round(cfg.stride_sec * fps)))
        bounds = _window_bounds(F, fpw, fps_stride, cfg.max_nodes)

        if len(bounds) < cfg.min_nodes:
            return {"track_id": track_id, "ok": False,
                    "reason": f"too_few_windows:{len(bounds)}"}

        N = len(bounds)
        chroma_w = np.zeros((N, cfg.n_chroma), dtype=np.float32)
        mfcc_w = np.zeros((N, cfg.n_mfcc), dtype=np.float32)
        rows = np.zeros((N, FEATURE_DIM), dtype=np.float32)

        for i, (a, b) in enumerate(bounds):
            c_m, c_s = chroma[:, a:b].mean(1), chroma[:, a:b].std(1)
            m_m, m_s = mfcc[:, a:b].mean(1), mfcc[:, a:b].std(1)
            ct_m = contrast[:, a:b].mean(1)
            r_m, r_s = rms[a:b].mean(), rms[a:b].std()
            z_m = zcr[a:b].mean()
            ce_m = centroid[a:b].mean()
            on = onset_env[a:b].mean()
            pos = i / max(1, N - 1)

            chroma_w[i] = c_m
            mfcc_w[i] = m_m
            rows[i] = np.concatenate([
                c_m, c_s, m_m, m_s, ct_m,
                [r_m, r_s, z_m, ce_m, on, pos],
            ]).astype(np.float32)

        # --- per-track z-score (position column excluded) ------------------
        x = rows.copy()
        mu, sd = x[:, :-1].mean(0), x[:, :-1].std(0)
        x[:, :-1] = (x[:, :-1] - mu) / (sd + 1e-6)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        np.savez_compressed(
            _OUT_FEATS / f"{track_id}.npz",
            x=x, chroma=chroma_w, mfcc=mfcc_w,
            track_id=track_id, sr=sr, window_sec=cfg.window_sec,
        )
        if cfg.save_melspec:
            np.save(_OUT_MEL / f"{track_id}.npy",
                    logmel[:, :int(round(30 * fps))].astype(np.float32))

        return {"track_id": track_id, "ok": True, "n_nodes": N,
                "secs": round(time.time() - t0, 2)}

    except Exception as e:  # noqa: BLE001 — one bad mp3 must not kill the pool
        return {"track_id": track_id, "ok": False,
                "reason": f"{type(e).__name__}:{str(e)[:120]}"}


# ----------------------------------------------------------------------
# Job discovery
# ----------------------------------------------------------------------
def collect_jobs(dataset: str, cfg_raw: dict, limit: int | None) -> list[tuple[str, str]]:
    d = cfg_raw["datasets"][dataset]
    root = Path(d["audio_dir"])
    if not root.exists():
        sys.exit(f"[fatal] audio dir not found: {root.resolve()}\n"
                 f"        check datasets.{dataset}.audio_dir in config.yaml")

    ext = d["ext"]
    files = sorted(root.rglob(f"*{ext}"))
    if not files:
        # GTZAN Kaggle mirror ships .wav, the original ships .au — catch it early.
        alt = {".au": ".wav", ".wav": ".au"}.get(ext)
        if alt and sorted(root.rglob(f"*{alt}")):
            sys.exit(f"[fatal] no *{ext} under {root}, but *{alt} files exist.\n"
                     f"        set datasets.{dataset}.ext: '{alt}' in config.yaml")
        sys.exit(f"[fatal] no *{ext} files under {root}")

    jobs = []
    for p in files:
        if dataset == "gtzan":
            tid = f"{p.parent.name}__{p.stem}"      # genre encoded in track_id
        else:
            tid = f"{p.parent.name}__{p.stem}"      # mtt: folder prefix 0..f
        jobs.append((str(p), tid))

    if limit is None:
        limit = d.get("subsample")
    if limit and len(jobs) > limit:
        rng = np.random.default_rng(cfg_raw["seed"])
        idx = rng.permutation(len(jobs))[:limit]
        jobs = [jobs[i] for i in sorted(idx)]
    return jobs


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["gtzan", "magnatagatune"])
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg, raw = load_cfg(args.config)
    proc = Path(raw["paths"]["processed"])
    out_feats = proc / "feats" / args.dataset
    out_mel = proc / "melspec" / args.dataset
    out_feats.mkdir(parents=True, exist_ok=True)
    out_mel.mkdir(parents=True, exist_ok=True)

    jobs = collect_jobs(args.dataset, raw, args.limit)
    if not args.overwrite:
        done = {p.stem for p in out_feats.glob("*.npz")}
        before = len(jobs)
        jobs = [j for j in jobs if j[1] not in done]
        if before != len(jobs):
            print(f"[resume] skipping {before - len(jobs)} already-extracted tracks")

    nw = args.workers or raw["runtime"]["num_workers"] or max(1, cpu_count() - 1)
    print(f"[run] {args.dataset}: {len(jobs)} tracks | {nw} workers | dim={FEATURE_DIM}")

    t0 = time.time()
    results, ok = [], 0
    with Pool(nw, initializer=_init_worker,
              initargs=(cfg, str(out_feats), str(out_mel))) as pool:
        for i, r in enumerate(pool.imap_unordered(extract_one, jobs, chunksize=8), 1):
            results.append(r)
            ok += bool(r["ok"])
            if i % 100 == 0 or i == len(jobs):
                el = time.time() - t0
                eta = el / i * (len(jobs) - i)
                print(f"  {i}/{len(jobs)}  ok={ok}  "
                      f"elapsed={el/60:.1f}m  eta={eta/60:.1f}m", flush=True)

    # failure log — this is the artefact that justifies "no leakage, documented"
    log = proc / f"extract_log_{args.dataset}.csv"
    with open(log, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["track_id", "ok", "reason", "n_nodes", "secs"])
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, "") for k in w.fieldnames})

    fails = [r for r in results if not r["ok"]]
    ns = [r["n_nodes"] for r in results if r["ok"]]
    print(f"\n[done] ok={ok}/{len(jobs)} in {(time.time()-t0)/60:.1f} min")
    if ns:
        print(f"[nodes] mean={np.mean(ns):.1f} min={min(ns)} max={max(ns)}")
    if fails:
        from collections import Counter
        print(f"[fail] {len(fails)} tracks; top reasons:")
        for reason, c in Counter(r["reason"].split(":")[0] for r in fails).most_common(5):
            print(f"       {c:5d}  {reason}")
    print(f"[log ] {log}")


if __name__ == "__main__":
    main()

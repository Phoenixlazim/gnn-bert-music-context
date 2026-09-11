"""
smoke_test.py
=============
Validates the whole preprocessing pipeline against SYNTHETIC audio.
No dataset required — run this before the real extraction.

    python src\\smoke_test.py

It checks, in order:
  1. every librosa call used by audio_features.py (catches API changes)
  2. soundfile can write and read back audio
  3. extract_one() produces a well-formed .npz
  4. build_graph() produces a valid multi-relation graph
  5. torch can save/load the graph record

Exit code 0 = safe to run the real extraction.
"""

from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

FAILS = []


def check(name, fn):
    try:
        out = fn()
        print(f"  [ ok ] {name}")
        return out
    except Exception as e:
        print(f"  [FAIL] {name}\n         {type(e).__name__}: {e}")
        FAILS.append((name, traceback.format_exc()))
        return None


def main():
    print("=" * 62)
    print("1. imports and versions")
    print("=" * 62)
    import librosa, soundfile, torch, yaml
    print(f"  librosa    {librosa.__version__}")
    print(f"  soundfile  {soundfile.__version__}")
    print(f"  numpy      {np.__version__}")
    print(f"  torch      {torch.__version__}   cuda={torch.cuda.is_available()}")

    sr, dur = 16000, 10.0
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    # synthetic music-ish signal: chord + vibrato + noise burst
    y = (0.3 * np.sin(2 * np.pi * 220 * t)
         + 0.2 * np.sin(2 * np.pi * 277.18 * t)
         + 0.2 * np.sin(2 * np.pi * 329.63 * t)
         + 0.05 * np.random.default_rng(0).normal(size=t.size)).astype(np.float32)

    print("\n" + "=" * 62)
    print("2. librosa API surface (every call used by audio_features.py)")
    print("=" * 62)
    S = check("stft", lambda: np.abs(librosa.stft(y, n_fft=2048, hop_length=512)))
    if S is None:
        return finish()
    mel = check("feature.melspectrogram(S=...)",
                lambda: librosa.feature.melspectrogram(S=S ** 2, sr=sr, n_mels=128))
    logmel = check("power_to_db", lambda: librosa.power_to_db(mel, ref=np.max))
    check("feature.chroma_stft(n_chroma=)",
          lambda: librosa.feature.chroma_stft(S=S, sr=sr, n_chroma=12))
    check("feature.mfcc(S=logmel)",
          lambda: librosa.feature.mfcc(S=logmel, n_mfcc=13))
    check("feature.spectral_contrast",
          lambda: librosa.feature.spectral_contrast(S=S, sr=sr))
    check("feature.rms(S=)", lambda: librosa.feature.rms(S=S))
    check("feature.zero_crossing_rate",
          lambda: librosa.feature.zero_crossing_rate(y, frame_length=2048, hop_length=512))
    check("feature.spectral_centroid",
          lambda: librosa.feature.spectral_centroid(S=S, sr=sr))
    check("onset.onset_strength",
          lambda: librosa.onset.onset_strength(S=logmel, sr=sr))

    print("\n" + "=" * 62)
    print("3. round-trip through disk + extract_one()")
    print("=" * 62)
    import audio_features as AF
    from audio_features import Cfg, extract_one, _init_worker, FEATURE_DIM

    tmp = Path(tempfile.mkdtemp(prefix="cse425_smoke_"))
    wav = tmp / "synthetic.wav"
    check("soundfile.write", lambda: soundfile.write(wav, y, sr))
    check("librosa.load round-trip",
          lambda: librosa.load(str(wav), sr=sr, mono=True))

    cfg = Cfg(sr=sr, hop_length=512, n_fft=2048, n_mels=128, n_mfcc=13,
              n_chroma=12, window_sec=1.0, stride_sec=1.0, min_nodes=8,
              max_nodes=64, min_file_bytes=4096, min_rms=1e-4, save_melspec=True)
    feats_dir, mel_dir = tmp / "feats", tmp / "mel"
    feats_dir.mkdir(); mel_dir.mkdir()
    _init_worker(cfg, str(feats_dir), str(mel_dir))

    res = extract_one((str(wav), "synthetic"))
    if not res.get("ok"):
        print(f"  [FAIL] extract_one -> {res}")
        FAILS.append(("extract_one", str(res)))
        return finish()
    print(f"  [ ok ] extract_one: {res['n_nodes']} nodes in {res['secs']}s")

    d = np.load(feats_dir / "synthetic.npz", allow_pickle=True)
    x, chroma, mfcc = d["x"], d["chroma"], d["mfcc"]
    print(f"  [ ok ] x={x.shape} (expect [~10, {FEATURE_DIM}])  "
          f"chroma={chroma.shape}  mfcc={mfcc.shape}")
    assert x.shape[1] == FEATURE_DIM, f"feature dim {x.shape[1]} != {FEATURE_DIM}"
    assert np.isfinite(x).all(), "non-finite values in feature matrix"

    print("\n" + "=" * 62)
    print("4. graph construction")
    print("=" * 62)
    import graph_builder as GB
    gcfg = yaml.safe_load(open("config.yaml"))["graph"]
    xx, ei, et = GB.build_graph(feats_dir / "synthetic.npz", gcfg, False,
                                np.random.default_rng(0))
    n = xx.shape[0]
    print(f"  [ ok ] nodes={n}  edges={ei.shape[1]}  "
          f"relations={sorted(set(et.tolist()))}")
    assert ei.shape[1] > 0, "empty graph"
    assert ei.max() < n and ei.min() >= 0, "edge index out of range"
    assert not (ei[0] == ei[1]).any(), "self-loop present"
    for r in sorted(set(et.tolist())):
        print(f"         relation {r}: {(et == r).sum()} edges")

    _, eis, _ = GB.build_graph(feats_dir / "synthetic.npz", gcfg, True,
                               np.random.default_rng(0))
    od0 = np.bincount(ei[0], minlength=n); od1 = np.bincount(eis[0], minlength=n)
    print(f"  [ ok ] shuffle control preserves out-degree: "
          f"{np.array_equal(od0, od1)}")

    print("\n" + "=" * 62)
    print("5. torch save/load")
    print("=" * 62)
    import torch
    rec = {"x": torch.from_numpy(xx), "edge_index": torch.from_numpy(ei),
           "edge_type": torch.from_numpy(et), "y": torch.zeros(10),
           "split": "train", "text": "guitar, drums", "track_id": "synthetic"}
    torch.save(rec, tmp / "g.pt")
    back = torch.load(tmp / "g.pt", weights_only=False)
    print(f"  [ ok ] saved and reloaded: x={tuple(back['x'].shape)}")

    return finish()


def finish():
    print("\n" + "=" * 62)
    if FAILS:
        print(f"RESULT: {len(FAILS)} FAILURE(S) — do not start extraction yet")
        for name, tb in FAILS:
            print(f"\n--- {name} ---\n{tb}")
        return 1
    print("RESULT: ALL CHECKS PASSED — safe to run the real extraction")
    return 0


if __name__ == "__main__":
    sys.exit(main())

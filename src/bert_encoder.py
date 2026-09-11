"""
bert_encoder.py
===============
Runs DistilBERT ONCE over every track's text and caches the result.

This is the decision that makes the ablation ladder affordable: every later run
loads tensors instead of re-running a transformer, so A0-C1 become GNN-only
training.

Cached (fp16 to keep the file small):
    cls    [T, 768]      pooled [CLS] vector      -> concat fusion, InfoNCE
    tokens [T, L, 768]   per-token states         -> cross-attention
    mask   [T, L]        1 = real token
    ids    [T]           track_id order

IMPORTANT: the A0-C1 ladder uses these FROZEN pretrained features. The Task-1
fine-tuning deliverable is a separate run (train.py --preset BERT --finetune)
and its features are NOT reused here — feeding fine-tuned text into the fusion
ablations would pre-optimise the text branch for the target and make the GNN's
apparent contribution uninterpretable.

    python src\\bert_encoder.py --dataset magnatagatune
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["gtzan", "magnatagatune"])
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--labels", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    bc = cfg["bert"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    lab_path = Path(args.labels) if args.labels else \
        Path(cfg["paths"]["splits"]) / f"{args.dataset}_labels.json"
    lab = json.load(open(lab_path))
    ids = sorted(lab["tracks"])
    texts = [lab["tracks"][t]["text"] for t in ids]
    print(f"[in ] {len(ids)} tracks from {lab_path}")
    print(f"[ex ] sample text: {texts[0]!r}")

    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(bc["model_name"])
    model = AutoModel.from_pretrained(bc["model_name"]).to(dev).eval()

    # Pad to the corpus max rather than the config max — tag strings are short
    # and this cuts the cache size several-fold.
    lens = [len(tok(t)["input_ids"]) for t in texts[:2000]]
    L = min(bc["max_length"], max(8, max(lens)))
    print(f"[len] padding to {L} tokens (config max {bc['max_length']}, "
          f"observed max {max(lens)})")

    cls_all, tokens_all, mask_all = [], [], []
    with torch.no_grad():
        for i in range(0, len(texts), args.batch_size):
            chunk = texts[i:i + args.batch_size]
            enc = tok(chunk, padding="max_length", truncation=True,
                      max_length=L, return_tensors="pt").to(dev)
            out = model(**enc).last_hidden_state          # [B, L, 768]
            cls_all.append(out[:, 0].half().cpu())
            tokens_all.append(out.half().cpu())
            mask_all.append(enc["attention_mask"].cpu())
            if (i // args.batch_size) % 20 == 0:
                print(f"  {min(i + args.batch_size, len(texts))}/{len(texts)}",
                      flush=True)

    cache = {
        "ids": ids,
        "cls": torch.cat(cls_all),
        "tokens": torch.cat(tokens_all),
        "mask": torch.cat(mask_all),
        "model_name": bc["model_name"],
        "frozen": True,
        "max_length": L,
    }
    out_path = Path(args.out) if args.out else \
        Path(cfg["paths"]["processed"]) / f"bert_cache_{args.dataset}.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, out_path)
    mb = out_path.stat().st_size / 1e6
    print(f"\n[done] cls={tuple(cache['cls'].shape)} "
          f"tokens={tuple(cache['tokens'].shape)} -> {out_path} ({mb:.0f} MB)")


if __name__ == "__main__":
    main()

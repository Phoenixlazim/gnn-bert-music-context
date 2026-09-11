# Advanced Multimodal Music Context Understanding with Relation-Aware GNNs and BERT

**Course:** CSE425 - Neural Networks  
**Author:** Liza Akther And Lazim Muhammad Sharar
**Student ID:* 22201148 And 22101325 * 
**Institution:** BRAC University

## Overview

This project studies music-context prediction by combining structural audio representations with pretrained language representations. Audio clips are split into fixed 1 s segments, represented with 63-dimensional handcrafted acoustic features, and converted into multi-relation graphs with temporal, harmonic, and timbral edges. A relation-aware GAT encodes the graph, while frozen DistilBERT embeddings encode non-target textual context. The main MagnaTagATune (MTT) experiments use a leakage-controlled split in which text-input tags are disjoint from the 20 prediction targets.

The project implements the required BERT-only, GNN-only, early-concatenation, and cross-attention ablations, together with a log-mel CNN, a PCA+MLP baseline, a prior baseline, and a degree-preserving shuffled-graph control.

## Main results

Three-seed MTT results (mean ± sample standard deviation):

| Model | Seeds | Macro-F1 | Micro-F1 | AUC-PR | AUC-ROC |
|---|---:|---:|---:|---:|---:|
| B1 prior | 1 | 0.0527 | 0.1697 | 0.0436 | 0.5000 |
| B4 PCA+MLP | 3 | 0.1789 ± 0.0166 | 0.2051 ± 0.0021 | 0.1855 ± 0.0137 | 0.8020 ± 0.0005 |
| BERT only | 3 | 0.2370 ± 0.0163 | 0.2766 ± 0.0252 | 0.2240 ± 0.0070 | 0.8031 ± 0.0066 |
| A3 relation-GNN | 3 | 0.2157 ± 0.0089 | 0.2781 ± 0.0193 | 0.2076 ± 0.0080 | 0.8135 ± 0.0082 |
| C1 shuffled-GNN | 3 | 0.2058 ± 0.0079 | 0.2985 ± 0.0179 | 0.1927 ± 0.0085 | 0.8139 ± 0.0102 |
| B2 log-mel CNN | 3 | 0.2495 ± 0.0148 | 0.3094 ± 0.0101 | 0.2570 ± 0.0109 | 0.8588 ± 0.0071 |
| A5 cross-attention | 3 | 0.2807 ± 0.0142 | 0.3629 ± 0.0361 | 0.3047 ± 0.0194 | 0.8765 ± 0.0026 |
| A4 concat | 3 | 0.2824 ± 0.0106 | 0.3799 ± 0.0504 | 0.2904 ± 0.0051 | 0.8727 ± 0.0014 |

A4 gives the strongest thresholded F1, while A5 gives the strongest threshold-free ranking metrics. The shuffled-topology control is weaker than the real relation graph on MTT, but GTZAN produces a negative topology result, which is discussed directly in the report.

## Method summary

- **Audio:** 22,050 Hz mono; 1 s non-overlapping windows.
- **Node features (63-D):** chroma mean/std, MFCC mean/std, spectral contrast, RMS mean/std, zero-crossing rate, spectral centroid, onset density, and normalized position.
- **Graph relations:** temporal adjacency, harmonic chroma-kNN, and timbral MFCC-kNN.
- **Graph encoder:** two-layer relation-specific GAT with residual connections and mean + attention pooling.
- **Text encoder:** frozen `distilbert-base-uncased` representations.
- **Fusion:** A4 early concatenation and A5 text-token -> audio-segment cross-attention.
- **MTT targets:** 20 genre/mood/context tags.
- **MTT text coverage:** 2,651 / 3,998 clips (66.31%); remaining clips use a neutral `unlabelled audio` placeholder.
- **Model selection:** validation AUC-PR with early stopping.
- **Decision thresholds:** tuned per tag on validation only.

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── config.yaml
├── src/
│   ├── audio_features.py
│   ├── labels.py
│   ├── graph_builder.py
│   ├── bert_encoder.py
│   ├── gnn_model.py
│   ├── train.py
│   ├── baselines.py
│   ├── evaluate.py
│   ├── final_analysis.py
│   └── smoke_test.py
├── notebooks/
│   └── demo_context.ipynb
├── data/
│   ├── splits/
│   └── processed/
│       └── graph_samples/
├── results/
│   ├── metrics.json
│   └── evaluation/
└── report/
    ├── final_report.pdf
    ├── final_report.tex
    └── figures/
```

Large raw audio, full feature caches, full graph tensors, BERT caches, and virtual-environment files are intentionally not committed to the lightweight submission package. They are reproducible from the scripts below.

## Environment

Tested with Python 3.13, PyTorch + CUDA, PyTorch Geometric, librosa, transformers, scikit-learn, NumPy, and matplotlib.

```bash
python -m venv .venv
# Windows CMD
.venv\Scripts\activate.bat
pip install -r requirements.txt
```

## Preprocessing

Place the datasets under the paths configured in `config.yaml`, then run:

```bash
# GTZAN
python src/audio_features.py --dataset gtzan
python src/labels.py --dataset gtzan
python src/graph_builder.py --dataset gtzan
python src/graph_builder.py --dataset gtzan --shuffle

# MagnaTagATune
python src/audio_features.py --dataset magnatagatune
python src/labels.py --dataset magnatagatune
python src/graph_builder.py --dataset magnatagatune
python src/graph_builder.py --dataset magnatagatune --shuffle
python src/bert_encoder.py --dataset magnatagatune
```

Feature extraction is resumable and skips already cached tracks unless overwrite is requested.

## Training

Representative MTT runs:

```bash
python src/train.py --dataset magnatagatune --preset BERT --epochs 100 --patience 20 --seed 425 --run-name MTT_BERT_s425
python src/train.py --dataset magnatagatune --preset A3   --epochs 100 --patience 20 --seed 425 --run-name MTT_A3_s425
python src/train.py --dataset magnatagatune --preset A4   --epochs 100 --patience 20 --seed 425 --run-name MTT_A4_s425
python src/train.py --dataset magnatagatune --preset A5   --epochs 100 --patience 20 --seed 425 --run-name MTT_A5_s425
python src/train.py --dataset magnatagatune --preset C1   --epochs 100 --patience 20 --seed 425 --run-name MTT_C1_s425
```

Baselines:

```bash
python src/baselines.py --dataset magnatagatune --baseline B1 --seed 425 --run-name MTT_B1_s425
python src/baselines.py --dataset magnatagatune --baseline B2 --epochs 25 --seed 425 --run-name MTT_B2_s425
python src/baselines.py --dataset magnatagatune --baseline B4 --epochs 25 --seed 425 --run-name MTT_B4_s425
```

The final paper reports the primary models and learned baselines over seeds 425, 426, and 427.

## Evaluation and demo

```bash
python src/evaluate.py
python src/final_analysis.py
python -m notebook notebooks/demo_context.ipynb
```

`evaluate.py` aggregates repeated seeds and creates the main MTT comparison plots. `final_analysis.py` creates training curves, the held-out A4 t-SNE visualization, BERT examples, GTZAN ablations, and A5 qualitative case studies.

The demo notebook contains saved output for one held-out MTT example. A full rerun requires the locally generated BERT cache and trained checkpoint.

## Scope and limitations

- MTT is deterministically subsampled to 4,000 clips; 3,998 are usable after filtering.
- Only 20 prediction targets are evaluated.
- Text is short tag context rather than full lyrics or MusicCaps captions.
- DistilBERT is frozen in the reported fusion experiments.
- Task 4 / MusicCaps contrastive retrieval is not included; it is treated as the optional bonus extension.
- GTZAN is used as a structural diagnostic rather than the headline multimodal benchmark.

See `report/final_report.pdf` for the complete methodology, ablations, results, qualitative analysis, and references.

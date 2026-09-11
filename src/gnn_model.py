"""
gnn_model.py
============
The full ablation ladder in one module, driven by a preset name.

    A0   node features + mean pool, NO message passing
    A1   temporal edges only, GraphSAGE (shared weights)
    A2   all relations, GraphSAGE (shared weights)
    A3   all relations, relation-specific GAT   <- our structural contribution
    A4   A3 + concat fusion with BERT [CLS]
    A5   A3 + token->segment cross-attention    <- our fusion contribution
    C1   A3, run on degree-shuffled graphs (control; same model, other data)

Relation-specific message passing is implemented as one GATConv per relation,
summed with a root transform. This is R-GAT in substance and it exposes
per-relation attention weights directly, which is what the "attention mass per
edge type" figure needs.

Cross-attention direction is text-tokens -> segments (query = text). The
attention matrix [B, L, N] is exactly the phrase-grounding map used in the demo:
"where in this clip is the distorted guitar?"

Selftest (no data needed):
    python src\\gnn_model.py --selftest
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGEConv
from torch_geometric.utils import to_dense_batch

N_RELATIONS = 3          # 0=temporal 1=harmonic 2=timbral
REL_NAMES = ["temporal", "harmonic", "timbral"]


# ----------------------------------------------------------------------
@dataclass
class ModelCfg:
    in_dim: int = 63
    text_dim: int = 768
    hidden: int = 128
    layers: int = 2
    heads: int = 4
    dropout: float = 0.2
    conv: str = "rgat"                       # none | sage | rgat
    relations: list[int] = field(default_factory=lambda: [0, 1, 2])
    readout: str = "mean_attn"               # mean | mean_attn
    fusion: str = "xattn"                    # none | concat | xattn
    n_classes: int = 20


ABLATIONS: dict[str, dict] = {
    "A0": dict(conv="none", relations=[0, 1, 2], fusion="none", readout="mean"),
    "A1": dict(conv="sage", relations=[0],       fusion="none"),
    "A2": dict(conv="sage", relations=[0, 1, 2], fusion="none"),
    "A3": dict(conv="rgat", relations=[0, 1, 2], fusion="none"),
    "A4": dict(conv="rgat", relations=[0, 1, 2], fusion="concat"),
    "A5": dict(conv="rgat", relations=[0, 1, 2], fusion="xattn"),
    "C1": dict(conv="rgat", relations=[0, 1, 2], fusion="none"),  # shuffled data
    # Task-1 deliverable: text only, no graph at all.
    "BERT": dict(conv="none", relations=[], fusion="text_only", readout="mean"),
}


def make_cfg(preset: str, **over) -> ModelCfg:
    if preset not in ABLATIONS:
        raise KeyError(f"unknown preset {preset}; have {list(ABLATIONS)}")
    return ModelCfg(**{**ABLATIONS[preset], **over})


# ----------------------------------------------------------------------
class RelationGAT(nn.Module):
    """One GATConv per relation + root transform, summed.

    Returns per-relation attention weights so we can report which edge type
    carries which tag family.
    """

    def __init__(self, in_dim, out_dim, relations, heads, dropout):
        super().__init__()
        assert out_dim % heads == 0, "hidden must be divisible by heads"
        self.relations = list(relations)
        self.convs = nn.ModuleDict({
            str(r): GATConv(in_dim, out_dim // heads, heads=heads,
                            dropout=dropout, add_self_loops=False)
            for r in self.relations
        })
        self.root = nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index, edge_type, want_attn=False):
        out = self.root(x)
        attn = {}
        for r in self.relations:
            m = edge_type == r
            ei = edge_index[:, m]
            if ei.numel() == 0:
                continue
            if want_attn:
                h, (ei_a, alpha) = self.convs[str(r)](
                    x, ei, return_attention_weights=True)
                attn[r] = (ei_a.detach(), alpha.detach())
            else:
                h = self.convs[str(r)](x, ei)
            out = out + h
        return out, attn


class SharedSAGE(nn.Module):
    """GraphSAGE over the union of the selected relations (no typing)."""

    def __init__(self, in_dim, out_dim, relations):
        super().__init__()
        self.relations = list(relations)
        self.conv = SAGEConv(in_dim, out_dim)

    def forward(self, x, edge_index, edge_type, want_attn=False):
        if self.relations:
            m = torch.isin(edge_type, torch.tensor(
                self.relations, device=edge_type.device))
            edge_index = edge_index[:, m]
        return self.conv(x, edge_index), {}


class AttnPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, h_dense, mask):
        s = self.score(h_dense).squeeze(-1)
        s = s.masked_fill(~mask, float("-inf"))
        w = torch.softmax(s, dim=1).unsqueeze(-1)
        return (h_dense * w).sum(1)


# ----------------------------------------------------------------------
class ContextModel(nn.Module):
    def __init__(self, cfg: ModelCfg):
        super().__init__()
        self.cfg = cfg
        H = cfg.hidden

        self.in_proj = nn.Sequential(
            nn.Linear(cfg.in_dim, H), nn.ReLU(), nn.Dropout(cfg.dropout))

        self.convs = nn.ModuleList()
        if cfg.conv != "none":
            for _ in range(cfg.layers):
                if cfg.conv == "rgat":
                    self.convs.append(
                        RelationGAT(H, H, cfg.relations, cfg.heads, cfg.dropout))
                elif cfg.conv == "sage":
                    self.convs.append(SharedSAGE(H, H, cfg.relations))
                else:
                    raise ValueError(cfg.conv)

        self.norms = nn.ModuleList(nn.LayerNorm(H) for _ in self.convs)
        self.attn_pool = AttnPool(H) if cfg.readout == "mean_attn" else None

        g_dim = H * (2 if cfg.readout == "mean_attn" else 1)

        self.text_proj = None
        self.xattn = None
        z_dim = g_dim
        if cfg.fusion == "concat":
            self.text_proj = nn.Linear(cfg.text_dim, H)
            z_dim = g_dim + H
        elif cfg.fusion == "xattn":
            self.text_proj = nn.Linear(cfg.text_dim, H)
            self.xattn = nn.MultiheadAttention(
                H, cfg.heads, dropout=cfg.dropout, batch_first=True)
            z_dim = g_dim + H * 2            # + pooled text + attended context
        elif cfg.fusion == "text_only":
            self.text_proj = nn.Linear(cfg.text_dim, H)
            z_dim = H

        self.head = nn.Sequential(
            nn.LayerNorm(z_dim), nn.Dropout(cfg.dropout),
            nn.Linear(z_dim, H), nn.ReLU(),
            nn.Linear(H, cfg.n_classes))

        self.proj_g = nn.Linear(g_dim, H)     # for the optional InfoNCE head
        self.proj_t = nn.Linear(cfg.text_dim, H)

    # ------------------------------------------------------------------
    def encode_graph(self, x, edge_index, edge_type, batch, want_attn=False):
        h = self.in_proj(x)
        attn_all = []
        for conv, norm in zip(self.convs, self.norms):
            h_new, a = conv(h, edge_index, edge_type, want_attn)
            h = norm(F.relu(h_new)) + h        # residual
            attn_all.append(a)

        h_dense, mask = to_dense_batch(h, batch)
        denom = mask.sum(1, keepdim=True).clamp(min=1)
        g = (h_dense * mask.unsqueeze(-1)).sum(1) / denom
        if self.attn_pool is not None:
            g = torch.cat([g, self.attn_pool(h_dense, mask)], dim=-1)
        return g, h_dense, mask, attn_all

    def forward(self, batch_obj, want_attn=False):
        b = batch_obj
        cfg = self.cfg

        if cfg.fusion == "text_only":
            t = self.text_proj(b.text_cls)
            return {"logits": self.head(t), "z": t}

        g, h_dense, mask, attn_all = self.encode_graph(
            b.x, b.edge_index, b.edge_type, b.batch, want_attn)

        out = {"g": g, "rel_attn": attn_all}

        if cfg.fusion == "none":
            z = g
        elif cfg.fusion == "concat":
            z = torch.cat([g, self.text_proj(b.text_cls)], dim=-1)
        else:  # xattn — text tokens query the segments
            tok = self.text_proj(b.text_tokens)                # [B, L, H]
            tmask = b.text_mask.bool()                         # [B, L]
            ctx, w = self.xattn(tok, h_dense, h_dense,
                                key_padding_mask=~mask,
                                need_weights=True, average_attn_weights=True)
            tm = tmask.unsqueeze(-1)
            pooled_ctx = (ctx * tm).sum(1) / tm.sum(1).clamp(min=1)
            pooled_tok = (tok * tm).sum(1) / tm.sum(1).clamp(min=1)
            z = torch.cat([g, pooled_tok, pooled_ctx], dim=-1)
            out["xattn_weights"] = w                           # [B, L, N] demo map

        out["z"] = z
        out["logits"] = self.head(z)
        if hasattr(b, "text_cls"):
            out["emb_g"] = F.normalize(self.proj_g(g), dim=-1)
            out["emb_t"] = F.normalize(self.proj_t(b.text_cls), dim=-1)
        return out


def info_nce(eg, et, temperature=0.07):
    logits = eg @ et.t() / temperature
    tgt = torch.arange(eg.size(0), device=eg.device)
    return 0.5 * (F.cross_entropy(logits, tgt) + F.cross_entropy(logits.t(), tgt))


# ----------------------------------------------------------------------
def _selftest():
    from torch_geometric.data import Batch, Data

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    K, L, TD = 20, 12, 768

    def fake(n):
        e, t = [], []
        for i in range(n - 1):
            e += [[i, i + 1], [i + 1, i]]; t += [0, 0]
        for _ in range(3 * n):
            a, b = torch.randint(0, n, (2,)).tolist()
            if a != b:
                e.append([a, b]); t.append(int(torch.randint(1, 3, (1,))))
        d = Data(x=torch.randn(n, 63),
                 edge_index=torch.tensor(e).t().contiguous(),
                 edge_type=torch.tensor(t))
        d.y = (torch.rand(1, K) > 0.8).float()
        d.text_cls = torch.randn(1, TD)
        d.text_tokens = torch.randn(1, L, TD)
        d.text_mask = torch.ones(1, L)
        return d

    batch = Batch.from_data_list([fake(int(n)) for n in
                                  torch.randint(9, 31, (8,))]).to(dev)
    print(f"device={dev}  batch: {batch.num_graphs} graphs, "
          f"{batch.x.size(0)} nodes, {batch.edge_index.size(1)} edges\n")

    ok = True
    for name in ABLATIONS:
        try:
            cfg = make_cfg(name, n_classes=K)
            m = ContextModel(cfg).to(dev)
            out = m(batch, want_attn=(name == "A3"))
            loss = F.binary_cross_entropy_with_logits(out["logits"], batch.y)
            if "emb_g" in out:
                loss = loss + 0.1 * info_nce(out["emb_g"], out["emb_t"])
            loss.backward()
            gnorm = sum(p.grad.abs().sum().item()
                        for p in m.parameters() if p.grad is not None)
            nparam = sum(p.numel() for p in m.parameters())
            extra = ""
            if name == "A3":
                nrel = sum(len(a) for a in out["rel_attn"])
                extra = f" rel_attn_layers={nrel}"
            if "xattn_weights" in out:
                extra = f" xattn={tuple(out['xattn_weights'].shape)}"
            assert out["logits"].shape == (8, K)
            assert gnorm > 0, "zero gradient"
            print(f"  [ ok ] {name:5s} params={nparam/1e3:7.1f}k  "
                  f"loss={loss.item():.4f}  z={tuple(out['z'].shape)}{extra}")
        except Exception as e:
            ok = False
            print(f"  [FAIL] {name:5s} {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()

    print("\n" + ("ALL PRESETS PASSED — forward + backward OK" if ok
                  else "FAILURES ABOVE"))
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(_selftest())
    print("nothing to do; try --selftest")

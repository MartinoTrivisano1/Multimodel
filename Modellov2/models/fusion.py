"""
models/fusion.py
================
Modulo di fusione multimodale MADDi-style con patch tokens.

Aggiornamento rispetto alla versione precedente:
  - Gli encoder ora producono sequenze di token invece di singoli vettori
  - TabularEncoder → (B, N_feat, D)   N_feat = N_tab + N_vol
  - Encoder2D      → (B, 49, D)       49 patch spaziali
  - Encoder3D      → (B, 216, D)      216 patch cerebrali

La cross-attention ora produce pesi informativi:
  - tab↔3d: matrice (N_feat × 216) → quale patch cerebrale per ogni feature clinica?
  - tab↔2d: matrice (N_feat × 49)  → quale patch slice per ogni biomarker?
  - 2d↔3d:  matrice (49 × 216)     → corrispondenza tra patch 2D e 3D

Questi pesi vengono esportati per la XAI e mappati sui volumi cerebrali.
"""

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import config as cfg


# ══════════════════════════════════════════════════════════════════════════════
# CROSS-MODAL ATTENTION UNIDIREZIONALE (su sequenze)
# ══════════════════════════════════════════════════════════════════════════════

class CrossModalAttention(nn.Module):
    """
    Cross-modal attention unidirezionale su sequenze di token.

    A come Query (seq_a token), B come Key/Value (seq_b token).
    Produce pesi (B, seq_a, seq_b) — matrice di importanza
    che per tab↔3d dice:
      "il token MMSE guarda queste 216 patch cerebrali con questi pesi"

    Input  : query (B, seq_a, D), key_value (B, seq_b, D)
    Output : out (B, seq_a, D), weights (B, seq_a, seq_b)
    """

    def __init__(
        self,
        embed_dim: int   = cfg.EMB_DIM,
        num_heads: int   = cfg.N_HEADS,
        dropout:   float = cfg.DROPOUT,
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm    = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query:     torch.Tensor,   # (B, seq_a, D)
        key_value: torch.Tensor,   # (B, seq_b, D)
        return_weights: bool = False,
    ):
        attn_out, attn_w = self.attn(query, key_value, key_value)
        out = self.norm(query + self.dropout(attn_out))
        if return_weights:
            return out, attn_w   # attn_w: (B, seq_a, seq_b)
        return out


# ══════════════════════════════════════════════════════════════════════════════
# CROSS-MODAL ATTENTION BIDIREZIONALE (su sequenze)
# ══════════════════════════════════════════════════════════════════════════════

class BiCrossModalAttention(nn.Module):
    """
    Cross-modal attention bidirezionale su sequenze:
      A→B: ogni token di A attende ai token di B
      B→A: ogni token di B attende ai token di A

    Per tab↔3d:
      tab→3d: matrice (N_feat × 216) — quale patch per quale feature
      3d→tab: matrice (216 × N_feat) — quale feature per quale patch

    I due output vengono aggregati tramite mean pooling e proiettati.
    """

    def __init__(
        self,
        embed_dim: int = cfg.EMB_DIM,
        num_heads: int = cfg.N_HEADS,
        dropout:   float = cfg.DROPOUT,
    ):
        super().__init__()
        self.a_to_b = CrossModalAttention(embed_dim, num_heads, dropout)
        self.b_to_a = CrossModalAttention(embed_dim, num_heads, dropout)
        self.proj   = nn.Linear(embed_dim * 2, embed_dim)
        self.norm   = nn.LayerNorm(embed_dim)

    def forward(
        self,
        feat_a: torch.Tensor,   # (B, seq_a, D)
        feat_b: torch.Tensor,   # (B, seq_b, D)
        return_weights: bool = False,
    ):
        if return_weights:
            out_ab, w_ab = self.a_to_b(feat_a, feat_b, return_weights=True)
            out_ba, w_ba = self.b_to_a(feat_b, feat_a, return_weights=True)
        else:
            out_ab = self.a_to_b(feat_a, feat_b)
            out_ba = self.b_to_a(feat_b, feat_a)
            w_ab = w_ba = None

        # pool su dim seq → (B, D) per ciascuna direzione
        out_ab_pool = out_ab.mean(dim=1)   # (B, D)
        out_ba_pool = out_ba.mean(dim=1)   # (B, D)

        fused = torch.cat([out_ab_pool, out_ba_pool], dim=-1)  # (B, 2D)
        out = self.norm(self.proj(fused))                     # (B, D)

        if return_weights:
            return out, w_ab, w_ba
        return out


# ══════════════════════════════════════════════════════════════════════════════
# FUSION MODULE DINAMICO
# ══════════════════════════════════════════════════════════════════════════════

class MultimodalFusion(nn.Module):
    """
    Modulo di fusione con patch tokens e cross-attention su sequenze.

    Pipeline:
      1. Self-attention intra-modale (su sequenze di token)
      2. Cross-modal attention bidirezionale sulle coppie attive
         → pesi informativi per XAI (matrici N_feat×216, ecc.)
      3. Media degli output delle coppie attive → (B, D)
      4. Proiezione finale

    I pesi attention ora sono matrici reali, non scalari 1.0.
    """

    def __init__(
        self,
        embed_dim: int   = cfg.EMB_DIM,
        num_heads: int   = cfg.N_HEADS,
        dropout:   float = cfg.DROPOUT,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # self-attention intra-modale per ogni modalità
        self.sa_tab = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.sa_2d = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.sa_3d = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout, batch_first=True, norm_first=True,
        )

        # cross-modal attention per ogni coppia
        self.cross_tab_2d = BiCrossModalAttention(embed_dim, num_heads, dropout)
        self.cross_tab_3d = BiCrossModalAttention(embed_dim, num_heads, dropout)
        self.cross_2d_3d = BiCrossModalAttention(embed_dim, num_heads, dropout)

        # proiezione finale
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        feat_tab: torch.Tensor,      # (B, N_feat, D)
        feat_2d:  torch.Tensor,      # (B, 49, D)
        feat_3d:  torch.Tensor,      # (B, 216, D)
        has_tabular: torch.Tensor,   # (B,) bool
        has_2d:      torch.Tensor,   # (B,) bool
        has_3d:      torch.Tensor,   # (B,) bool
        return_weights: bool = False,
    ):
        B = feat_tab.size(0)

        # ── self-attention intra-modale ────────────────────────────────────
        s_tab = self.sa_tab(feat_tab)   # (B, N_feat, D)
        s_2d  = self.sa_2d(feat_2d)    # (B, 49, D)
        s_3d  = self.sa_3d(feat_3d)    # (B, 216, D)

        # maschera modalità assenti
        mask_tab = has_tabular.float().view(B, 1, 1)
        mask_2d  = has_2d.float().view(B, 1, 1)
        mask_3d  = has_3d.float().view(B, 1, 1)
        s_tab = s_tab * mask_tab
        s_2d  = s_2d  * mask_2d
        s_3d  = s_3d  * mask_3d

        # ── cross-modal attention dinamica ────────────────────────────────
        pair_tab_2d = (has_tabular & has_2d).any()
        pair_tab_3d = (has_tabular & has_3d).any()
        pair_2d_3d  = (has_2d      & has_3d).any()

        cross_outputs = []
        weights = {"tab_2d": None, "tab_3d": None, "2d_3d": None}

        if pair_tab_2d:
            if return_weights:
                c, w_ab, w_ba = self.cross_tab_2d(s_tab, s_2d, return_weights=True)
                weights["tab_2d"] = (w_ab, w_ba)
                # w_ab: (B, N_feat, 49) — feature clinica → patch slice
                # w_ba: (B, 49, N_feat) — patch slice → feature clinica
            else:
                c = self.cross_tab_2d(s_tab, s_2d)
            pair_mask = (has_tabular & has_2d).float().view(B, 1)
            cross_outputs.append(c * pair_mask)

        if pair_tab_3d:
            if return_weights:
                c, w_ab, w_ba = self.cross_tab_3d(s_tab, s_3d, return_weights=True)
                weights["tab_3d"] = (w_ab, w_ba)
                # w_ab: (B, N_feat, 216) — feature clinica → patch cerebrale
                # w_ba: (B, 216, N_feat) — patch cerebrale → feature clinica
            else:
                c = self.cross_tab_3d(s_tab, s_3d)
            pair_mask = (has_tabular & has_3d).float().view(B, 1)
            cross_outputs.append(c * pair_mask)

        if pair_2d_3d:
            if return_weights:
                c, w_ab, w_ba = self.cross_2d_3d(s_2d, s_3d, return_weights=True)
                weights["2d_3d"] = (w_ab, w_ba)
                # w_ab: (B, 49, 216)  — patch 2D → patch 3D
                # w_ba: (B, 216, 49)  — patch 3D → patch 2D
            else:
                c = self.cross_2d_3d(s_2d, s_3d)
            pair_mask = (has_2d & has_3d).float().view(B, 1)
            cross_outputs.append(c * pair_mask)

        # ── aggregazione ──────────────────────────────────────────────────
        if len(cross_outputs) == 0:
            # fallback: pool su sequenza → mean di modalità attive
            active = []
            for s, m in [(s_tab, mask_tab), (s_2d, mask_2d), (s_3d, mask_3d)]:
                if m.any():
                    active.append(s.mean(dim=1))   # (B, D)
            fused = torch.stack(active, dim=0).mean(dim=0) if active else \
                    torch.zeros(B, self.embed_dim, device=feat_tab.device)
        else:
            fused = torch.stack(cross_outputs, dim=0).mean(dim=0)  # (B, D)

        out = self.output_proj(fused)

        if return_weights:
            return out, weights
        return out


# ══════════════════════════════════════════════════════════════════════════════
# TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    B = 2
    D = cfg.EMB_DIM
    N_FEAT = cfg.N_TABULAR_FEATURES + 1 + 21   # tab + vol
    dev = torch.device("cpu")

    fusion = MultimodalFusion(embed_dim=D, num_heads=cfg.N_HEADS).to(dev)
    fusion.eval()
    n = sum(p.numel() for p in fusion.parameters()) / 1e6
    print(f"MultimodalFusion params: {n:.3f}M")

    # sequenze patch
    feat_tab = torch.randn(B, N_FEAT, D)
    feat_2d  = torch.randn(B, 49,    D)
    feat_3d  = torch.randn(B, 216,   D)

    print("\n=== Scenario 1: tutti e 3 ===")
    has_t = torch.ones(B, dtype=torch.bool)
    has_2 = torch.ones(B, dtype=torch.bool)
    has_3 = torch.ones(B, dtype=torch.bool)
    with torch.no_grad():
        out, w = fusion(feat_tab, feat_2d, feat_3d, has_t, has_2, has_3, return_weights=True)
    assert out.shape == (B, D)
    print(f"  out: {tuple(out.shape)}")
    for k, v in w.items():
        if v is not None:
            w_ab, w_ba = v
            print(f"  {k}: w_ab={tuple(w_ab.shape)}  w_ba={tuple(w_ba.shape)}")

    w_ab = w["tab_3d"][0]
    assert not torch.allclose(w_ab, torch.ones_like(w_ab)), "Pesi ancora tutti 1.0!"
    print("  Pesi attention variabili (non tutti 1.0) ✓")

    print("\n=== Scenario 2: solo tab + 3D ===")
    with torch.no_grad():
        out, w = fusion(feat_tab, feat_2d, feat_3d,
                        torch.ones(B, dtype=torch.bool),
                        torch.zeros(B, dtype=torch.bool),
                        torch.ones(B, dtype=torch.bool),
                        return_weights=True)
    assert out.shape == (B, D)
    print(f"  out: {tuple(out.shape)}")
    print(f"  coppie attive: {[k for k,v in w.items() if v is not None]}")

    print("\n=== Scenario 3: batch misto ===")
    with torch.no_grad():
        out, w = fusion(feat_tab, feat_2d, feat_3d,
                        torch.ones(B, dtype=torch.bool),
                        torch.tensor([True, False]),
                        torch.tensor([False, True]),
                        return_weights=True)
    assert out.shape == (B, D)
    print(f"  out: {tuple(out.shape)}")
    print(f"  coppie attive: {[k for k,v in w.items() if v is not None]}")

    print("\n  Tutti i test OK")
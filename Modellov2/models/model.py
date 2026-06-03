"""
models/model.py
===============
Modello multimodale completo con patch tokens e backbone MONAI.

Aggiornamento: aggiunto freeze_backbone per ridurre memoria su MPS.
Con freeze_backbone=True i parametri trainabili scendono da 48M a ~5M.
"""

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import config as cfg
from models.encoders import TabularEncoder, Encoder2D, Encoder3D
from models.fusion   import MultimodalFusion


# ══════════════════════════════════════════════════════════════════════════════
# MOE ROUTER
# ══════════════════════════════════════════════════════════════════════════════

class MoERouter(nn.Module):
    def __init__(self, embed_dim=cfg.EMB_DIM, dropout=cfg.DROPOUT):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 3),
        )

    def forward(self, feat_tab, feat_2d, feat_3d,
                has_tabular, has_2d, has_3d):
        B = feat_tab.size(0)
        agg_tab = feat_tab.mean(dim=1)
        agg_2d  = feat_2d.mean(dim=1)
        agg_3d  = feat_3d.mean(dim=1)

        mask_tab = has_tabular.float().unsqueeze(1)
        mask_2d  = has_2d.float().unsqueeze(1)
        mask_3d  = has_3d.float().unsqueeze(1)

        n_active = (mask_tab + mask_2d + mask_3d).clamp(min=1.0)
        agg      = (agg_tab*mask_tab + agg_2d*mask_2d + agg_3d*mask_3d) / n_active
        scores   = self.gate(agg)

        avail  = torch.stack([has_tabular.float(),
                               has_2d.float(), has_3d.float()], dim=1)
        scores = scores.masked_fill(avail == 0, float("-inf"))
        gates  = F.softmax(scores, dim=-1)
        nan_m  = torch.isnan(gates).any(dim=-1, keepdim=True)
        gates  = torch.where(nan_m, torch.ones_like(gates)/3, gates)

        return (gates[:, 0:1].unsqueeze(-1),   # (B,1,1)
                gates[:, 1:2].unsqueeze(-1),
                gates[:, 2:3].unsqueeze(-1))


# ══════════════════════════════════════════════════════════════════════════════
# CLASSIFICATION HEAD
# ══════════════════════════════════════════════════════════════════════════════

class ClassificationHead(nn.Module):
    def __init__(self, embed_dim=cfg.EMB_DIM, hidden_dim=128,
                 n_classes=cfg.N_CLASSES, dropout=cfg.DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x): return self.net(x)


# ══════════════════════════════════════════════════════════════════════════════
# MODELLO COMPLETO
# ══════════════════════════════════════════════════════════════════════════════

class AlzheimerMultimodalNet(nn.Module):
    """
    Rete multimodale con patch tokens, backbone MONAI e cross-attention.

    freeze_backbone=True → backbone MONAI congelati, solo patch_proj
    e layers superiori vengono addestrati (~5M params invece di 48M).
    Consigliato per Apple Silicon / GPU con poca memoria.
    """

    def __init__(
        self,
        n_tabular:       int = cfg.N_TABULAR_FEATURES + 1,
        n_volumetric:    int = 21,
        embed_dim:       int = cfg.EMB_DIM,
        num_heads:       int = cfg.N_HEADS,
        n_classes:       int = cfg.N_CLASSES,
        dropout:         float = cfg.DROPOUT,
        pretrained_2d:   bool = True,
        use_monai:       bool = True,
        freeze_backbone: bool = True,
    ):
        super().__init__()

        self.enc_tab = TabularEncoder(
            n_tabular=n_tabular, n_volumetric=n_volumetric,
            embed_dim=embed_dim, dropout=dropout,
        )
        self.enc_2d = Encoder2D(
            embed_dim=embed_dim, dropout=dropout,
            pretrained=pretrained_2d,
            use_monai=use_monai,
            freeze_backbone=freeze_backbone,
        )
        self.enc_3d = Encoder3D(
            embed_dim=embed_dim, dropout=dropout,
            use_monai=use_monai,
            freeze_backbone=freeze_backbone,
        )
        self.router = MoERouter(embed_dim=embed_dim, dropout=dropout)
        self.fusion = MultimodalFusion(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout,
        )
        self.head = ClassificationHead(
            embed_dim=embed_dim, hidden_dim=embed_dim//2,
            n_classes=n_classes, dropout=dropout,
        )

    def forward(self, batch: dict, return_weights: bool = False):
        has_tab = batch["has_tabular"]
        has_2d = batch["has_2d"]
        has_3d = batch["has_3d"]

        feat_tab = self.enc_tab(batch["tabular"], batch["volumetric"])
        feat_2d = self.enc_2d(batch["slice_2d"])
        feat_3d = self.enc_3d(batch["volume_3d"])

        g_tab, g_2d, g_3d = self.router(
            feat_tab, feat_2d, feat_3d, has_tab, has_2d, has_3d,
        )
        feat_tab = feat_tab * g_tab
        feat_2d = feat_2d * g_2d
        feat_3d = feat_3d * g_3d

        if return_weights:
            fused, attn_weights = self.fusion(
                feat_tab, feat_2d, feat_3d,
                has_tab, has_2d, has_3d, return_weights=True,
            )
        else:
            fused = self.fusion(
                feat_tab, feat_2d, feat_3d,
                has_tab, has_2d, has_3d,
            )
            attn_weights = None

        logits = self.head(fused)
        if return_weights:
            return logits, attn_weights
        return logits

    def count_parameters(self) -> dict:
        """Conta parametri totali e trainabili per modulo."""
        modules = {
            "enc_tab": self.enc_tab,
            "enc_2d":  self.enc_2d,
            "enc_3d":  self.enc_3d,
            "router":  self.router,
            "fusion":  self.fusion,
            "head":    self.head,
        }
        counts = {}
        for name, mod in modules.items():
            total = sum(p.numel() for p in mod.parameters()) / 1e6
            trainable = sum(p.numel() for p in mod.parameters()
                            if p.requires_grad) / 1e6
            counts[name] = {"total": round(total, 3),
                            "trainable": round(trainable, 3)}
        total_all = sum(v["total"]     for v in counts.values())
        trainable_all = sum(v["trainable"] for v in counts.values())
        counts["TOTAL"] = round(total_all, 3)
        counts["TRAINABLE"] = round(trainable_all, 3)
        return counts


# ══════════════════════════════════════════════════════════════════════════════
# FACTORY
# ══════════════════════════════════════════════════════════════════════════════

def build_model(
    model_cfg:       str = "4",
    n_tabular:       int = cfg.N_TABULAR_FEATURES + 1,
    n_volumetric:    int = 21,
    pretrained_2d:   bool = True,
    device:          str = None,
    embed_dim:       int = None,
    num_heads:       int = None,
    dropout:         float = None,
    use_monai:       bool = True,
    freeze_backbone: bool = True,
) -> AlzheimerMultimodalNet:
    if device is None: device = cfg.DEVICE
    if embed_dim is None: embed_dim = cfg.EMB_DIM
    if num_heads is None: num_heads = cfg.N_HEADS
    if dropout is None: dropout = cfg.DROPOUT

    model = AlzheimerMultimodalNet(
        n_tabular=n_tabular, n_volumetric=n_volumetric,
        embed_dim=embed_dim, num_heads=num_heads,
        n_classes=cfg.N_CLASSES, dropout=dropout,
        pretrained_2d=pretrained_2d,
        use_monai=use_monai,
        freeze_backbone=freeze_backbone,
    )

    dev = torch.device(
        device if (device != "cuda" or torch.cuda.is_available()) else "cpu"
    )
    model = model.to(dev)

    params = model.count_parameters()
    print(f"\n[MODEL] {cfg.MODEL_CONFIGS[model_cfg]['name']}")
    print(f"  Device    : {dev}  |  freeze_backbone: {freeze_backbone}")
    print(f"  embed_dim : {embed_dim}  num_heads: {num_heads}  dropout: {dropout}")
    for k, v in params.items():
        if k not in ("TOTAL", "TRAINABLE"):
            print(f"  {k:<10}: {v['total']:.3f}M  "
                  f"(trainable: {v['trainable']:.3f}M)")
    print(f"  TOTAL     : {params['TOTAL']:.3f}M  "
          f"(trainable: {params['TRAINABLE']:.3f}M)")

    return model


# ══════════════════════════════════════════════════════════════════════════════
# TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    B = 2
    N_TAB = cfg.N_TABULAR_FEATURES + 1
    N_VOL = 21

    print("=" * 60)
    print("  AlzheimerMultimodalNet – MONAI + freeze – test")
    print("=" * 60)

    for freeze in [True, False]:
        model = build_model(
            model_cfg="4", n_tabular=N_TAB, n_volumetric=N_VOL,
            pretrained_2d=False, device="cpu",
            use_monai=True, freeze_backbone=freeze,
        )
        model.eval()
        batch = {
            "tabular":     torch.randn(B, N_TAB),
            "volumetric":  torch.randn(B, N_VOL),
            "slice_2d":    torch.randn(B, 3, *cfg.SIZE_2D),
            "volume_3d":   torch.randn(B, 1, *cfg.SIZE_3D),
            "has_tabular": torch.ones(B, dtype=torch.bool),
            "has_2d":      torch.ones(B, dtype=torch.bool),
            "has_3d":      torch.ones(B, dtype=torch.bool),
        }
        with torch.no_grad():
            logits, w = model(batch, return_weights=True)
        assert logits.shape == (B, cfg.N_CLASSES)
        active = {k: tuple(v[0].shape) for k,v in w.items() if v is not None}
        print(f"\n  freeze={freeze} → logits={tuple(logits.shape)}  "
              f"weights={active}  OK")

    print("\n  Tutti i test OK")
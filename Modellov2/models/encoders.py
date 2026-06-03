"""
models/encoders.py
==================
Encoder con patch tokens e backbone pre-addestrati MONAI.

  - TabularEncoder : ogni feature → token → (B, N_feat, D)
  - Encoder2D      : MONAI ResNet18 2D medico → 196 patch → (B, 196, D)
                     (input 224×224 → feature map 14×14)
  - Encoder3D      : MONAI ResNet18 3D medico → 216 patch → (B, 216, D)
                     (input 96×96×96 → feature map 6×6×6)

Entrambi i backbone sono pre-addestrati su immagini medicali —
molto superiori rispetto a ImageNet o CNN da zero per MRI cerebrali.

Fallback automatico alla CNN leggera se MONAI non è installato.
"""

import os
import sys

import torch
import torch.nn as nn
from torchvision import models

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import config as cfg

try:
    from monai.networks.nets import resnet18 as monai_resnet18
    MONAI_AVAILABLE = True
except ImportError:
    MONAI_AVAILABLE = False
    print("[WARN] MONAI non disponibile — uso backbone leggeri. "
          "Installa con: pip install monai")


# ══════════════════════════════════════════════════════════════════════════════
# UTILITÀ
# ══════════════════════════════════════════════════════════════════════════════

def _mlp_block(in_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.LayerNorm(out_dim),
        nn.GELU(),
        nn.Dropout(dropout),
    )


# ══════════════════════════════════════════════════════════════════════════════
# 1. TABULAR ENCODER
# ══════════════════════════════════════════════════════════════════════════════

class TabularEncoder(nn.Module):
    """
    Encoder tabellare con token per feature.
    Ogni feature scalare → embedding → sequenza di N_feat token.

    Input  : tabular (B, N_tab), volumetric (B, N_vol)
    Output : (B, N_tab+N_vol, embed_dim)
    """

    def __init__(
        self,
        n_tabular:    int,
        n_volumetric: int,
        embed_dim:    int = cfg.EMB_DIM,
        dropout:      float = cfg.DROPOUT,
    ):
        super().__init__()
        self.n_tabular = n_tabular
        self.n_volumetric = n_volumetric
        self.n_tokens = n_tabular + n_volumetric

        self.feature_proj = nn.Linear(1, embed_dim)
        self.pos_emb      = nn.Embedding(self.n_tokens, embed_dim)
        self.self_attn    = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=min(4, embed_dim // 64),
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, tabular: torch.Tensor, volumetric: torch.Tensor) -> torch.Tensor:
        x = torch.cat([tabular, volumetric], dim=-1)
        x_tokens = self.feature_proj(x.unsqueeze(-1))
        pos = torch.arange(self.n_tokens, device=x.device)
        x_tokens = x_tokens + self.pos_emb(pos).unsqueeze(0)
        return self.norm(self.self_attn(x_tokens))


# ══════════════════════════════════════════════════════════════════════════════
# 2. ENCODER 2D — MONAI ResNet18 medico → 196 patch
# ══════════════════════════════════════════════════════════════════════════════

class Encoder2D(nn.Module):
    """
    Encoder 2D con MONAI ResNet18 pre-addestrato su immagini medicali.

    Input  : (B, 3, 224, 224)   — 3 slice grayscale (ax, co, sa)
    Output : (B, 196, embed_dim) — 196 patch spaziali (14×14)

    MONAI ResNet18 layer4 su 224×224 → (B, 512, 14, 14) → 196 patch.
    Più ricco dei 49 patch precedenti — risoluzione spaziale più alta
    per la cross-attention con le feature cliniche.

    Fallback: ResNet18 ImageNet se MONAI non disponibile.
    """

    def __init__(
        self,
        embed_dim:       int = cfg.EMB_DIM,
        dropout:         float = cfg.DROPOUT,
        pretrained:      bool = True,
        use_monai:       bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.use_monai = use_monai and MONAI_AVAILABLE

        if self.use_monai:
            full = monai_resnet18(
                spatial_dims=2,
                n_input_channels=3,
                num_classes=2,
            )
            # backbone senza avgpool e fc → (B, 512, 14, 14)
            self.backbone = nn.Sequential(
                full.conv1, full.bn1, full.act, full.maxpool,
                full.layer1, full.layer2, full.layer3, full.layer4,
            )
            self.n_patches = 196   # 14×14
            backbone_out_ch = 512
            print("  [Encoder2D] Backbone: MONAI ResNet18 medico 2D (196 patch)")
        else:
            # fallback ResNet18 ImageNet
            weights  = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.resnet18(weights=weights)
            old_conv = backbone.conv1
            new_conv = nn.Conv2d(3, old_conv.out_channels,
                                 kernel_size=old_conv.kernel_size,
                                 stride=old_conv.stride,
                                 padding=old_conv.padding, bias=False)
            if pretrained:
                with torch.no_grad():
                    new_conv.weight.copy_(old_conv.weight)
            backbone.conv1 = new_conv
            self.backbone = nn.Sequential(*list(backbone.children())[:-2])
            self.n_patches = 49    # 7×7
            backbone_out_ch = 512
            print("  [Encoder2D] Backbone: ResNet18 ImageNet (49 patch)")

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
            print("  [Encoder2D] Backbone congelato")

        self.patch_proj = nn.Sequential(
            nn.Linear(backbone_out_ch, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pos_emb = nn.Embedding(self.n_patches, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)                                    # (B, 512, H, W)
        B, C, H, W = feat.shape
        patches = feat.permute(0, 2, 3, 1).reshape(B, H*W, C)        # (B, N, 512)
        out = self.patch_proj(patches)                            # (B, N, D)
        pos = torch.arange(H*W, device=x.device)
        return out + self.pos_emb(pos).unsqueeze(0)                   # (B, N, D)


# ══════════════════════════════════════════════════════════════════════════════
# 3. ENCODER 3D — MONAI ResNet18 medico → 216 patch
# ══════════════════════════════════════════════════════════════════════════════

class ConvBlock3D(nn.Module):
    """Fallback CNN 3D leggera."""
    def __init__(self, in_ch, out_ch, pool=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch), nn.GELU(),
        )
        self.pool = nn.MaxPool3d(2) if pool else nn.Identity()
    def forward(self, x): return self.pool(self.conv(x))


class Encoder3D(nn.Module):
    """
    Encoder 3D con MONAI ResNet18 pre-addestrato su MRI medicali.

    Input  : (B, 1, 96, 96, 96)
    Output : (B, 216, embed_dim) — 216 patch cerebrali (6×6×6)

    Fallback: CNN 3D leggera se MONAI non disponibile.
    """

    def __init__(
        self,
        embed_dim:       int   = cfg.EMB_DIM,
        dropout:         float = cfg.DROPOUT,
        use_monai:       bool  = True,
        freeze_backbone: bool  = False,
    ):
        super().__init__()
        self.use_monai = use_monai and MONAI_AVAILABLE

        if self.use_monai:
            full = monai_resnet18(
                spatial_dims=3,
                n_input_channels=1,
                num_classes=2,
            )
            self.backbone = nn.Sequential(
                full.conv1, full.bn1, full.act, full.maxpool,
                full.layer1, full.layer2, full.layer3, full.layer4,
            )
            backbone_out_ch = 512
            print("  [Encoder3D] Backbone: MONAI ResNet18 medico 3D (216 patch)")
        else:
            self.backbone = nn.Sequential(
                ConvBlock3D(1,  16,  pool=True),
                ConvBlock3D(16, 32,  pool=True),
                ConvBlock3D(32, 64,  pool=True),
                ConvBlock3D(64, 128, pool=True),
            )
            backbone_out_ch = 128
            print("  [Encoder3D] Backbone: CNN leggera (216 patch)")

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
            print("  [Encoder3D] Backbone congelato")

        self.patch_proj = nn.Sequential(
            nn.Linear(backbone_out_ch, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pos_emb = nn.Embedding(216, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)                                    # (B, C, 6, 6, 6)
        B, C, D, H, W = feat.shape
        patches = feat.permute(0, 2, 3, 4, 1).reshape(B, D*H*W, C)  # (B, 216, C)
        out = self.patch_proj(patches)                            # (B, 216, D)
        pos = torch.arange(D*H*W, device=x.device)
        return out + self.pos_emb(pos).unsqueeze(0)


# ══════════════════════════════════════════════════════════════════════════════
# FACTORY
# ══════════════════════════════════════════════════════════════════════════════

def build_encoders(
    n_tabular:       int,
    n_volumetric:    int,
    embed_dim:       int   = cfg.EMB_DIM,
    dropout:         float = cfg.DROPOUT,
    pretrained_2d:   bool  = True,
    use_monai:       bool  = True,
    freeze_backbone: bool  = False,
) -> dict:
    return {
        "tabular":   TabularEncoder(n_tabular, n_volumetric, embed_dim, dropout),
        "encoder2d": Encoder2D(embed_dim, dropout, pretrained_2d,
                               use_monai, freeze_backbone),
        "encoder3d": Encoder3D(embed_dim, dropout, use_monai, freeze_backbone),
    }


# ══════════════════════════════════════════════════════════════════════════════
# TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    B = 2
    N_TAB = cfg.N_TABULAR_FEATURES + 1
    N_VOL = 21
    EMB = cfg.EMB_DIM

    print("=" * 60)
    print(f"  Encoders MONAI  B={B}  EMB={EMB}")
    print(f"  MONAI: {MONAI_AVAILABLE}")
    print("=" * 60)

    encoders = build_encoders(N_TAB, N_VOL, EMB,
                               pretrained_2d=False,
                               use_monai=MONAI_AVAILABLE,
                               freeze_backbone=False)

    enc_tab = encoders["tabular"].train()
    out_tab = enc_tab(torch.randn(B, N_TAB), torch.randn(B, N_VOL))
    n_tok   = N_TAB + N_VOL
    assert out_tab.shape == (B, n_tok, EMB)
    n = sum(p.numel() for p in enc_tab.parameters())/1e6
    print(f"\n  TabularEncoder → {tuple(out_tab.shape)}  {n:.3f}M  OK")

    enc2d  = encoders["encoder2d"].train()
    out_2d = enc2d(torch.randn(B, 3, *cfg.SIZE_2D))
    n2d    = enc2d.n_patches
    assert out_2d.shape == (B, n2d, EMB)
    n = sum(p.numel() for p in enc2d.parameters())/1e6
    print(f"  Encoder2D      → {tuple(out_2d.shape)}  {n:.3f}M  OK")

    enc3d  = encoders["encoder3d"].train()
    out_3d = enc3d(torch.randn(B, 1, *cfg.SIZE_3D))
    assert out_3d.shape == (B, 216, EMB)
    n = sum(p.numel() for p in enc3d.parameters())/1e6
    print(f"  Encoder3D      → {tuple(out_3d.shape)}  {n:.3f}M  OK")

    total = sum(
        sum(p.numel() for p in e.parameters())
        for e in encoders.values()
    ) / 1e6
    print(f"\n  Totale encoder  : {total:.3f}M params")
    print("\n  Tutti i test OK")
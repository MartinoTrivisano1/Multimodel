"""
Early Fusion — Modello Completo Multimodale (v1)
================================================
Architettura originale:
    Encoder 2D  (CNN leggera)    → e_2d  (256,)
    Encoder 3D  (CNN 3D leggera) → e_3d  (256,)
    Encoder Tab (MLP)            → e_tab (256,)
         ↓
    concat([e_2d, e_3d, e_tab])  → (768,)
         ↓
    Classificatore MLP           → logits (3,)
         ↓
    CN / MCI / AD
"""

import torch
import torch.nn as nn


# ── Encoder 2D ────────────────────────────────────────────────────────────────

class Encoder2D(nn.Module):
    """
    CNN 2D per slice MRI (1, 224, 224).
    Output: embedding (emb_dim,)
    """

    def __init__(self, emb_dim: int = 256):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),          # 112×112

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),          # 56×56

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),          # 28×28

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # 1×1
        )

        self.fc = nn.Linear(256, emb_dim)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ── Encoder 3D ────────────────────────────────────────────────────────────────

class Encoder3D(nn.Module):
    """
    CNN 3D per volumi MRI (1, 96, 96, 96).
    Output: embedding (emb_dim,)
    """

    def __init__(self, emb_dim: int = 256):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(2),          # 48×48×48

            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2),          # 24×24×24

            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(2),          # 12×12×12

            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),  # 1×1×1
        )

        self.fc = nn.Linear(128, emb_dim)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ── Encoder Tabellare ─────────────────────────────────────────────────────────

class EncoderTab(nn.Module):
    """
    MLP per dati tabellari (8 feature).
    Output: embedding (emb_dim,)
    """

    def __init__(self, in_dim: int = 8, emb_dim: int = 256, dropout: float = 0.3):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


# ── Early Fusion Model ────────────────────────────────────────────────────────

class EarlyFusionModel(nn.Module):
    """
    Early Fusion multimodale completo (v1).

    Flusso:
        x_2d  → Encoder2D  → e_2d  (256,)
        x_3d  → Encoder3D  → e_3d  (256,)
        x_tab → EncoderTab → e_tab (256,)
              ↓
        concat([e_2d, e_3d, e_tab]) → (768,)
              ↓
        Classificatore MLP → logits (3,)
    """

    def __init__(self, emb_dim: int = 256, n_classes: int = 3,
                 dropout: float = 0.3):
        super().__init__()

        self.encoder_2d  = Encoder2D(emb_dim=emb_dim)
        self.encoder_3d  = Encoder3D(emb_dim=emb_dim)
        self.encoder_tab = EncoderTab(emb_dim=emb_dim, dropout=dropout)

        self.classifier = nn.Sequential(
            nn.Linear(emb_dim * 3, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, n_classes),
        )

    def forward(self, x_tab, x_2d, x_3d):
        e_2d  = self.encoder_2d(x_2d)
        e_3d  = self.encoder_3d(x_3d)
        e_tab = self.encoder_tab(x_tab)

        e_concat = torch.cat([e_2d, e_3d, e_tab], dim=1)
        return self.classifier(e_concat)


# ── Test rapido ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Test Early Fusion Model v1")
    print("=" * 60)

    batch = 4
    x_tab = torch.randn(batch, 8)
    x_2d  = torch.randn(batch, 1, 224, 224)
    x_3d  = torch.randn(batch, 1, 96, 96, 96)

    model  = EarlyFusionModel()
    logits = model(x_tab, x_2d, x_3d)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nInput  x_tab : {x_tab.shape}")
    print(f"Input  x_2d  : {x_2d.shape}")
    print(f"Input  x_3d  : {x_3d.shape}")
    print(f"Output logits: {logits.shape}")
    print(f"Parametri    : {n_params:,}")
    print("\nEarly Fusion Model v1 OK!")
    print("=" * 60)

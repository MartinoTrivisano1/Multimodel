"""
OASIS-2 Multimodal Dataset — PyTorch
======================================
Carica le tre modalità per ogni sessione:
    x_tab  → (8,)            feature tabellari
    x_2d   → (1, 224, 224)   slice MRI 2D
    x_3d   → (1, 96, 96, 96) volume MRI 3D
    y      → 0/1/2           CN / MCI / AD

Uso:
    from multimodal_dataset import get_dataloaders
    train_loader, val_loader, test_loader = get_dataloaders()
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# ── Configurazione ────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.normpath(os.path.join(BASE_DIR, ".."))

MANIFEST_CSV = os.path.join(PROJECT_DIR, "data_preprocessed", "mri_manifest.csv")
TABULAR_CSV  = os.path.join(PROJECT_DIR, "data_preprocessed", "oasis_tabular_aligned.csv")

FEATURE_COLS = ["Age", "EDUC", "MMSE", "eTIV", "nWBV", "ASF", "SES", "M/F"]
LABEL_NAMES  = {0: "CN", 1: "MCI", 2: "AD"}


# ── Dataset ───────────────────────────────────────────────────────────────────

class OASISMultimodalDataset(Dataset):
    """
    Dataset PyTorch multimodale OASIS-2.
    Carica tabellare + slice 2D + volume 3D per ogni sessione.
    """

    def __init__(self, split: str = "train"):
        # Carica manifest MRI
        manifest = pd.read_csv(MANIFEST_CSV)
        manifest = manifest[manifest["split"] == split].reset_index(drop=True)

        # Carica tabellare
        tabular = pd.read_csv(TABULAR_CSV)

        # Join su MRI ID
        df = manifest.merge(tabular[["MRI ID"] + FEATURE_COLS], on="MRI ID", how="left")

        self.mri_ids  = df["MRI ID"].tolist()
        self.labels   = torch.tensor(df["label"].values, dtype=torch.long)
        self.features = torch.tensor(df[FEATURE_COLS].values, dtype=torch.float32)
        self.paths_2d = df["path_2d"].tolist()
        self.paths_3d = df["path_3d"].tolist()
        self.split    = split

        print(f"  [{split:5s}] {len(self)} sessioni — "
              f"CN={sum(self.labels==0).item()} "
              f"MCI={sum(self.labels==1).item()} "
              f"AD={sum(self.labels==2).item()}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Tabellare
        x_tab = self.features[idx]               # (8,)

        # 2D slice → aggiungi canale → (1, 224, 224)
        x_2d = np.load(self.paths_2d[idx])       # (224, 224)
        x_2d = torch.tensor(x_2d, dtype=torch.float32).unsqueeze(0)

        # 3D volume → aggiungi canale → (1, 96, 96, 96)
        x_3d = np.load(self.paths_3d[idx])       # (96, 96, 96)
        x_3d = torch.tensor(x_3d, dtype=torch.float32).unsqueeze(0)

        y = self.labels[idx]

        return x_tab, x_2d, x_3d, y

    def get_class_weights(self):
        counts  = torch.bincount(self.labels, minlength=3).float()
        weights = 1.0 / (counts + 1e-8)
        return weights / weights.sum()


# ── DataLoaders ───────────────────────────────────────────────────────────────

def get_dataloaders(batch_size: int = 4, num_workers: int = 0):
    """
    Crea DataLoader per train, val e test.
    Batch size piccolo (4) perché i volumi 3D sono grandi.
    """
    print("Caricamento dataset multimodale OASIS-2...")
    train_ds = OASISMultimodalDataset(split="train")
    val_ds   = OASISMultimodalDataset(split="val")
    test_ds  = OASISMultimodalDataset(split="test")

    # Sampler bilanciato per train
    class_weights  = train_ds.get_class_weights()
    sample_weights = class_weights[train_ds.labels]
    sampler = torch.utils.data.WeightedRandomSampler(
        weights     = sample_weights,
        num_samples = len(train_ds),
        replacement = True,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        sampler=sampler, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size,
        shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size,
        shuffle=False, num_workers=num_workers
    )

    print(f"\nBatch size    : {batch_size}")
    print(f"Train batches : {len(train_loader)}")
    print(f"Val   batches : {len(val_loader)}")
    print(f"Test  batches : {len(test_loader)}")

    return train_loader, val_loader, test_loader


# ── Test rapido ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Test OASISMultimodalDataset")
    print("=" * 60)

    train_loader, val_loader, test_loader = get_dataloaders(batch_size=4)

    print("\nVerifica primo batch train:")
    x_tab, x_2d, x_3d, y = next(iter(train_loader))
    print(f"  x_tab shape : {x_tab.shape}   dtype={x_tab.dtype}")
    print(f"  x_2d  shape : {x_2d.shape}  dtype={x_2d.dtype}")
    print(f"  x_3d  shape : {x_3d.shape} dtype={x_3d.dtype}")
    print(f"  y     shape : {y.shape}   dtype={y.dtype}")
    print(f"  labels      : {[LABEL_NAMES[l.item()] for l in y]}")
    print("\nDataset pronto!")
    print("=" * 60)

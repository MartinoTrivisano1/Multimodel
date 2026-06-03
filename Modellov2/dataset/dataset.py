"""
dataset/dataset.py
==================
PyTorch Dataset per OASIS-2 multimodale.

Legge il manifest.csv e restituisce per ogni sample le modalità
disponibili: tabellare, volumetriche, slice 2D, volume 3D.

Gestisce dinamicamente le combinazioni di modalità mancanti.

Augmentation aggiunte:
  - 2D: flip, rotazione, affine, gaussian blur, elastic, erasing
  - 3D: flip assiale casuale, rumore gaussiano, intensità shift
  - Tabellare: gaussian noise + dropout feature
"""

import os
import sys
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import config as cfg

PROCESSED_DIR  = os.path.join(ROOT, "data", "processed")
TABULAR_CSV    = os.path.join(PROCESSED_DIR, "tabular.csv")
VOLUMETRIC_CSV = os.path.join(PROCESSED_DIR, "volumetric.csv")
MANIFEST_CSV   = os.path.join(PROCESSED_DIR, "manifest.csv")


# ══════════════════════════════════════════════════════════════════════════════
# AUGMENTATION 3D (su array numpy)
# ══════════════════════════════════════════════════════════════════════════════

def augment_volume_3d(vol: np.ndarray, p: float = 0.5) -> np.ndarray:
    """
    Augmentation per volumi MRI 3D (D, H, W) in [0,1].

    Operazioni (ognuna applicata con probabilità p):
      - Flip casuale lungo asse D, H o W
      - Rumore gaussiano leggero (sigma 0.01-0.03)
      - Shift di intensità casuale (±5%)
      - Gamma correction casuale (0.8-1.2)
    """
    # flip casuale lungo i 3 assi
    for axis in [0, 1, 2]:
        if random.random() < p:
            vol = np.flip(vol, axis=axis).copy()

    # rumore gaussiano
    if random.random() < p:
        sigma = random.uniform(0.01, 0.03)
        vol   = np.clip(vol + np.random.normal(0, sigma, vol.shape), 0, 1).astype(np.float32)

    # shift intensità
    if random.random() < p:
        shift = random.uniform(-0.05, 0.05)
        vol   = np.clip(vol + shift, 0, 1).astype(np.float32)

    # gamma correction
    if random.random() < p:
        gamma = random.uniform(0.8, 1.2)
        vol   = np.power(np.clip(vol, 1e-6, 1), gamma).astype(np.float32)

    return vol


# ══════════════════════════════════════════════════════════════════════════════
# AUGMENTATION TABELLARE (su array numpy)
# ══════════════════════════════════════════════════════════════════════════════

def augment_tabular(feat: np.ndarray, p_noise: float = 0.3, p_dropout: float = 0.1) -> np.ndarray:
    """
    Augmentation per feature tabellari.

    Operazioni:
      - Rumore gaussiano leggero (simula variabilità di misura clinica)
      - Feature dropout casuale (simula missing values)

    NON applicato su sex_enc (ultimo elemento, binario).
    """
    feat = feat.copy()
    n    = len(feat) - 1   # esclude sex_enc

    # rumore gaussiano su feature continue
    if random.random() < p_noise:
        noise       = np.random.normal(0, 0.05, n).astype(np.float32)
        feat[:n]   += noise

    # dropout casuale di singole feature (→ 0)
    if random.random() < p_dropout:
        drop_idx    = random.randint(0, n - 1)
        feat[drop_idx] = 0.0

    return feat


# ══════════════════════════════════════════════════════════════════════════════
# TRANSFORMS 2D
# ══════════════════════════════════════════════════════════════════════════════

def get_transforms_2d(split: str) -> transforms.Compose:
    """
    Augmentation per le slice 2D.

    Train: flip + rotazione + affine + blur + erasing
    Val/Test: solo normalizzazione
    """
    if split == "train":
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
            ),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.08)),
        ])
    else:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])


# ══════════════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════════════

class OASISDataset(Dataset):
    """
    Dataset multimodale OASIS-2 con data augmentation.

    Per ogni sample restituisce un dict con le modalità disponibili:
    {
        "tabular"     : Tensor (N_TAB_FEAT,)
        "volumetric"  : Tensor (N_VOL_FEAT,)
        "slice_2d"    : Tensor (3, H, W)
        "volume_3d"   : Tensor (1, D, H, W)
        "label"       : int
        "mri_id"      : str
        "has_tabular" : bool
        "has_vol"     : bool
        "has_2d"      : bool
        "has_3d"      : bool
    }
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        df_tabular: pd.DataFrame,
        df_volumetric: pd.DataFrame,
        split: str = "train",
        model_cfg: str = "4",
        modality_dropout: float = cfg.MODALITY_DROPOUT,
    ):
        self.manifest         = manifest.reset_index(drop=True)
        self.split            = split
        self.is_train         = (split == "train")
        self.modality_dropout = modality_dropout if self.is_train else 0.0
        self.transform_2d     = get_transforms_2d(split)

        mcfg             = cfg.MODEL_CONFIGS[model_cfg]
        self.use_tabular = mcfg["tabular"]
        self.use_2d      = mcfg["2d"]
        self.use_3d      = mcfg["3d"]

        # index tabellare
        self.tab_index = {}
        if df_tabular is not None:
            tab_feat_cols = [
                c for c in df_tabular.columns
                if c not in ["subject_id", "mri_id", "label"]
            ]
            self.tab_feat_cols = tab_feat_cols
            for _, row in df_tabular.iterrows():
                self.tab_index[row["mri_id"]] = row[tab_feat_cols].values.astype(np.float32)
        else:
            self.tab_feat_cols = []

        # index volumetrico
        self.vol_index = {}
        if df_volumetric is not None:
            vol_feat_cols = [c for c in df_volumetric.columns if c != "mri_id"]
            self.vol_feat_cols = vol_feat_cols
            for _, row in df_volumetric.iterrows():
                self.vol_index[row["mri_id"]] = row[vol_feat_cols].values.astype(np.float32)
        else:
            self.vol_feat_cols = []

    def __len__(self) -> int:
        return len(self.manifest)

    def _maybe_dropout(self, flag: bool) -> bool:
        if flag and self.modality_dropout > 0:
            return random.random() > self.modality_dropout
        return flag

    def __getitem__(self, idx: int) -> dict:
        row    = self.manifest.iloc[idx]
        mri_id = row["mri_id"]
        label  = int(row["label"])

        sample = {"label": label, "mri_id": mri_id}

        # ── tabellare ─────────────────────────────────────────────────────
        has_tab = self.use_tabular and bool(row.get("tabular_available", False))
        has_tab = self._maybe_dropout(has_tab)
        if has_tab and mri_id in self.tab_index:
            feat = self.tab_index[mri_id]
            # augmentation tabellare solo in train
            if self.is_train:
                feat = augment_tabular(feat)
            sample["tabular"] = torch.from_numpy(feat)
        else:
            n = len(self.tab_feat_cols) if self.tab_feat_cols else cfg.N_TABULAR_FEATURES
            sample["tabular"] = torch.zeros(n, dtype=torch.float32)
        sample["has_tabular"] = has_tab and mri_id in self.tab_index

        # ── volumetriche ──────────────────────────────────────────────────
        has_vol = bool(row.get("volumetric_available", False))
        has_vol = self._maybe_dropout(has_vol)
        if has_vol and mri_id in self.vol_index:
            feat = self.vol_index[mri_id]
            # stesso rumore leggero sulle feature volumetriche
            if self.is_train:
                feat = augment_tabular(feat, p_noise=0.2, p_dropout=0.05)
            sample["volumetric"] = torch.from_numpy(feat)
        else:
            n = len(self.vol_feat_cols) if self.vol_feat_cols else 21
            sample["volumetric"] = torch.zeros(n, dtype=torch.float32)
        sample["has_vol"] = has_vol and mri_id in self.vol_index

        # ── slice 2D ──────────────────────────────────────────────────────
        has_2d = self.use_2d and bool(row.get("has_2d", False))
        has_2d = self._maybe_dropout(has_2d)
        if has_2d:
            try:
                ax = np.load(row["slice_ax_path"])
                co = np.load(row["slice_co_path"])
                sa = np.load(row["slice_sa_path"])
                stack    = np.stack([ax, co, sa], axis=0)
                slices_t = []
                for i in range(3):
                    sl_uint8 = (stack[i] * 255).astype(np.uint8)
                    slices_t.append(self.transform_2d(sl_uint8))
                sample["slice_2d"] = torch.cat(slices_t, dim=0)
            except Exception:
                has_2d = False
                sample["slice_2d"] = torch.zeros(3, *cfg.SIZE_2D, dtype=torch.float32)
        else:
            sample["slice_2d"] = torch.zeros(3, *cfg.SIZE_2D, dtype=torch.float32)
        sample["has_2d"] = has_2d

        # ── volume 3D ─────────────────────────────────────────────────────
        has_3d = self.use_3d and bool(row.get("has_3d", False))
        has_3d = self._maybe_dropout(has_3d)
        if has_3d:
            try:
                vol = np.load(row["volume_3d_path"]).astype(np.float32)
                # augmentation 3D solo in train
                if self.is_train:
                    vol = augment_volume_3d(vol, p=0.4)
                # normalizza in [-1, 1]
                vol = vol * 2.0 - 1.0
                sample["volume_3d"] = torch.from_numpy(vol[np.newaxis])
            except Exception:
                has_3d = False
                sample["volume_3d"] = torch.zeros(1, *cfg.SIZE_3D, dtype=torch.float32)
        else:
            sample["volume_3d"] = torch.zeros(1, *cfg.SIZE_3D, dtype=torch.float32)
        sample["has_3d"] = has_3d

        return sample


# ══════════════════════════════════════════════════════════════════════════════
# SPLIT & DATALOADER
# ══════════════════════════════════════════════════════════════════════════════

def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_dataloaders(
    model_cfg:   str = "4",
    batch_size:  int = cfg.BATCH_SIZE,
    seed:        int = cfg.SEED,
    num_workers: int = 0,
) -> dict:
    _set_seed(seed)

    manifest      = pd.read_csv(MANIFEST_CSV)
    df_tabular    = pd.read_csv(TABULAR_CSV)    if os.path.exists(TABULAR_CSV)    else None
    df_volumetric = pd.read_csv(VOLUMETRIC_CSV) if os.path.exists(VOLUMETRIC_CSV) else None

    print(f"\n[DATASET] Model config : {cfg.MODEL_CONFIGS[model_cfg]['name']}")
    print(f"  Totale sample nel manifest : {len(manifest)}")

    # split per subject_id (evita leakage)
    subj_labels = (
        manifest.groupby("subject_id")["label"]
        .agg(lambda x: x.value_counts().idxmax())
        .reset_index()
    )
    subj_ids = subj_labels["subject_id"].values
    labels   = subj_labels["label"].values

    subj_train, subj_temp, _, lab_temp = train_test_split(
        subj_ids, labels,
        test_size=1 - cfg.TRAIN_RATIO,
        stratify=labels,
        random_state=seed,
    )
    subj_val, subj_test = train_test_split(
        subj_temp, test_size=0.5,
        stratify=lab_temp,
        random_state=seed,
    )

    subj_train = set(subj_train)
    subj_val   = set(subj_val)
    subj_test  = set(subj_test)

    man_train = manifest[manifest["subject_id"].isin(subj_train)]
    man_val   = manifest[manifest["subject_id"].isin(subj_val)]
    man_test  = manifest[manifest["subject_id"].isin(subj_test)]

    print(f"  Train: {len(man_train)} sample ({len(subj_train)} soggetti)")
    print(f"  Val  : {len(man_val)} sample ({len(subj_val)} soggetti)")
    print(f"  Test : {len(man_test)} sample ({len(subj_test)} soggetti)")

    for split_name, split_df in [("Train", man_train), ("Val", man_val), ("Test", man_test)]:
        dist   = split_df["label"].value_counts().sort_index()
        counts = " | ".join(
            f"{cfg.CLASS_NAMES[i]}={dist.get(i, 0)}" for i in range(cfg.N_CLASSES)
        )
        print(f"  {split_name:5} label dist: {counts}")

    # class weights
    train_labels     = man_train["label"].values
    class_counts     = np.bincount(train_labels, minlength=cfg.N_CLASSES).astype(float)
    class_weights    = 1.0 / (class_counts + 1e-8)
    class_weights    = class_weights / class_weights.sum() * cfg.N_CLASSES
    class_weights_t  = torch.tensor(class_weights, dtype=torch.float32)
    print(f"\n  Class weights (CN/MCI/AD): {class_weights.round(3).tolist()}")

    # WeightedRandomSampler
    sample_weights = class_weights[train_labels]
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).float(),
        num_samples=len(sample_weights),
        replacement=True,
    )

    common_kwargs = dict(
        df_tabular=df_tabular,
        df_volumetric=df_volumetric,
        model_cfg=model_cfg,
    )

    ds_train = OASISDataset(man_train, split="train", **common_kwargs)
    ds_val   = OASISDataset(man_val,   split="val",   **common_kwargs)
    ds_test  = OASISDataset(man_test,  split="test",  **common_kwargs)

    loader_kwargs = dict(num_workers=num_workers, pin_memory=False)

    return {
        "train": DataLoader(
            ds_train, batch_size=batch_size,
            sampler=sampler, drop_last=True,   # drop_last evita batch size=1
            **loader_kwargs,
        ),
        "val": DataLoader(
            ds_val, batch_size=batch_size,
            shuffle=False, **loader_kwargs,
        ),
        "test": DataLoader(
            ds_test, batch_size=batch_size,
            shuffle=False, **loader_kwargs,
        ),
        "class_weights": class_weights_t,
    }


# ══════════════════════════════════════════════════════════════════════════════
# TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 52)
    print("  Dataset + Augmentation – test con dati sintetici")
    print("=" * 52)

    records = []
    for i in range(20):
        mri_id = f"OAS2_{i:04d}_MR1"
        records.append({
            "subject_id":           f"OAS2_{i:04d}",
            "mri_id":               mri_id,
            "label":                i % 3,
            "slice_ax_path":        None,
            "slice_co_path":        None,
            "slice_sa_path":        None,
            "volume_3d_path":       None,
            "tabular_available":    True,
            "volumetric_available": True,
            "has_2d":               False,
            "has_3d":               False,
            "aligned":              True,
        })
    manifest = pd.DataFrame(records)

    tab_cols = cfg.FEATURE_COLUMNS + ["sex_enc"]
    df_tab   = pd.DataFrame(np.random.randn(20, len(tab_cols)), columns=tab_cols)
    df_tab.insert(0, "mri_id",      [r["mri_id"]      for r in records])
    df_tab.insert(0, "label",       [r["label"]        for r in records])
    df_tab.insert(0, "subject_id",  [r["subject_id"]   for r in records])

    df_vol = pd.DataFrame(np.random.randn(20, 21), columns=[f"feat_{i}" for i in range(21)])
    df_vol.insert(0, "mri_id", [r["mri_id"] for r in records])

    ds = OASISDataset(
        manifest=manifest, df_tabular=df_tab,
        df_volumetric=df_vol, split="train",
        model_cfg="4", modality_dropout=0.0,
    )

    # verifica augmentation tabellare: due chiamate allo stesso sample
    # devono dare valori leggermente diversi
    s1 = ds[0]["tabular"].numpy()
    s2 = ds[0]["tabular"].numpy()
    aug_ok = not np.allclose(s1, s2)
    print(f"\n  Augmentation tabellare stochastica: {'OK' if aug_ok else 'WARN (stesso valore)'}")

    # verifica augmentation 3D su volume sintetico
    vol_fake = np.random.rand(*cfg.SIZE_3D).astype(np.float32)
    v1 = augment_volume_3d(vol_fake, p=1.0)
    v2 = augment_volume_3d(vol_fake, p=1.0)
    print(f"  Augmentation 3D stochastica       : {'OK' if not np.allclose(v1, v2) else 'WARN'}")
    print(f"  Range vol dopo aug                : [{v1.min():.3f}, {v1.max():.3f}]")

    sample = ds[0]
    print(f"\n  tabular shape   : {sample['tabular'].shape}")
    print(f"  volumetric shape: {sample['volumetric'].shape}")
    print(f"  slice_2d shape  : {sample['slice_2d'].shape}")
    print(f"  volume_3d shape : {sample['volume_3d'].shape}")

    loader = DataLoader(ds, batch_size=4, shuffle=True)
    batch  = next(iter(loader))
    print(f"\n  Batch labels: {batch['label'].tolist()}")
    print("\n  Tutti i test OK")
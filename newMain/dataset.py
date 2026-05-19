import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch.nn.functional as F

# --- Configurazione Percorsi ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.normpath(BASE_DIR)
MANIFEST_CSV = os.path.join(PROJECT_DIR, "data_preprocessed", "mri_manifest.csv")
TABULAR_CSV = os.path.join(PROJECT_DIR, "data_preprocessed", "oasis_tabular_aligned.csv")
FEATURE_COLS = ["Age", "EDUC", "MMSE", "eTIV", "nWBV", "ASF", "SES", "M/F"]
LABEL_NAMES = ["CN", "MCI", "AD"]


def formatta_percorsi(array_percorsi, cartella_radice="data_preprocessed"):
    nuovi_percorsi = []
    for percorso in array_percorsi:
        parti = percorso.split(cartella_radice)
        if len(parti) > 1:
            rel_path = cartella_radice + parti[-1].replace('\\', '/')
            nuovi_percorsi.append(os.path.join(PROJECT_DIR, rel_path))
    return nuovi_percorsi


class OASISFullDataset(Dataset):
    def __init__(self, augment=False):
        manifest = pd.read_csv(MANIFEST_CSV)
        tabular = pd.read_csv(TABULAR_CSV)
        df = manifest.merge(tabular[["MRI ID"] + FEATURE_COLS], on="MRI ID", how="left")

        self.labels = torch.tensor(df["label"].values, dtype=torch.long)
        self.features = torch.tensor(df[FEATURE_COLS].values, dtype=torch.float32)
        self.paths_2d = formatta_percorsi(df["path_2d"].tolist())
        self.paths_3d = formatta_percorsi(df["path_3d"].tolist())
        self.augment = augment

        # Trasformazioni 2D (Rotazione, Traslazione, Noise)
        self.transform_2d = T.Compose([
            T.RandomRotation(10),
            T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        ])

    def __len__(self):
        return len(self.labels)

    def _add_noise(self, tensor, std=0.01):
        noise = torch.randn(tensor.size()) * std
        return tensor + noise

    def __getitem__(self, idx):
        x_tab = self.features[idx]
        x_2d = torch.tensor(np.load(self.paths_2d[idx]), dtype=torch.float32).unsqueeze(0) / 255.0
        x_3d = torch.tensor(np.load(self.paths_3d[idx]), dtype=torch.float32).unsqueeze(0)

        if self.augment:
            # Applica Augmentation 2D
            x_2d = self.transform_2d(x_2d)
            # Aggiungi Rumore Gaussiano
            x_2d = self._add_noise(x_2d)
            # (Opzionale) Rumore leggero anche sul 3D
            x_3d = self._add_noise(x_3d, std=0.005)

        return x_tab, x_2d, x_3d, self.labels[idx]
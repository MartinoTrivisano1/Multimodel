"""
OASIS-2 MRI Preprocessing — 2D e 3D
======================================
Formato input: .hdr/.img (NIfTI paired format)

Struttura attesa:
    data/immagini/OAS2_RAW_PART1/
    └── OAS2_0001_MR1/
        └── RAW/
            ├── mpr-1.nifti.hdr
            ├── mpr-1.nifti.img
            ├── mpr-2.nifti.hdr
            └── mpr-2.nifti.img

Output:
    data_preprocessed/
    ├── slices_2d/
    │   └── OAS2_0001_MR1.npy   # (224, 224) float32
    ├── volumes_3d/
    │   └── OAS2_0001_MR1.npy   # (96, 96, 96) float32
    └── mri_manifest.csv
"""

import os
import numpy as np
import nibabel as nib
import pandas as pd
from scipy.ndimage import zoom
from pathlib import Path

# ── Configurazione ────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.normpath(os.path.join(BASE_DIR, ".."))

MRI_DIR     = os.path.join(PROJECT_DIR, "data", "immagini", "OAS2_RAW_PART1")
TABULAR_CSV = os.path.join(PROJECT_DIR, "data_preprocessed", "oasis_tabular_aligned.csv")
OUTPUT_DIR  = os.path.join(PROJECT_DIR, "data_preprocessed")
OUT_2D      = os.path.join(OUTPUT_DIR, "slices_2d")
OUT_3D      = os.path.join(OUTPUT_DIR, "volumes_3d")

# Dimensioni target
SIZE_2D  = (224, 224)
SIZE_3D  = (96, 96, 96)
AXIS_2D  = 2   # slice assiale


# ── Funzioni ──────────────────────────────────────────────────────────────────

def find_hdr_file(session_dir: Path) -> Path | None:
    """
    Trova il file .hdr nella cartella RAW della sessione.
    Preferisce mpr-1 come acquisizione più stabile.
    """
    raw_dir = session_dir / "RAW"
    if not raw_dir.exists():
        return None

    hdr_files = sorted(raw_dir.glob("*.nifti.hdr"))
    if not hdr_files:
        return None

    # Preferisci mpr-1
    for f in hdr_files:
        if "mpr-1" in f.name:
            return f

    return hdr_files[0]


def load_and_normalize(hdr_path: Path) -> np.ndarray:
    """
    Carica volume NIfTI .hdr/.img e normalizza in [0,1].
    Se il volume è 4D prende solo il primo volume (index 0).
    """
    img = nib.load(str(hdr_path))
    vol = img.get_fdata(dtype=np.float32)

    # Gestione volume 4D → prendi primo volume
    if vol.ndim == 4:
        vol = vol[:, :, :, 0]

    # Assicurati che sia 3D
    if vol.ndim != 3:
        raise ValueError(f"Volume ha {vol.ndim} dimensioni, attese 3.")

    # Normalizzazione percentile robusta
    p1, p99 = np.percentile(vol, [1, 99])
    vol = np.clip(vol, p1, p99)
    vol = (vol - p1) / (p99 - p1 + 1e-8)

    return vol


def resize_3d(vol: np.ndarray, target: tuple) -> np.ndarray:
    """Resize volume 3D con interpolazione trilineare."""
    factors = [t / s for t, s in zip(target, vol.shape)]
    return zoom(vol, factors, order=1).astype(np.float32)


def extract_slice_2d(vol: np.ndarray, axis: int, target: tuple) -> np.ndarray:
    """Estrae slice centrale e ridimensiona."""
    idx = vol.shape[axis] // 2
    if axis == 0:   sl = vol[idx, :, :]
    elif axis == 1: sl = vol[:, idx, :]
    else:           sl = vol[:, :, idx]

    factors = [t / s for t, s in zip(target, sl.shape)]
    return zoom(sl, factors, order=1).astype(np.float32)


def preprocess_session(mri_id: str, mri_dir: Path) -> dict | None:
    """Preprocessa una singola sessione MRI."""
    session_dir = mri_dir / mri_id
    if not session_dir.exists():
        return None

    hdr_path = find_hdr_file(session_dir)
    if hdr_path is None:
        print(f"    [!] Nessun .hdr trovato in: {session_dir}/RAW/")
        return None

    try:
        vol = load_and_normalize(hdr_path)

        # 2D
        slice_2d = extract_slice_2d(vol, axis=AXIS_2D, target=SIZE_2D)
        path_2d  = os.path.join(OUT_2D, f"{mri_id}.npy")
        np.save(path_2d, slice_2d)

        # 3D
        vol_3d  = resize_3d(vol, target=SIZE_3D)
        path_3d = os.path.join(OUT_3D, f"{mri_id}.npy")
        np.save(path_3d, vol_3d)

        return {"path_2d": path_2d, "path_3d": path_3d}

    except Exception as e:
        print(f"    [!] Errore su {mri_id}: {e}")
        return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("OASIS-2 MRI Preprocessing — 2D e 3D")
    print("=" * 60)

    if not os.path.exists(MRI_DIR):
        print(f"\n[ERRORE] Cartella MRI non trovata: {MRI_DIR}")
        return
    if not os.path.exists(TABULAR_CSV):
        print(f"\n[ERRORE] CSV non trovato: {TABULAR_CSV}")
        print("Esegui prima check_alignment.py")
        return

    os.makedirs(OUT_2D, exist_ok=True)
    os.makedirs(OUT_3D, exist_ok=True)

    df = pd.read_csv(TABULAR_CSV)
    print(f"\nSessioni da processare : {len(df)}")
    print(f"Output 2D  → {OUT_2D}")
    print(f"Output 3D  → {OUT_3D}\n")

    mri_dir = Path(MRI_DIR)
    results = []
    ok, skip = 0, 0

    for i, row in df.iterrows():
        mri_id = row["MRI ID"]
        print(f"[{i+1:3d}/{len(df)}] {mri_id} "
              f"(label={row['label']}, split={row['split']})", end=" ")

        res = preprocess_session(mri_id, mri_dir)

        if res is not None:
            results.append({
                "Subject ID": row["Subject ID"],
                "MRI ID":     mri_id,
                "label":      row["label"],
                "split":      row["split"],
                "path_2d":    res["path_2d"],
                "path_3d":    res["path_3d"],
            })
            print("✅")
            ok += 1
        else:
            print("❌")
            skip += 1

    # Salva manifest
    manifest      = pd.DataFrame(results)
    manifest_path = os.path.join(OUTPUT_DIR, "mri_manifest.csv")
    manifest.to_csv(manifest_path, index=False)

    print("\n" + "=" * 60)
    print(f"Completato!")
    print(f"  ✅ Processati : {ok}")
    print(f"  ❌ Saltati    : {skip}")
    print(f"  Manifest     : {manifest_path}")
    print(f"\n  Distribuzione split:")
    if len(manifest) > 0:
        for split in ["train", "val", "test"]:
            sub = manifest[manifest["split"] == split]
            print(f"  {split:5s}: {len(sub):3d} sessioni | "
                  f"CN={sum(sub['label']==0)} "
                  f"MCI={sum(sub['label']==1)} "
                  f"AD={sum(sub['label']==2)}")
    else:
        print("  Nessuna sessione processata con successo.")
    print("=" * 60)


if __name__ == "__main__":
    main()

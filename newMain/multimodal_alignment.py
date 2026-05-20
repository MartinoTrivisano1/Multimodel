import os
import json
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from scipy.ndimage import zoom

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PREPROCESSED = os.path.join(BASE_DIR, "data_preprocessed")
TABULAR_CSV = os.path.join(DATA_PREPROCESSED,"oasis_tabular_aligned.csv")
MRI_ROOT = os.path.join(BASE_DIR,"data","immagini","OAS2_RAW_PART1")
OUT_2D = os.path.join(DATA_PREPROCESSED,"slices_2d")
OUT_3D = os.path.join(DATA_PREPROCESSED,"volumes_3d")
MANIFEST_OUT = os.path.join(DATA_PREPROCESSED,"multimodal_manifest.csv")

SIZE_2D = (224, 224)
SIZE_3D = (96, 96, 96)

ID_COL = "MRI ID"

def preprocessing_done():
    return (
        os.path.exists(MANIFEST_OUT)
        and os.path.isdir(OUT_2D)
        and os.path.isdir(OUT_3D)
        and len(os.listdir(OUT_2D)) > 0
        and len(os.listdir(OUT_3D)) > 0
    )

def find_hdr(session_dir):
    raw_dir = Path(session_dir) / "RAW"
    if not raw_dir.exists():
        return None
    hdr_files = sorted(
        raw_dir.glob("*.nifti.hdr")
    )
    if not hdr_files:
        hdr_files = sorted(
            raw_dir.glob("*.hdr")
        )
    if not hdr_files:
        return None
    for f in hdr_files:
        if "mpr-1" in f.name:
            return f
    return hdr_files[0]

def load_volume(hdr_path):
    img = nib.load(str(hdr_path))
    vol = np.asarray(
        img.dataobj,
        dtype=np.float32
    )
    if vol.ndim == 4:
        vol = vol[:, :, :, 0]
    if vol.ndim != 3:
        raise ValueError(
            f"Volume con {vol.ndim} dimensioni."
        )
    return vol

def normalize_volume(vol):
    p_low, p_high = np.percentile(
        vol,
        [1.0, 99.0]
    )
    vol = np.clip(
        vol,
        p_low,
        p_high
    )
    mask = vol > 1e-6
    brain = vol[mask]
    if brain.size == 0:
        return vol.astype(np.float32)
    mean = brain.mean()
    std = brain.std()
    out = np.zeros_like(
        vol,
        dtype=np.float32
    )
    out[mask] = (
        (vol[mask] - mean) /
        (std + 1e-8)
    )
    return out

def resize_3d(vol):
    factors = [
        target / source
        for target, source in zip(
            SIZE_3D,
            vol.shape
        )
    ]
    return zoom(
        vol,
        factors,
        order=1
    ).astype(np.float32)

def extract_2d(vol):
    idx = vol.shape[2] // 2
    sl = vol[:, :, idx]
    factors = [
        SIZE_2D[0] / sl.shape[0],
        SIZE_2D[1] / sl.shape[1]
    ]
    return zoom(
        sl,
        factors,
        order=1
    ).astype(np.float32)

def scan_mri_folders():
    root = Path(MRI_ROOT)
    if not root.exists():
        raise FileNotFoundError(
            f"Cartella immagini non trovata:\n{root}"
        )
    registry = {}
    for item in root.iterdir():
        if item.is_dir():
            registry[item.name] = str(item)
    return registry


def preprocess_one(mri_id, session_path):
    hdr_path = find_hdr(session_path)
    if hdr_path is None:
        return None
    vol = load_volume(hdr_path)
    original_shape = list(vol.shape)
    vol = normalize_volume(vol)
    slice_2d = extract_2d(vol)
    slice_2d = np.expand_dims(slice_2d, axis=0)
    vol_3d = resize_3d(vol)
    vol_3d = np.expand_dims(vol_3d, axis=0)
    path_2d = os.path.join(
        OUT_2D,
        f"{mri_id}.npy"
    )
    path_3d = os.path.join(
        OUT_3D,
        f"{mri_id}.npy"
    )
    np.save(
        path_2d,
        slice_2d.astype(np.float32)
    )
    np.save(
        path_3d,
        vol_3d.astype(np.float32)
    )
    return {
        "path_2d": path_2d,
        "path_3d": path_3d,
        "source_file": str(hdr_path),
        "original_shape": json.dumps(original_shape)
    }
def main():
    print("=" * 70)
    print("Multimodal Full Preprocessing Pipeline")
    print("=" * 70)
    if preprocessing_done():
        print("\nPreprocessing già esistente")
        print("Skip preprocessing")
        return
    if not os.path.exists(TABULAR_CSV):

        raise FileNotFoundError(
            f"CSV tabellare non trovato:\n{TABULAR_CSV}"
        )
    os.makedirs(
        DATA_PREPROCESSED,
        exist_ok=True
    )
    os.makedirs(
        OUT_2D,
        exist_ok=True
    )
    os.makedirs(
        OUT_3D,
        exist_ok=True
    )
    df = pd.read_csv(TABULAR_CSV)
    if ID_COL not in df.columns:
        raise ValueError(
            f"Colonna '{ID_COL}' non trovata.\n"
            f"Colonne disponibili:\n{list(df.columns)}"
        )
    registry = scan_mri_folders()
    print(f"\n[1] Righe tabellari : {len(df)}")
    print(f"[2] Cartelle MRI    : {len(registry)}")
    df["mri_path"] = df[ID_COL].map(registry)
    df["has_mri"] = df["mri_path"].notna()
    aligned_df = df[df["has_mri"]].copy()
    print(f"[3] MRI allineate   : {len(aligned_df)}")
    print(f"[4] MRI mancanti    : {len(df) - len(aligned_df)}")
    rows = []
    ok = 0
    skip = 0
    total = len(aligned_df)
    print("\n[5] Preprocessing MRI\n")
    for idx, (_, row) in enumerate(
        aligned_df.iterrows(),
        start=1
    ):
        progress = (idx / total) * 100
        print(
            f"\rProgress: "
            f"{idx}/{total} "
            f"({progress:.1f}%)",
            end=""
        )
        try:
            mri_id = row[ID_COL]
            result = preprocess_one(
                mri_id,
                row["mri_path"]
            )
            if result is None:
                skip += 1
                continue
            new_row = row.to_dict()
            new_row.update(result)
            rows.append(new_row)
            ok += 1
        except:
            skip += 1
    print()

    manifest = pd.DataFrame(rows)

    manifest.to_csv(
        MANIFEST_OUT,
        index=False
    )
    print("\n" + "=" * 70)
    print("Pipeline completata")
    print(f"✅ Processati : {ok}")
    print(f"❌ Saltati    : {skip}")
    print(f"\nManifest:")
    print(MANIFEST_OUT)
    print("=" * 70)
if __name__ == "__main__":
    main()
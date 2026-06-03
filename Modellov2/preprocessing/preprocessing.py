"""
preprocessing/preprocessing.py
================================
Preprocessing completo per OASIS-2 con:
  - Allineamento tabellare ↔ immagini
  - Ripetibilità (--force per rieseguire)
  - Estrazione feature volumetriche dai volumi grezzi (senza FreeSurfer)

Struttura input:
  data/tabellare/Tabellare.xlsx
  data/immagini/OAS2_RAW_PART1/OAS2_{ID}_MR{n}/RAW/mpr-1.nifti.hdr

Struttura output:
  data/processed/tabular.csv
  data/processed/volumetric.csv
  data/processed/slices_2d/{MRI_ID}_{ax|co|sa}.npy
  data/processed/volumes_3d/{MRI_ID}.npy
  data/processed/manifest.csv
  data/artifacts/tabular_scaler.pkl
  data/artifacts/volumetric_scaler.pkl

Uso:
  python preprocessing/preprocessing.py
  python preprocessing/preprocessing.py --force
"""

import os
import sys
import pickle
import warnings
import argparse
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import nibabel as nib
from PIL import Image
from tqdm import tqdm
from scipy.stats import entropy as scipy_entropy
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer

# ── config dalla root ─────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import config as cfg

# ── path ──────────────────────────────────────────────────────────────────────
TABULAR_FILE = os.path.join(ROOT, "data", "tabellare", "Tabellare.xlsx")
MRI_ROOT = os.path.join(ROOT, "data", "immagini", "OAS2_RAW_PART1")
PROCESSED_DIR = os.path.join(ROOT, "data", "processed")
SLICES_2D_DIR = os.path.join(PROCESSED_DIR, "slices_2d")
VOLUMES_3D_DIR = os.path.join(PROCESSED_DIR, "volumes_3d")
TABULAR_CSV = os.path.join(PROCESSED_DIR, "tabular.csv")
VOLUMETRIC_CSV = os.path.join(PROCESSED_DIR, "volumetric.csv")
MANIFEST_CSV = os.path.join(PROCESSED_DIR, "manifest.csv")
ARTIFACTS_DIR = os.path.join(ROOT, "data", "artifacts")
SCALER_TAB_PATH = os.path.join(ARTIFACTS_DIR, "tabular_scaler.pkl")
SCALER_VOL_PATH = os.path.join(ARTIFACTS_DIR, "volumetric_scaler.pkl")

for d in [PROCESSED_DIR, SLICES_2D_DIR, VOLUMES_3D_DIR, ARTIFACTS_DIR]:
    os.makedirs(d, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# UTILITÀ – ripetibilità
# ══════════════════════════════════════════════════════════════════════════════

def _already_processed_2d(mri_id: str) -> bool:
    for axis in ["ax", "co", "sa"]:
        path = os.path.join(SLICES_2D_DIR, f"{mri_id}_{axis}.npy")
        if not os.path.exists(path):
            return False
        try:
            if np.load(path).shape != cfg.SIZE_2D:
                return False
        except Exception:
            return False
    return True


def _already_processed_3d(mri_id: str) -> bool:
    path = os.path.join(VOLUMES_3D_DIR, f"{mri_id}.npy")
    if not os.path.exists(path):
        return False
    try:
        return np.load(path).shape == cfg.SIZE_3D
    except Exception:
        return False


# ══════════════════════════════════════════════════════════════════════════════
# UTILITÀ – allineamento
# ══════════════════════════════════════════════════════════════════════════════

def _find_mri_path(mri_id: str) -> str | None:
    candidate = os.path.join(MRI_ROOT, mri_id, "RAW", "mpr-1.nifti.hdr")
    if os.path.exists(candidate):
        return candidate
    raw_dir = os.path.join(MRI_ROOT, mri_id, "RAW")
    if os.path.isdir(raw_dir):
        for f in sorted(os.listdir(raw_dir)):
            if f.endswith(".hdr"):
                return os.path.join(raw_dir, f)
    return None


def check_alignment(df_tab: pd.DataFrame) -> dict:
    tab_ids  = set(df_tab["mri_id"].tolist())
    disk_ids = set()
    if os.path.isdir(MRI_ROOT):
        for name in os.listdir(MRI_ROOT):
            if os.path.isdir(os.path.join(MRI_ROOT, name)):
                disk_ids.add(name)

    ok = tab_ids & disk_ids
    missing = tab_ids - disk_ids
    extra = disk_ids - tab_ids

    shape_errors = []
    for mri_id in ok:
        for axis in ["ax", "co", "sa"]:
            p = os.path.join(SLICES_2D_DIR, f"{mri_id}_{axis}.npy")
            if os.path.exists(p):
                try:
                    arr = np.load(p)
                    if arr.shape != cfg.SIZE_2D:
                        shape_errors.append(f"{mri_id}_{axis}: {arr.shape} ≠ {cfg.SIZE_2D}")
                except Exception as e:
                    shape_errors.append(f"{mri_id}_{axis}: errore ({e})")
        p3d = os.path.join(VOLUMES_3D_DIR, f"{mri_id}.npy")
        if os.path.exists(p3d):
            try:
                arr = np.load(p3d)
                if arr.shape != cfg.SIZE_3D:
                    shape_errors.append(f"{mri_id}_3d: {arr.shape} ≠ {cfg.SIZE_3D}")
            except Exception as e:
                shape_errors.append(f"{mri_id}_3d: errore ({e})")

    report = {
        "mri_ids_ok":      sorted(ok),
        "mri_ids_missing": sorted(missing),
        "mri_ids_extra":   sorted(extra),
        "shape_errors":    shape_errors,
        "n_ok":            len(ok),
        "n_missing":       len(missing),
        "n_extra":         len(extra),
        "n_shape_errors":  len(shape_errors),
    }

    print("\n[ALLINEAMENTO tabellare ↔ immagini]")
    print(f"  Nel tabellare       : {len(tab_ids)}")
    print(f"  Su disco            : {len(disk_ids)}")
    print(f"  Allineati (ok)      : {len(ok)}")
    print(f"  Mancanti su disco   : {len(missing)}")
    if missing:
        print(f"    → {sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}")
    print(f"  Extra su disco      : {len(extra)}")
    print(f"  Errori di shape     : {len(shape_errors)}")
    if shape_errors:
        for e in shape_errors[:5]:
            print(f"    [ERR] {e}")
    return report


# ══════════════════════════════════════════════════════════════════════════════
# 1. PREPROCESSING TABELLARE
# ══════════════════════════════════════════════════════════════════════════════

def preprocess_tabular(force: bool = False, save: bool = True) -> pd.DataFrame:
    """
    Carica, pulisce, imputa e normalizza il tabellare con RobustScaler.
    Se tabular.csv esiste e force=False, lo carica direttamente.
    """
    if not force and os.path.exists(TABULAR_CSV):
        print(f"\n[TABULAR] Già processato → carico {TABULAR_CSV}")
        return pd.read_csv(TABULAR_CSV)

    print("\n[TABULAR] Caricamento Tabellare.xlsx...")
    df = pd.read_excel(TABULAR_FILE, engine="openpyxl")
    df = df.rename(columns={
        "Subject ID": "subject_id",
        "MRI ID":     "mri_id",
        "Group":      "group",
        "M/F":        "sex",
    })
    print(f"  Righe: {len(df)}  |  Soggetti: {df['subject_id'].nunique()}")

    # label da CDR
    df["label"] = df["CDR"].map(cfg.CDR_TO_CLASS)
    n_invalid = df["label"].isna().sum()
    if n_invalid > 0:
        print(f"  [WARN] {n_invalid} righe con CDR non mappabile → rimosse")
        df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    print("\n  Distribuzione label:")
    for cls, name in enumerate(cfg.CLASS_NAMES):
        print(f"    {name}: {(df['label'] == cls).sum()} samples")

    # codifica sesso
    df["sex_enc"] = df["sex"].map({"M": 1, "F": 0}).astype(float)

    # imputazione: mediana per soggetto poi globale
    for col in ["SES", "MMSE"]:
        df[col] = df.groupby("subject_id")[col].transform(
            lambda x: x.fillna(x.median())
        )
    all_feat = cfg.FEATURE_COLUMNS + ["sex_enc"]
    existing = [c for c in all_feat if c in df.columns]
    imputer = SimpleImputer(strategy="median")
    df[existing] = imputer.fit_transform(df[existing])
    print(f"\n  Mancanti dopo imputazione: {df[existing].isna().sum().sum()}")

    # RobustScaler (mediana + IQR, robusto agli outlier clinici)
    num_cols = [c for c in cfg.FEATURE_COLUMNS if c in df.columns]
    scaler = RobustScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    with open(SCALER_TAB_PATH, "wb") as f:
        pickle.dump(scaler, f)
    print(f"  Scaler (RobustScaler) → {SCALER_TAB_PATH}")

    out_cols = ["subject_id", "mri_id", "label"] + num_cols + ["sex_enc"]
    df_out = df[out_cols].copy()

    if save:
        df_out.to_csv(TABULAR_CSV, index=False)
        print(f"  tabular.csv → {TABULAR_CSV}  ({len(df_out)} righe)")

    return df_out


# ══════════════════════════════════════════════════════════════════════════════
# 2. FEATURE VOLUMETRICHE (senza FreeSurfer)
# ══════════════════════════════════════════════════════════════════════════════

def extract_volumetric_features(vol: np.ndarray) -> dict:
    """
    Estrae feature volumetriche e di texture da un volume MRI grezzo.
    Il volume deve essere già normalizzato in [0,1] con squeeze applicato.

    Feature estratte
    ----------------
    Volumi approssimati (threshold sull'intensità):
      - brain_vol        : volume totale cervello (voxel > 0.15)
      - wm_vol           : materia bianca (intensità alta: > 0.65)
      - gm_vol           : materia grigia (intensità media: 0.35–0.65)
      - csf_vol          : liquor/ventricoli (intensità bassa: 0.05–0.20)
      - wm_gm_ratio      : rapporto WM/GM (proxy di atrofia)

    Asimmetria emisferica:
      - asymmetry_index  : (L - R) / (L + R) sull'intensità media

    Statistiche per zona anatomica (superiore / media / inferiore):
      - {zona}_mean, {zona}_std, {zona}_p25, {zona}_p75

    Texture globale:
      - global_entropy   : entropia dell'istogramma (proxy eterogeneità)
      - global_mean      : intensità media del cervello
      - global_std       : deviazione standard intensità

    Ritorna
    -------
    dict con ~25 feature float
    """
    feats = {}
    X, Y, Z = vol.shape

    # ── maschere per tipo di tessuto ─────────────────────────────────────────
    brain_mask = vol > 0.15
    wm_mask = vol > 0.65
    gm_mask = (vol >= 0.35) & (vol <= 0.65)
    csf_mask = (vol >= 0.05) & (vol < 0.20)

    brain_vox = brain_mask.sum()
    feats["brain_vol"]   = float(brain_vox)
    feats["wm_vol"]      = float(wm_mask.sum())
    feats["gm_vol"]      = float(gm_mask.sum())
    feats["csf_vol"]     = float(csf_mask.sum())
    feats["wm_gm_ratio"] = (
        feats["wm_vol"] / feats["gm_vol"]
        if feats["gm_vol"] > 0 else 0.0
    )

    # ── asimmetria emisferica (asse X = sinistra/destra) ─────────────────────
    mid = X // 2
    left_mean  = float(vol[:mid][brain_mask[:mid]].mean()) if brain_mask[:mid].any() else 0.0
    right_mean = float(vol[mid:][brain_mask[mid:]].mean()) if brain_mask[mid:].any() else 0.0
    denom = left_mean + right_mean
    feats["asymmetry_index"] = (left_mean - right_mean) / denom if denom > 0 else 0.0

    # ── statistiche per zona anatomica (terzi del volume lungo Z) ────────────
    zone_bounds = {
        "inferior": (0,        Z // 3),
        "middle":   (Z // 3,   2 * Z // 3),
        "superior": (2 * Z // 3, Z),
    }
    for zone, (z0, z1) in zone_bounds.items():
        zone_vol  = vol[:, :, z0:z1]
        zone_mask = brain_mask[:, :, z0:z1]
        voxels    = zone_vol[zone_mask]
        if len(voxels) > 0:
            feats[f"{zone}_mean"] = float(voxels.mean())
            feats[f"{zone}_std"]  = float(voxels.std())
            feats[f"{zone}_p25"]  = float(np.percentile(voxels, 25))
            feats[f"{zone}_p75"]  = float(np.percentile(voxels, 75))
        else:
            feats[f"{zone}_mean"] = 0.0
            feats[f"{zone}_std"]  = 0.0
            feats[f"{zone}_p25"]  = 0.0
            feats[f"{zone}_p75"]  = 0.0

    # ── texture globale ───────────────────────────────────────────────────────
    brain_voxels = vol[brain_mask]
    if len(brain_voxels) > 0:
        feats["global_mean"] = float(brain_voxels.mean())
        feats["global_std"]  = float(brain_voxels.std())
        # entropia dell'istogramma normalizzato (32 bin)
        hist, _ = np.histogram(brain_voxels, bins=32, range=(0, 1))
        hist_norm = hist / (hist.sum() + 1e-8)
        feats["global_entropy"] = float(scipy_entropy(hist_norm + 1e-8))
    else:
        feats["global_mean"]    = 0.0
        feats["global_std"]     = 0.0
        feats["global_entropy"] = 0.0

    return feats


def preprocess_volumetric(
    mri_ids: list[str],
    alignment_report: dict,
    force: bool = False,
    save: bool = True,
) -> pd.DataFrame:
    """
    Estrae feature volumetriche per ogni MRI disponibile su disco.
    Se volumetric.csv esiste e force=False, lo carica direttamente.

    Ritorna DataFrame con colonne [mri_id] + feature volumetriche normalizzate.
    """
    if not force and os.path.exists(VOLUMETRIC_CSV):
        print(f"\n[VOLUMETRIC] Già processato → carico {VOLUMETRIC_CSV}")
        return pd.read_csv(VOLUMETRIC_CSV)

    ok_ids = set(alignment_report["mri_ids_ok"])
    records = []
    errors = 0

    print(f"\n[VOLUMETRIC] Estrazione feature da {len(ok_ids)} volumi...")

    for mri_id in tqdm(mri_ids, desc="  Feature"):
        if mri_id not in ok_ids:
            # sample senza immagine → NaN per tutte le feature
            records.append({"mri_id": mri_id, "_missing": True})
            continue

        hdr_path = _find_mri_path(mri_id)
        if hdr_path is None:
            records.append({"mri_id": mri_id, "_missing": True})
            errors += 1
            continue

        try:
            vol  = _load_volume(hdr_path)
            feat = extract_volumetric_features(vol)
            feat["mri_id"]   = mri_id
            feat["_missing"] = False
            records.append(feat)
        except Exception as e:
            print(f"\n  [ERR] {mri_id}: {e}")
            records.append({"mri_id": mri_id, "_missing": True})
            errors += 1

    df = pd.DataFrame(records)

    # feature columns (escluse mri_id e _missing)
    feat_cols = [c for c in df.columns if c not in ["mri_id", "_missing"]]

    # imputa NaN (sample mancanti) con mediana globale
    imputer = SimpleImputer(strategy="median")
    df[feat_cols] = imputer.fit_transform(df[feat_cols])

    # normalizza con RobustScaler
    scaler = RobustScaler()
    df[feat_cols] = scaler.fit_transform(df[feat_cols])

    with open(SCALER_VOL_PATH, "wb") as f:
        pickle.dump(scaler, f)
    print(f"\n  Feature estratte    : {len(feat_cols)}")
    print(f"  Errori              : {errors}")
    print(f"  Scaler → {SCALER_VOL_PATH}")

    df_out = df.drop(columns=["_missing"])

    if save:
        df_out.to_csv(VOLUMETRIC_CSV, index=False)
        print(f"  volumetric.csv → {VOLUMETRIC_CSV}")

    return df_out


# ══════════════════════════════════════════════════════════════════════════════
# 3. PREPROCESSING MRI (2D slice + 3D volume)
# ══════════════════════════════════════════════════════════════════════════════

def _load_volume(hdr_path: str) -> np.ndarray:
    """
    Carica volume Analyze 7.5.
    Squeeze per rimuovere dim spurie (es. OASIS: X,Y,Z,1).
    Normalizza in [0,1] con clip percentilico robusto.
    """
    vol = nib.load(hdr_path).get_fdata(dtype=np.float32)
    vol = np.squeeze(vol)
    assert vol.ndim == 3, f"Shape inattesa dopo squeeze: {vol.shape}"
    p1, p99 = np.percentile(vol, 1), np.percentile(vol, 99)
    if p99 > p1:
        vol = np.clip(vol, p1, p99)
        vol = (vol - p1) / (p99 - p1)
    else:
        vol = np.zeros_like(vol)
    return vol


def _resize_volume_3d(vol: np.ndarray, target: tuple) -> np.ndarray:
    D, H, W = target
    vol_t   = vol.transpose(2, 0, 1)
    resized = []
    for sl in vol_t:
        img = Image.fromarray((sl * 255).astype(np.uint8))
        img = img.resize((W, H), Image.BILINEAR)
        resized.append(np.array(img, dtype=np.float32) / 255.0)
    vol_xy = np.stack(resized, axis=0)
    z_idx  = np.linspace(0, vol_xy.shape[0] - 1, D).astype(int)
    return vol_xy[z_idx]


def _extract_slices_2d(vol: np.ndarray, pct: float = cfg.SLICE_PCT_2D) -> dict:
    X, Y, Z = vol.shape
    H, W = cfg.SIZE_2D

    def _resize(sl):
        img = Image.fromarray((sl * 255).astype(np.uint8))
        img = img.resize((W, H), Image.BILINEAR)
        return np.array(img, dtype=np.float32) / 255.0

    return {
        "ax": _resize(vol[:, :, int(Z * pct)]),
        "co": _resize(vol[:, int(Y * pct), :]),
        "sa": _resize(vol[int(X * pct), :, :]),
    }


def preprocess_mri(
    mri_ids: list[str],
    alignment_report: dict,
    force: bool = False,
    save_2d: bool = True,
    save_3d: bool = True,
) -> dict:
    """
    Processa i volumi MRI: estrae slice 2D e volume 3D.
    Salta sample già processati correttamente (se force=False).
    """
    results = {}
    ok_ids = set(alignment_report["mri_ids_ok"])
    skipped = 0
    processed = 0
    errors = 0

    print(f"\n[MRI] {len(mri_ids)} sample nel tabellare")
    print(f"      {len(ok_ids)} allineati con disco → da processare")

    for mri_id in tqdm(mri_ids, desc="  Volumi"):
        if mri_id not in ok_ids:
            results[mri_id] = {
                "2d_ax": None, "2d_co": None, "2d_sa": None, "3d": None
            }
            continue

        done_2d = _already_processed_2d(mri_id)
        done_3d = _already_processed_3d(mri_id)

        if not force and done_2d and done_3d:
            skipped += 1
            results[mri_id] = {
                "2d_ax": os.path.join(SLICES_2D_DIR, f"{mri_id}_ax.npy"),
                "2d_co": os.path.join(SLICES_2D_DIR, f"{mri_id}_co.npy"),
                "2d_sa": os.path.join(SLICES_2D_DIR, f"{mri_id}_sa.npy"),
                "3d":    os.path.join(VOLUMES_3D_DIR, f"{mri_id}.npy"),
            }
            continue

        hdr_path = _find_mri_path(mri_id)
        if hdr_path is None:
            errors += 1
            results[mri_id] = {
                "2d_ax": None, "2d_co": None, "2d_sa": None, "3d": None
            }
            continue

        try:
            vol = _load_volume(hdr_path)
        except Exception as e:
            print(f"\n  [ERR] {mri_id}: {e}")
            errors += 1
            results[mri_id] = {
                "2d_ax": None, "2d_co": None, "2d_sa": None, "3d": None
            }
            continue

        entry = {"2d_ax": None, "2d_co": None, "2d_sa": None, "3d": None}

        if save_2d and (force or not done_2d):
            slices = _extract_slices_2d(vol)
            for axis, arr in slices.items():
                path = os.path.join(SLICES_2D_DIR, f"{mri_id}_{axis}.npy")
                np.save(path, arr)
                assert np.load(path).shape == cfg.SIZE_2D
                entry[f"2d_{axis}"] = path
        elif done_2d:
            for axis in ["ax", "co", "sa"]:
                entry[f"2d_{axis}"] = os.path.join(
                    SLICES_2D_DIR, f"{mri_id}_{axis}.npy"
                )

        if save_3d and (force or not done_3d):
            vol3d = _resize_volume_3d(vol, cfg.SIZE_3D)
            path  = os.path.join(VOLUMES_3D_DIR, f"{mri_id}.npy")
            np.save(path, vol3d)
            assert np.load(path).shape == cfg.SIZE_3D
            entry["3d"] = path
        elif done_3d:
            entry["3d"] = os.path.join(VOLUMES_3D_DIR, f"{mri_id}.npy")

        results[mri_id] = entry
        processed += 1

    print(f"\n  Processati  : {processed}")
    print(f"  Saltati (ok): {skipped}")
    print(f"  Errori      : {errors}")
    print(f"  Mancanti    : {len(mri_ids) - len(ok_ids)}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# 4. MANIFEST
# ══════════════════════════════════════════════════════════════════════════════

def build_manifest(
    df_tab: pd.DataFrame,
    df_vol: pd.DataFrame,
    mri_results: dict,
    alignment_report: dict,
    force: bool = False,
    save: bool = True,
) -> pd.DataFrame:
    """
    Costruisce manifest.csv con path di ogni modalità e feature volumetriche.

    Colonne output
    --------------
    subject_id, mri_id, label,
    slice_ax_path, slice_co_path, slice_sa_path,
    volume_3d_path,
    tabular_available, volumetric_available,
    has_2d, has_3d, aligned
    """
    if not force and os.path.exists(MANIFEST_CSV):
        print(f"\n[MANIFEST] Già esistente → carico {MANIFEST_CSV}")
        return pd.read_csv(MANIFEST_CSV)

    ok_ids = set(alignment_report["mri_ids_ok"])
    vol_ids = set(df_vol["mri_id"].tolist()) if df_vol is not None else set()
    records = []

    for _, row in df_tab.iterrows():
        mri_id = row["mri_id"]
        res = mri_results.get(mri_id, {})

        has_2d = all(res.get(f"2d_{ax}") is not None for ax in ["ax", "co", "sa"])
        has_3d = res.get("3d") is not None

        records.append({
            "subject_id":           row["subject_id"],
            "mri_id":               mri_id,
            "label":                row["label"],
            "slice_ax_path":        res.get("2d_ax"),
            "slice_co_path":        res.get("2d_co"),
            "slice_sa_path":        res.get("2d_sa"),
            "volume_3d_path":       res.get("3d"),
            "tabular_available":    True,
            "volumetric_available": mri_id in vol_ids and mri_id in ok_ids,
            "has_2d":               has_2d,
            "has_3d":               has_3d,
            "aligned":              mri_id in ok_ids,
        })

    manifest = pd.DataFrame(records)

    n_misaligned = (~manifest["aligned"]).sum()
    if n_misaligned > 0:
        print(f"\n  [WARN] {n_misaligned} sample senza immagine su disco")

    print(f"\n[MANIFEST]")
    print(f"  Totale sample         : {len(manifest)}")
    print(f"  Allineati             : {manifest['aligned'].sum()}")
    print(f"  Con 2D                : {manifest['has_2d'].sum()}")
    print(f"  Con 3D                : {manifest['has_3d'].sum()}")
    print(f"  Con tabellare         : {manifest['tabular_available'].sum()}")
    print(f"  Con volumetriche      : {manifest['volumetric_available'].sum()}")
    full = (
        manifest["has_2d"] &
        manifest["has_3d"] &
        manifest["tabular_available"] &
        manifest["aligned"]
    )
    print(f"  Full (tutte e 3)      : {full.sum()}")

    if save:
        manifest.to_csv(MANIFEST_CSV, index=False)
        print(f"  Salvato → {MANIFEST_CSV}")

    return manifest


# ══════════════════════════════════════════════════════════════════════════════
# 5. ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run_all(force: bool = False):
    print("=" * 56)
    print("  OASIS-2 · Preprocessing Pipeline")
    print(f"  Modalità: {'FORCE' if force else 'INCREMENTALE'}")
    print("=" * 56)

    # 1 – tabellare
    df_tab = preprocess_tabular(force=force, save=True)

    # 2 – allineamento
    alignment_report = check_alignment(df_tab)
    mri_ids = df_tab["mri_id"].tolist()

    # 3 – feature volumetriche (dai volumi grezzi, senza FreeSurfer)
    df_vol = preprocess_volumetric(
        mri_ids, alignment_report, force=force, save=True
    )

    # 4 – slice 2D + volume 3D
    mri_results = preprocess_mri(
        mri_ids, alignment_report,
        force=force, save_2d=True, save_3d=True,
    )

    # 5 – manifest
    manifest = build_manifest(
        df_tab, df_vol, mri_results, alignment_report,
        force=force, save=True,
    )

    print("\n[DONE] ✓ Preprocessing completato.")
    print(f"  tabular.csv     → {TABULAR_CSV}")
    print(f"  volumetric.csv  → {VOLUMETRIC_CSV}")
    print(f"  manifest.csv    → {MANIFEST_CSV}")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force", action="store_true",
        help="Riesegue tutto anche se i file esistono già",
    )
    args = parser.parse_args()
    run_all(force=args.force)
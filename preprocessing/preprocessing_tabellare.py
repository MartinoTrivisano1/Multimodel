"""
Check Allineamento MRI ↔ Tabellare
=====================================
Verifica quali MRI ID del CSV tabellare hanno
la corrispondente cartella MRI in data/immagini/
e produce un CSV tabellare filtrato solo con i soggetti
che hanno entrambe le modalità disponibili.

Struttura attesa:
    Multimodel/
    ├── data/
    │   └── immagini/
    │       └── OAS2_RAW_PART1/
    │           ├── OAS2_0001_MR1/
    │           ├── OAS2_0001_MR2/
    │           └── ...
    ├── data_preprocessed/
    │   └── oasis_tabular_preprocessed.csv
    └── preprocessing/
        └── check_alignment.py   ← questo script

Output:
    data_preprocessed/oasis_tabular_aligned.csv
    (solo sessioni con MRI disponibile)
"""

import os
import pandas as pd
from pathlib import Path

# ── Configurazione ────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.normpath(os.path.join(BASE_DIR, ".."))

MRI_DIR     = os.path.join(PROJECT_DIR, "data", "immagini", "OAS2_RAW_PART1")
TABULAR_CSV = os.path.join(PROJECT_DIR, "data_preprocessed", "oasis_tabular_preprocessed.csv")
OUTPUT_CSV  = os.path.join(PROJECT_DIR, "data_preprocessed", "oasis_tabular_aligned.csv")


def main():
    print("=" * 60)
    print("Check Allineamento MRI ↔ Tabellare")
    print("=" * 60)

    # ── Verifica path ─────────────────────────────────────────────────────────
    if not os.path.exists(MRI_DIR):
        print(f"\n[ERRORE] Cartella MRI non trovata: {MRI_DIR}")
        return

    if not os.path.exists(TABULAR_CSV):
        print(f"\n[ERRORE] CSV non trovato: {TABULAR_CSV}")
        print("Esegui prima preprocessing_tabellare.py")
        return

    # ── Leggi cartelle MRI disponibili ───────────────────────────────────────
    mri_folders = set(
        f.name for f in Path(MRI_DIR).iterdir() if f.is_dir()
    )
    print(f"\n[1] Cartelle MRI trovate in OAS2_RAW_PART1: {len(mri_folders)}")

    # ── Leggi CSV tabellare ───────────────────────────────────────────────────
    df = pd.read_csv(TABULAR_CSV)
    print(f"[2] Sessioni nel CSV tabellare      : {len(df)}")

    # ── Check allineamento ────────────────────────────────────────────────────
    df["mri_available"] = df["MRI ID"].isin(mri_folders)

    n_ok      = df["mri_available"].sum()
    n_missing = (~df["mri_available"]).sum()

    print(f"\n[3] Risultato allineamento:")
    print(f"    ✅ Sessioni con MRI disponibile : {n_ok}")
    print(f"    ❌ Sessioni senza MRI           : {n_missing}")

    # ── Mostra sessioni mancanti ──────────────────────────────────────────────
    if n_missing > 0:
        missing = df[~df["mri_available"]][["Subject ID", "MRI ID", "label", "split"]]
        print(f"\n[4] Sessioni senza MRI corrispondente:")
        print(missing.to_string(index=False))

        # Verifica se sono tutti soggetti > 0099 (PART2 non ancora scaricata)
        missing_nums = missing["Subject ID"].str.extract(r"OAS2_(\d+)")[0].astype(int)
        part2 = missing_nums[missing_nums > 99]
        part1 = missing_nums[missing_nums <= 99]

        if len(part2) > 0:
            print(f"\n    → {len(part2)} sessioni appartengono a PART2 (OAS2_01xx)")
            print(f"      Scarica OAS2_RAW_PART2.tar.gz per averle tutte")
        if len(part1) > 0:
            print(f"\n    → {len(part1)} sessioni di PART1 mancanti:")
            print(f"      {missing[missing_nums <= 99]['MRI ID'].tolist()}")

    # ── Crea CSV allineato ────────────────────────────────────────────────────
    df_aligned = df[df["mri_available"]].drop(columns=["mri_available"])
    df_aligned.to_csv(OUTPUT_CSV, index=False)

    print(f"\n[5] CSV allineato salvato: {OUTPUT_CSV}")
    print(f"    Shape: {df_aligned.shape}")
    print(f"\n    Distribuzione split:")
    for split in ["train", "val", "test"]:
        sub = df_aligned[df_aligned["split"] == split]
        print(f"    {split:5s}: {len(sub):3d} sessioni | "
              f"CN={sum(sub['label']==0)} "
              f"MCI={sum(sub['label']==1)} "
              f"AD={sum(sub['label']==2)}")

    print("\n" + "=" * 60)
    print("Check completato!")
    print("=" * 60)


if __name__ == "__main__":
    main()
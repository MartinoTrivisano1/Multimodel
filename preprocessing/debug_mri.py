"""
Debug struttura cartelle MRI
"""
import os
from pathlib import Path

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.normpath(os.path.join(BASE_DIR, ".."))
MRI_DIR     = os.path.join(PROJECT_DIR, "data", "immagini", "OAS2_RAW_PART1")

print(f"PROJECT_DIR : {PROJECT_DIR}")
print(f"MRI_DIR     : {MRI_DIR}")
print(f"Esiste      : {os.path.exists(MRI_DIR)}")
print()

if os.path.exists(MRI_DIR):
    # Mostra prime 3 cartelle
    folders = sorted(Path(MRI_DIR).iterdir())
    print(f"Cartelle trovate: {len(folders)}")
    print()

    for folder in folders[:3]:
        print(f"Cartella: {folder.name}")

        # Mostra contenuto
        for item in sorted(folder.iterdir()):
            print(f"  {item.name}/")
            if item.is_dir():
                for sub in sorted(item.iterdir()):
                    print(f"    {sub.name}")
        print()

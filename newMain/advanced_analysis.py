import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve

# Configurazioni
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

MANIFEST_CSV = os.path.join(BASE_DIR, "data_preprocessed", "mri_manifest.csv")
TABULAR_CSV = os.path.join(BASE_DIR, "data_preprocessed", "oasis_tabular_aligned.csv")


def analyze_clinical_thresholds():
    """Analizza le soglie cliniche (MMSE e nWBV) calcolando cut-off ottimali."""
    print("--- 1. Analisi Soglie Cliniche ---")
    if not os.path.exists(TABULAR_CSV):
        print("Errore: Dati tabellari non trovati.")
        return

    df = pd.read_csv(TABULAR_CSV)

    # Calcolo soglie
    df_clean = df.dropna(subset=['nWBV', 'MMSE', 'CDR']).copy()
    y_true = (df_clean['CDR'] > 0).astype(int)

    fpr_v, tpr_v, thresholds_v = roc_curve(y_true, -df_clean['nWBV'])
    ottimo_nwbv = -thresholds_v[np.argmax(tpr_v - fpr_v)]

    fpr_m, tpr_m, thresholds_m = roc_curve(y_true, -df_clean['MMSE'])
    ottimo_mmse = -thresholds_m[np.argmax(tpr_m - fpr_m)]

    with open(os.path.join(OUT_DIR, 'soglie_cliniche_report.txt'), 'w', encoding='utf-8') as f:
        f.write(f"SOGLIE CLINICHE CALCOLATE:\n- nWBV: {ottimo_nwbv:.3f}\n- MMSE: {ottimo_mmse:.1f}\n")
    print(f"Salvato: {os.path.join(OUT_DIR, 'soglie_cliniche_report.txt')}")


def formatta_percorsi(array_percorsi, cartella_radice="data_preprocessed"):
    """Assicura che i percorsi siano corretti per il sistema operativo corrente"""
    nuovi_percorsi = []
    for percorso in array_percorsi:
        parti = percorso.split(cartella_radice)
        if len(parti) > 1:
            rel_path = cartella_radice + parti[-1].replace('\\', '/')
            nuovi_percorsi.append(os.path.join(BASE_DIR, rel_path))
    return nuovi_percorsi


def calculate_mean_image(paths):
    """Calcola l'immagine media iterando per non saturare la RAM"""
    if len(paths) == 0:
        return None

    # Leggiamo la prima immagine per capire le dimensioni (es. 224x224)
    first_img = np.load(paths[0])
    sum_img = np.zeros_like(first_img, dtype=np.float64)

    valid_count = 0
    for path in paths:
        try:
            img = np.load(path)
            sum_img += img
            valid_count += 1
        except Exception as e:
            print(f"  Errore lettura {path}: {e}")

    if valid_count == 0:
        return None

    return (sum_img / valid_count)


def analyze_image_intensities():
    """Analisi Voxel-by-Voxel: Differenza tra Sani (CN) e Alzheimer (AD)"""
    print("\n--- 2. Calcolo Mappe Differenziali (Immagini 2D) ---")
    if not os.path.exists(MANIFEST_CSV):
        print("Errore: Manifest CSV non trovato.")
        return

    df_manifest = pd.read_csv(MANIFEST_CSV)
    df_manifest['path_2d_corretto'] = formatta_percorsi(df_manifest['path_2d'].tolist())

    # Estraiamo i percorsi separati per classe
    paths_cn = df_manifest[df_manifest['label'] == 0]['path_2d_corretto'].tolist()
    paths_ad = df_manifest[df_manifest['label'] == 2]['path_2d_corretto'].tolist()

    print(f"Calcolo immagine media per {len(paths_cn)} Sani (CN)...")
    mean_cn = calculate_mean_image(paths_cn)

    print(f"Calcolo immagine media per {len(paths_ad)} Alzheimer (AD)...")
    mean_ad = calculate_mean_image(paths_ad)

    if mean_cn is None or mean_ad is None:
        print("Errore: Impossibile calcolare le medie. Controlla i file .npy.")
        return

    # Calcoliamo la differenza (Sani - Malati)
    # Nelle MRI, il liquor cerebrospinale (nei ventricoli) è nero (basso valore),
    # la materia cerebrale è grigia/bianca (alto valore).
    # Se i ventricoli si allargano (atrofia AD), i pixel che prima erano grigi diventano neri.
    # Quindi Mean_CN (grigio) - Mean_AD (nero) = Valore Positivo (Atrofia).
    difference_map = mean_cn - mean_ad

    # --- VISUALIZZAZIONE ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1. Medio Sano
    im0 = axes[0].imshow(mean_cn, cmap='gray')
    axes[0].set_title('Media Cervelli Sani (CN)')
    axes[0].axis('off')

    # 2. Medio AD
    im1 = axes[1].imshow(mean_ad, cmap='gray')
    axes[1].set_title('Media Cervelli Alzheimer (AD)')
    axes[1].axis('off')

    # 3. Mappa delle Differenze (Heatmap)
    # Usiamo una colormap "coolwarm":
    # Rosso = Maggiore densità nei Sani (Zona atrofizzata in AD)
    # Blu = Maggiore densità in AD
    vmax = np.percentile(np.abs(difference_map), 99)  # Per bilanciare i colori
    im2 = axes[2].imshow(difference_map, cmap='coolwarm', vmin=-vmax, vmax=vmax)
    axes[2].set_title('Mappa Atrofia (Rosso = Perdita di volume)')
    axes[2].axis('off')
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    img_path = os.path.join(OUT_DIR, 'mri_2d_difference_map.png')
    plt.savefig(img_path, dpi=300, bbox_inches='tight')
    print(f"✅ Mappa differenziale salvata: {img_path}")


if __name__ == "__main__":
    analyze_clinical_thresholds()
    analyze_image_intensities()
    print("\n✅ Analisi radiologica e clinica completata con successo.")
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from models import KFoldAttentionModel

# --- CONFIGURAZIONI ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

LABEL_NAMES = ["Sano (CN)", "Lieve (MCI)", "Alzheimer (AD)"]

# NUOVO ORDINE: AGE [0], EDUC [1], SES [2], MMSE [3], CDR [4], eTIV [5], nWBV [6], ASF [7]
IDX_MMSE = 3
IDX_NWBV = 6


class SinglePatientWrapper(torch.nn.Module):
    def __init__(self, model, x_tab, x_3d):
        super().__init__()
        self.model = model
        self.x_tab = x_tab
        self.x_3d = x_3d

    def forward(self, x_2d):
        return self.model(self.x_tab, x_2d, self.x_3d)


def draw_bg_map(models, patient_data_norm, x_2d, x_3d, device, ax):
    """Disegna la mappa decisionale di sfondo aggiornata per i nuovi indici"""
    val_range = np.linspace(-3.0, 3.0, 15)
    MMSE_grid, NWBV_grid = np.meshgrid(val_range, val_range)
    grid_preds = np.zeros_like(MMSE_grid)

    with torch.no_grad():
        for i in range(MMSE_grid.shape[0]):
            for j in range(MMSE_grid.shape[1]):
                clone = patient_data_norm.copy()
                clone[IDX_MMSE] = MMSE_grid[i, j]  # Aggiornato al nuovo indice [3]
                clone[IDX_NWBV] = NWBV_grid[i, j]  # Aggiornato al nuovo indice [6]
                x_t = torch.tensor(clone, dtype=torch.float32).unsqueeze(0).to(device)

                probs = [torch.softmax(m(x_t, x_2d, x_3d), dim=1).cpu().numpy()[0] for m in models]
                grid_preds[i, j] = np.argmax(np.mean(probs, axis=0))

    cmap_ai = ListedColormap(['#a7f3d0', '#fef08a', '#fecaca'])
    ax.contourf(MMSE_grid, NWBV_grid, grid_preds, alpha=0.4, cmap=cmap_ai)


def process_patient_row(models, data_norm, p_2d, p_3d, device, axes_row, label_true):
    x_tab = torch.tensor(data_norm, dtype=torch.float32).unsqueeze(0).to(device)
    x_2d = torch.tensor(np.load(p_2d), dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device) / 255.0
    x_3d = torch.tensor(np.load(p_3d), dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    # 1. Inferenza
    probs = []
    with torch.no_grad():
        for m in models:
            probs.append(torch.softmax(m(x_tab, x_2d, x_3d), dim=1).cpu().numpy()[0])
    probs = np.array(probs)
    mean_p = np.mean(probs, axis=0)
    std_p = np.std(probs, axis=0)
    pred_idx = np.argmax(mean_p)

    # Colonna 1: Mappa Decisionale
    ax_map = axes_row[0]
    draw_bg_map(models, data_norm, x_2d, x_3d, device, ax_map)
    # Aggiornato scatter plot per estrarre dagli indici 3 e 6
    ax_map.scatter(data_norm[IDX_MMSE], data_norm[IDX_NWBV], color='black', s=120, edgecolors='white', zorder=5)
    ax_map.set_xlim(-3, 3)
    ax_map.set_ylim(-3, 3)
    ax_map.set_title(f"Target: {label_true}\nPred: {LABEL_NAMES[pred_idx]}", fontsize=9)
    ax_map.set_xlabel("MMSE (Norm)", fontsize=7)
    ax_map.set_ylabel("nWBV (Norm)", fontsize=7)

    # Colonne 2, 3, 4: Grad-CAM
    best_model = models[0]
    target_layers = [best_model.enc_2d.model.layer4[-1]]
    wrapped_model = SinglePatientWrapper(best_model, x_tab, x_3d)
    cam = GradCAM(model=wrapped_model, target_layers=target_layers)

    img_orig = x_2d.cpu().numpy().squeeze()
    img_norm = (img_orig - np.min(img_orig)) / (np.max(img_orig) - np.min(img_orig) + 1e-8)
    img_rgb = np.stack((img_norm,) * 3, axis=-1)

    for c_idx in range(3):
        ax_cam = axes_row[1 + c_idx]
        grayscale_cam = cam(input_tensor=x_2d, targets=[ClassifierOutputTarget(c_idx)])[0, :]
        cam_image = show_cam_on_image(img_rgb, grayscale_cam, use_rgb=True)
        ax_cam.imshow(cam_image)
        ax_cam.set_title(f"Grad-CAM se {LABEL_NAMES[c_idx]}", fontsize=8)
        ax_cam.axis('off')

    # Colonna 5: Intervallo di Confidenza
    ax_bar = axes_row[4]
    x_pos = np.arange(3)
    ax_bar.bar(x_pos, mean_p * 100, yerr=std_p * 100, color=['#a7f3d0', '#fef08a', '#fecaca'], edgecolor='black',
               capsize=5)
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(["CN", "MCI", "AD"], fontsize=8)
    ax_bar.set_ylabel("Probabilità (%)", fontsize=7)
    ax_bar.set_ylim(0, 110)
    max_std = np.max(std_p) * 100
    conf_status = "Alta" if max_std < 10 else "Media" if max_std < 20 else "Bassa"
    ax_bar.set_title(f"Confidenza: {conf_status}\n(±{max_std:.1f}%)", fontsize=8)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🧠 Generazione del Dossier Diagnostico Multimodale...")

    models = []
    for i in range(1, 6):
        ckpt_path = os.path.join(CKPT_DIR, f"best_model_fold{i}.pth")
        if os.path.exists(ckpt_path):
            m = KFoldAttentionModel(emb_dim=64).to(device)
            m.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
            m.eval()
            models.append(m)

    if not models:
        print("❌ Modelli non trovati. Esegui prima il training!")
        return

    fig, axes = plt.subplots(3, 5, figsize=(18, 12))
    fig.suptitle("DOSSIER CLINICO STRUTTURATO: ANALISI MULTIMODALE ENSEMBLE AI", fontsize=16, fontweight='bold')

    # --- DATI RI-ORDINATI: AGE, EDUC, SES, MMSE, CDR, eTIV, nWBV, ASF ---

    # Riga 1: Sano (CN)
    pat_cn = [0.4325259402118016,-0.864283241431904,1.3545037761188354,0.2088371570561916,0.0,1.0980413820601942,-0.5016157480145673,-1.0962476323681731
]
    file_2d_cn = "../data_preprocessed/slices_2d/OAS2_0005_MR1.npy"
    file_3d_cn = "../data_preprocessed/volumes_3d/OAS2_0005_MR1.npy"

    # Riga 2: Lieve (MCI)
    pat_mci = [-0.2250700593862305,-0.864283241431904,-0.4695975015401976,-1.1051943929602943,0.5,1.0407710947329227,0.1647296796199413,-1.0509089522590496
]
    file_2d_mci = "../data_preprocessed/slices_2d/OAS2_0002_MR1.npy"
    file_3d_mci = "../data_preprocessed/volumes_3d/OAS2_0002_MR1.npy"

    # Riga 3: Alzheimer (AD) - Dati riorganizzati in base al nuovo ordine
    pat_ad = [-1.1457044588234755,-0.1736787964351862,1.3545037761188354,-1.6308070129668888,1.0,-0.8816516405161445,-1.2125290968579374,0.8748479321879219
]
    file_2d_ad = "../data_preprocessed/slices_2d/OAS2_0044_MR1.npy"
    file_3d_ad = "../data_preprocessed/volumes_3d/OAS2_0044_MR1.npy"

    print("Processamento Riga 1: Paziente Sano (CN)...")
    if os.path.exists(file_2d_cn): process_patient_row(models, pat_cn, file_2d_cn, file_3d_cn, device, axes[0],
                                                       "Vero Sano (CN)")

    print("Processamento Riga 2: Paziente Borderline (MCI)...")
    if os.path.exists(file_2d_mci): process_patient_row(models, pat_mci, file_2d_mci, file_3d_mci, device, axes[1],
                                                        "Vero Lieve (MCI)")

    print("Processamento Riga 3: Paziente Alzheimer (AD)...")
    if os.path.exists(file_2d_ad): process_patient_row(models, pat_ad, file_2d_ad, file_3d_ad, device, axes[2],
                                                       "Vero Alzheimer (AD)")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    dossier_path = os.path.join(OUT_DIR, "dossier_clinico_multimodale.png")
    plt.savefig(dossier_path, dpi=300, bbox_inches='tight')
    print(f"\n🏆 COMPILAZIONE COMPLETATA! Dossier salvato in: {dossier_path}")


if __name__ == "__main__":
    main()
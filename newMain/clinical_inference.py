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

# --- VARIABILE GLOBALE PER L'ATTENZIONE ---
captured_attention_weights = None


def attention_hook(module, input, output):
    global captured_attention_weights
    captured_attention_weights = output[1].detach().cpu().numpy()[0, 0]


class SinglePatientWrapper(torch.nn.Module):
    def __init__(self, model, x_tab, x_3d):
        super().__init__()
        self.model = model
        self.x_tab = x_tab
        self.x_3d = x_3d

    def forward(self, x_2d):
        return self.model(self.x_tab, x_2d, self.x_3d)


def generate_dynamic_decision_map(models, patient_data_norm, x_2d, x_3d, device, ax):
    """Genera lo sfondo colorato in base all'AI (Dati Normalizzati)"""
    val_range = np.linspace(-3.0, 3.0, 20)
    MMSE_grid_norm, NWBV_grid_norm = np.meshgrid(val_range, val_range)
    grid_predictions = np.zeros_like(MMSE_grid_norm)

    with torch.no_grad():
        for i in range(MMSE_grid_norm.shape[0]):
            for j in range(MMSE_grid_norm.shape[1]):
                clone_data = patient_data_norm.copy()
                clone_data[2] = MMSE_grid_norm[i, j]
                clone_data[4] = NWBV_grid_norm[i, j]

                x_tab_clone = torch.tensor(clone_data, dtype=torch.float32).unsqueeze(0).to(device)

                probs = []
                for model in models:
                    logits = model(x_tab_clone, x_2d, x_3d)
                    probs.append(torch.softmax(logits, dim=1).cpu().numpy()[0])

                mean_probs = np.mean(probs, axis=0)
                grid_predictions[i, j] = np.argmax(mean_probs)

    cmap_ai = ListedColormap(['#a7f3d0', '#fef08a', '#fecaca'])
    ax.contourf(MMSE_grid_norm, NWBV_grid_norm, grid_predictions, alpha=0.5, cmap=cmap_ai)
    return ax


def generate_ultimate_report(patient_data_norm, path_2d, path_3d, patient_id="Paziente_001"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 Generazione REFERTO SUPREMO per {patient_id}...")

    # 1. Caricamento Dati
    x_tab = torch.tensor(patient_data_norm, dtype=torch.float32).unsqueeze(0).to(device)
    x_2d = torch.tensor(np.load(path_2d), dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device) / 255.0
    x_3d = torch.tensor(np.load(path_3d), dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    # 2. Caricamento Modelli e Hook
    models = []
    for i in range(1, 6):
        ckpt_path = os.path.join(CKPT_DIR, f"best_model_fold{i}.pth")
        if os.path.exists(ckpt_path):
            m = KFoldAttentionModel(emb_dim=64).to(device)
            m.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
            m.eval()
            models.append(m)

    if not models:
        print("❌ Errore: Modelli non trovati. Hai eseguito il training?")
        return

    hook_handle = models[0].fusion.attn.register_forward_hook(attention_hook)

    # 3. Inferenza Ensemble
    ensemble_probs = []
    with torch.no_grad():
        for m in models:
            logits = m(x_tab, x_2d, x_3d)
            ensemble_probs.append(torch.softmax(logits, dim=1).cpu().numpy()[0])

    hook_handle.remove()

    # Statistiche
    probs_matrix = np.array(ensemble_probs)
    mean_probs = np.mean(probs_matrix, axis=0)
    std_probs = np.std(probs_matrix, axis=0)
    pred_idx = np.argmax(mean_probs)
    diagnosi = LABEL_NAMES[pred_idx]

    # Attenzione
    weight_2d = captured_attention_weights[0] * 100 if captured_attention_weights is not None else 50
    weight_3d = captured_attention_weights[1] * 100 if captured_attention_weights is not None else 50

    # 4. Grad-CAM
    print("  > Calcolo Grad-CAM...")
    best_model = models[0]
    target_layers = [best_model.enc_2d.model.layer4[-1]]
    wrapped_model = SinglePatientWrapper(best_model, x_tab, x_3d)
    cam = GradCAM(model=wrapped_model, target_layers=target_layers)

    grayscale_cam = cam(input_tensor=x_2d, targets=[ClassifierOutputTarget(pred_idx)])[0, :]
    img_original = x_2d.cpu().numpy().squeeze()
    img_normalized = (img_original - np.min(img_original)) / (np.max(img_original) - np.min(img_original) + 1e-8)
    img_rgb = np.stack((img_normalized,) * 3, axis=-1)
    cam_image = show_cam_on_image(img_rgb, grayscale_cam, use_rgb=True)

    # ==========================================
    # 5. IMPAGINAZIONE DEL REFERTO (2 Righe, 4 Grafici)
    # ==========================================
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f"REFERTO CDSS MULTIMODALE AI AVANZATO - {patient_id}", fontsize=20, fontweight='bold')

    # --- RIGA 1 ---
    # R1.C1: Mappa Decisionale
    ax1 = plt.subplot(2, 2, 1)
    ax1 = generate_dynamic_decision_map(models, patient_data_norm, x_2d, x_3d, device, ax1)
    ax1.scatter(patient_data_norm[2], patient_data_norm[4], color='black', s=250, edgecolors='white', linewidth=2,
                zorder=5)
    ax1.set_xlabel("MMSE (Valore Normalizzato Z-score)")
    ax1.set_ylabel("nWBV (Valore Normalizzato Z-score)")
    ax1.set_title("Mappa Rischio AI & Posizione Paziente")

    # R1.C2: Grad-CAM
    ax2 = plt.subplot(2, 2, 2)
    ax2.imshow(cam_image)
    ax2.set_title("Biomarcatori Visivi Rilevati (Grad-CAM)")
    ax2.axis('off')

    # --- RIGA 2 ---
    # R2.C1: Affidabilità e Incertezza
    ax3 = plt.subplot(2, 3, 4)  # Usiamo una griglia 2x3 per la riga inferiore per centrare meglio
    x_pos = np.arange(len(LABEL_NAMES))
    axes_colors = ['#a7f3d0', '#fef08a', '#fecaca']
    ax3.bar(x_pos, mean_probs * 100, yerr=std_probs * 100, align='center',
            alpha=0.8, ecolor='black', capsize=10, color=axes_colors, edgecolor='black')
    ax3.set_ylabel('Probabilità Media (%)')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(LABEL_NAMES)
    ax3.set_title('Incertezza Diagnostica (Ensemble Agreement)')
    ax3.set_ylim(0, 110)

    # R2.C2: Donut Chart
    ax4 = plt.subplot(2, 3, 5)
    ax4.pie([weight_2d, weight_3d], labels=['Risonanza 2D', 'Volume 3D'], colors=['#3b82f6', '#8b5cf6'],
            autopct='%1.1f%%', startangle=90, wedgeprops=dict(width=0.4, edgecolor='w'))
    ax4.set_title('Sorgente Attenzione Visiva')

    # R2.C3: Verdetto Testuale
    ax5 = plt.subplot(2, 3, 6)
    ax5.axis('off')
    max_std = np.max(std_probs) * 100
    affidabilita = "ALTA (Accordo Modelli)" if max_std < 10 else "MEDIA" if max_std < 20 else "BASSA (Discordanza)"

    testo = f"🩺 DIAGNOSI FINALE:\n[{diagnosi}]\n\n"
    testo += f"Affidabilità: {affidabilita}\n\n"
    testo += f"Confidenza Ensemble:\n"
    for i, nome in enumerate(LABEL_NAMES):
        testo += f"- {nome}: {mean_probs[i] * 100:.1f}%\n"

    ax5.text(0.1, 0.5, testo, fontsize=14, verticalalignment='center',
             bbox=dict(facecolor='#f8fafc', alpha=0.9, boxstyle='round,pad=1', edgecolor='#3b82f6', linewidth=2))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path = os.path.join(OUT_DIR, f"referto_completo_{patient_id}.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"✅ REFERTO SUPREMO SALVATO: {out_path}")


if __name__ == "__main__":
    dati_paziente_norm = [1.3531603396490466, -0.1736787964351862, -0.0539691529471055,
                          2.75643083929121, -0.9147209346079822, -2.2040606170408124,
                          -0.4695975015401976, 0.0]

    # Sostituisci i percorsi con i tuoi file veri!
    file_2d = "../data_preprocessed/slices_2d/OAS2_0001_MR1.npy"  # <-- SOSTITUISCI QUESTO
    file_3d = "../data_preprocessed/volumes_3d/OAS2_0001_MR1.npy"  # <-- SOSTITUISCI QUESTO

    if os.path.exists(file_2d) and os.path.exists(file_3d):
        generate_ultimate_report(dati_paziente_norm, file_2d, file_3d, "Paziente_RealTime")
    else:
        print("\n⚠️ Inserisci percorsi validi in fondo al codice.")

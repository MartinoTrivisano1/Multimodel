import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import f1_score

# Importiamo dal nostro nuovo sistema modulare
from dataset import OASISFullDataset, MANIFEST_CSV, FEATURE_COLS, LABEL_NAMES
from models import KFoldAttentionModel

# Grad-CAM imports
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# --- Configurazione ---
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(PROJECT_DIR, "checkpoints")


# ==========================================
# 1. ANALISI DATI TABELLARI
# ==========================================
def analyze_tabular_importance(model, test_loader, device):
    print("\n--- 1. Permutation Feature Importance (Dati Clinici) ---")
    model.eval()

    # Calcolo F1 Baseline
    all_targets, all_preds = [], []
    with torch.no_grad():
        for x_tab, x_2d, x_3d, y in test_loader:
            out = model(x_tab.to(device), x_2d.to(device), x_3d.to(device))
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_targets.extend(y.numpy())

    baseline_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    print(f"F1 Baseline: {baseline_f1:.4f}")

    drops = {}
    for col_idx, feature_name in enumerate(FEATURE_COLS):
        all_targets, all_preds = [], []
        with torch.no_grad():
            for x_tab, x_2d, x_3d, y in test_loader:
                x_tab, x_2d, x_3d = x_tab.to(device), x_2d.to(device), x_3d.to(device)

                # Mescoliamo solo questa colonna
                x_tab_shuffled = x_tab.clone()
                shuffled_indices = torch.randperm(x_tab.size(0))
                x_tab_shuffled[:, col_idx] = x_tab[shuffled_indices, col_idx]

                logits = model(x_tab_shuffled, x_2d, x_3d)
                all_preds.extend(logits.argmax(1).cpu().numpy())
                all_targets.extend(y.numpy())

        shuffled_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
        drop = baseline_f1 - shuffled_f1
        drops[feature_name] = drop
        print(f"  > {feature_name:5s} mescolata -> F1: {shuffled_f1:.4f} (Drop: {drop:+.4f})")

    # Grafico
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(drops.values()), y=list(drops.keys()), palette="Reds_r")
    plt.title("Importanza delle Feature Tabellari")
    plt.xlabel("Perdita di F1-Score")
    plt.tight_layout()
    plt.savefig('outputs/tabular_importance.png')
    print("Salvato: 'tabular_importance.png'")


# ==========================================
# 2. ANALISI IMMAGINI (GRAD-CAM)
# ==========================================
class ModelWrapper(torch.nn.Module):
    def __init__(self, model, x_tab, x_3d):
        super().__init__()
        self.model = model
        self.x_tab = x_tab
        self.x_3d = x_3d

    def forward(self, x_2d):
        return self.model(self.x_tab, x_2d, self.x_3d)


def generate_gradcam(model, dataset, test_indices, device):
    print("\n--- 2. Generazione Grad-CAM (MRI 2D) ---")
    model.eval()

    # Layer bersaglio della ResNet18
    target_layers = [model.enc_2d.model.layer4[-1]]

    # Selezioniamo i primi 3 pazienti del test set per l'esempio
    patients_to_analyze = [
        ("Paziente 1", test_indices[0]),
        ("Paziente 2", test_indices[1]),
        ("Paziente 3", test_indices[2])
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Grad-CAM: Focus della Rete sull'Atrofia Cerebrale", fontsize=16)

    for i, (title, idx) in enumerate(patients_to_analyze):
        x_tab, x_2d, x_3d, label = dataset[idx]
        x_tab, x_2d, x_3d = x_tab.unsqueeze(0).to(device), x_2d.unsqueeze(0).to(device), x_3d.unsqueeze(0).to(device)
        true_class = label.item()

        wrapped_model = ModelWrapper(model, x_tab, x_3d)
        cam = GradCAM(model=wrapped_model, target_layers=target_layers)

        targets = [ClassifierOutputTarget(true_class)]
        grayscale_cam = cam(input_tensor=x_2d, targets=targets)[0, :]

        img_original = x_2d.cpu().numpy().squeeze()
        img_normalized = (img_original - np.min(img_original)) / (np.max(img_original) - np.min(img_original) + 1e-8)
        img_rgb = np.stack((img_normalized,) * 3, axis=-1)

        visualization = show_cam_on_image(img_rgb, grayscale_cam, use_rgb=True)

        axes[i].imshow(visualization)
        axes[i].set_title(f"{title}\nReale: {LABEL_NAMES[true_class]}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig('outputs/gradcam_analysis.png')
    print("Salvato: 'gradcam_analysis.png'")


# ==========================================
# MAIN ROUTINE
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = OASISFullDataset()

    manifest = pd.read_csv(MANIFEST_CSV)
    test_indices = manifest[manifest["split"] == "test"].index.tolist()
    test_loader = DataLoader(Subset(dataset, test_indices), batch_size=len(test_indices), shuffle=False)

    # Carichiamo il miglior modello del Fold 1
    ckpt_path = os.path.join(CKPT_DIR, "best_model_fold1.pth")
    if not os.path.exists(ckpt_path):
        print(f"Errore: Modello {ckpt_path} non trovato. Esegui prima train.py!")
        return

    model = KFoldAttentionModel(emb_dim=64).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))

    analyze_tabular_importance(model, test_loader, device)
    generate_gradcam(model, dataset, test_indices, device)


if __name__ == "__main__":
    main()
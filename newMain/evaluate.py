import os
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import classification_report, f1_score, confusion_matrix

# Importiamo i nostri moduli
from dataset import OASISFullDataset, MANIFEST_CSV, LABEL_NAMES
from models import KFoldAttentionModel


def get_ensemble_prediction(models, x_tab, x_2d, x_3d):
    """Calcola la media delle probabilità di tutti i modelli (Soft Voting)"""
    all_probs = []
    with torch.no_grad():
        for model in models:
            logits = model(x_tab, x_2d, x_3d)
            all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.mean(all_probs, axis=0)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = OASISFullDataset()

    # Identifica il Test Set originale
    manifest = pd.read_csv(MANIFEST_CSV)
    test_idx = manifest[manifest["split"] == "test"].index.tolist()
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=1, shuffle=False)

    # Carica i 5 modelli addestrati durante il K-Fold
    ensemble = []
    for i in range(1, 6):
        path = f"checkpoints/best_model_fold{i}.pth"
        if os.path.exists(path):
            m = KFoldAttentionModel(64).to(device)
            m.load_state_dict(torch.load(path, map_location=device, weights_only=True))
            m.eval()
            ensemble.append(m)

    if not ensemble:
        print("❌ Errore: Nessun modello trovato in 'checkpoints/'. Esegui prima il training.")
        return

    print(f"🩺 Valutazione Ensemble su {len(test_idx)} pazienti del Test Set...")
    y_true, y_pred = [], []

    for xt, x2, x3, y in test_loader:
        probs = get_ensemble_prediction(ensemble, xt.to(device), x2.to(device), x3.to(device))
        y_pred.append(np.argmax(probs))
        y_true.append(y.item())

    # --- Calcolo Metriche Finali ---
    f1_macro = f1_score(y_true, y_pred, average='macro')

    print("\n" + "=" * 50)
    print("   RISULTATI FINALI ENSEMBLE AI")
    print("=" * 50)
    print(f"F1-Score Macro: {f1_macro:.4f}")
    print("\nReport Dettagliato:")
    print(classification_report(y_true, y_pred, target_names=LABEL_NAMES))

    # --- Generazione Matrice di Confusione ---
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES)
    plt.xlabel('Diagnosi Predetta dall\'AI')
    plt.ylabel('Diagnosi Reale (Gold Standard)')
    plt.title(f'Matrice di Confusione Ensemble (F1: {f1_macro:.2f})')

    # Salvataggio del grafico
    os.makedirs("../outputs", exist_ok=True)
    plt.savefig('outputs/ensemble_confusion_matrix.png')
    print("\n📊 Matrice di confusione salvata in 'outputs/ensemble_confusion_matrix.png'")


if __name__ == "__main__":
    main()
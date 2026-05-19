import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import optuna
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score

# Importiamo dai nostri moduli
from dataset import OASISFullDataset
from models import KFoldAttentionModel


def objective(trial):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = OASISFullDataset(augment=True)

    # 1. OPTUNA SUGGERISCE I PARAMETRI
    # Chiediamo a Optuna di testare diversi Learning Rate su scala logaritmica
    lr_images = trial.suggest_float("lr_images", 1e-6, 1e-3, log=True)
    lr_tab = trial.suggest_float("lr_tab", 1e-6, 1e-3, log=True)
    lr_fusion = trial.suggest_float("lr_fusion", 1e-5, 1e-2, log=True)

    # Testiamo la regolarizzazione (Weight Decay)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True)

    # Testiamo i pesi della Loss per capire come bilanciare le classi
    weight_mci = trial.suggest_float("weight_mci", 1.0, 2.5)
    weight_ad = trial.suggest_float("weight_ad", 2.0, 4.0)

    # Impostiamo un K-Fold ridotto (3 splits) per velocizzare la ricerca
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    fold_f1s = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(dataset)))):
        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=4, shuffle=True, drop_last=True)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=4, shuffle=False)

        # Inizializziamo il modello (potremmo far scegliere a Optuna anche l'emb_dim!)
        model = KFoldAttentionModel(emb_dim=64).to(device)

        # Assegniamo i parametri suggeriti da Optuna all'ottimizzatore
        optimizer = optim.Adam([
            {'params': model.enc_2d.parameters(), 'lr': lr_images},
            {'params': model.enc_3d.parameters(), 'lr': lr_images},
            {'params': model.enc_tab.parameters(), 'lr': lr_tab},
            {'params': model.fusion.parameters(), 'lr': lr_fusion},
            {'params': model.classifier.parameters(), 'lr': lr_fusion}
        ], weight_decay=weight_decay)

        # Assegniamo i pesi suggeriti per la Loss
        class_weights = torch.tensor([0.6, weight_mci, weight_ad]).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

        best_f1_fold = 0

        # Solo 10 epoche per trial (Fast Search)
        for epoch in range(10):
            model.train()
            for xt, x2, x3, y in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(xt.to(device), x2.to(device), x3.to(device)), y.to(device))
                loss.backward()
                optimizer.step()

            model.eval()
            va_preds, va_targets = [], []
            with torch.no_grad():
                for xt, x2, x3, y in val_loader:
                    out = model(xt.to(device), x2.to(device), x3.to(device))
                    va_preds.extend(out.argmax(1).cpu().numpy())
                    va_targets.extend(y.numpy())

            f1 = f1_score(va_targets, va_preds, average='macro', zero_division=0)
            if f1 > best_f1_fold:
                best_f1_fold = f1

        fold_f1s.append(best_f1_fold)

    # Optuna cerca di massimizzare questo valore di ritorno (la media dei 3 fold)
    return np.mean(fold_f1s)


def main():
    print("=" * 60)
    print(" 🤖 AVVIO OPTUNA HYPERPARAMETER OPTIMIZATION")
    print("=" * 60)

    # Creiamo uno "Study" dicendo a Optuna che vogliamo massimizzare l'F1-Score
    study = optuna.create_study(direction="maximize", study_name="Alzheimer_Multimodal")

    # Diciamo a Optuna di fare 20 tentativi (Trials)
    study.optimize(objective, n_trials=20)

    print("\n" + "=" * 60)
    print(" 🏆 RICERCA COMPLETATA! I MIGLIORI PARAMETRI SONO:")
    print("=" * 60)

    best_params = study.best_params
    for key, value in best_params.items():
        print(f" - {key}: {value:.6f}")

    print(f"\nMiglior F1-Score Stimato: {study.best_value:.4f}")
    print("\nOra copia questi parametri nel tuo train.py e lancia l'addestramento finale a 5-Fold e 20 Epoche!")


if __name__ == "__main__":
    main()
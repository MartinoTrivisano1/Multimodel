import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, accuracy_score

# Importiamo i nostri moduli personalizzati
from dataset import OASISFullDataset
from models import KFoldAttentionModel


def run_kfold():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Avvio Training su: {device}")

    dataset = OASISFullDataset()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    os.makedirs("../checkpoints", exist_ok=True)

    fold_f1s = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(dataset)))):
        print(f"\n" + "=" * 60)
        print(f"   FOLD {fold + 1}/5")
        print("=" * 60)

        dataset_train = OASISFullDataset(augment=True)
        dataset_val = OASISFullDataset(augment=False)

        # I loader pescano dagli indici corretti ma dai dataset con/senza augmentation
        train_loader = DataLoader(Subset(dataset_train, train_idx), batch_size=4, shuffle=True, drop_last=True)
        val_loader = DataLoader(Subset(dataset_val, val_idx), batch_size=4, shuffle=False)

        # Definizione del modello
        model = KFoldAttentionModel(emb_dim=64).to(device)

        # Pesi delle classi (Optuna)
        class_weights = torch.tensor([0.6, 2.212236, 2.466673]).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

        # ==========================================
        # FASE 1: CONGELAMENTO (Warm-up della fusione)
        # ==========================================
        for param in model.enc_2d.parameters(): param.requires_grad = False
        for param in model.enc_3d.parameters(): param.requires_grad = False

        # Ottimizzatore Fase 1 (Impara solo la tabella e la fusione)
        optimizer = optim.Adam([
            {'params': model.enc_tab.parameters(), 'lr': 1e-4},
            {'params': model.fusion.parameters(), 'lr': 0.009},
            {'params': model.classifier.parameters(), 'lr': 0.009}
        ], weight_decay=0.014398)

        best_f1 = 0
        for epoch in range(1, 21):

            # ==========================================
            # FASE 2: SCONGELAMENTO (Fine-Tuning con Optuna)
            # ==========================================
            if epoch == 6:
                print("  🔓 SCONGELAMENTO: Avvio Fase 2 (Fine-Tuning Globale)")
                for param in model.enc_2d.parameters(): param.requires_grad = True
                for param in model.enc_3d.parameters(): param.requires_grad = True

                # Nuovo ottimizzatore con i parametri chirurgici di Optuna
                optimizer = optim.Adam([
                    {'params': model.enc_2d.parameters(), 'lr': 0.000052},
                    {'params': model.enc_3d.parameters(), 'lr': 0.000052},
                    {'params': model.enc_tab.parameters(), 'lr': 0.000017},
                    {'params': model.fusion.parameters(), 'lr': 0.009},
                    {'params': model.classifier.parameters(), 'lr': 0.009}
                ], weight_decay=0.014398)

            # --- TRAINING ---
            model.train()
            tr_loss = 0.0
            tr_preds, tr_targets = [], []

            for xt, x2, x3, y in train_loader:
                xt, x2, x3, y = xt.to(device), x2.to(device), x3.to(device), y.to(device)
                optimizer.zero_grad()
                out = model(xt, x2, x3)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()

                tr_loss += loss.item() * xt.size(0)
                tr_preds.extend(out.argmax(1).cpu().numpy())
                tr_targets.extend(y.cpu().numpy())

            tr_loss /= len(train_idx)
            tr_f1 = f1_score(tr_targets, tr_preds, average='macro', zero_division=0)

            # --- VALIDAZIONE ---
            model.eval()
            va_loss = 0.0
            va_preds, va_targets = [], []

            with torch.no_grad():
                for xt, x2, x3, y in val_loader:
                    xt, x2, x3, y = xt.to(device), x2.to(device), x3.to(device), y.to(device)
                    out = model(xt, x2, x3)
                    loss = criterion(out, y)
                    va_loss += loss.item() * xt.size(0)
                    va_preds.extend(out.argmax(1).cpu().numpy())
                    va_targets.extend(y.cpu().numpy())

            va_loss /= len(val_idx)
            va_f1 = f1_score(va_targets, va_preds, average='macro', zero_division=0)

            print(
                f"Epoca {epoch:02d} | Tr Loss: {tr_loss:.4f} | Tr F1: {tr_f1:.3f} || Va Loss: {va_loss:.4f} | Va F1: {va_f1:.3f}")

            if va_f1 > best_f1:
                best_f1 = va_f1
                torch.save(model.state_dict(), f"checkpoints/best_model_fold{fold + 1}.pth")
                print(f"  ↳ 💾 Miglior modello salvato! (F1: {best_f1:.4f})")

    print("\n" + "=" * 60)
    print(f"🏆 PERFORMANCE MEDIA K-FOLD: {np.mean(fold_f1s):.4f} +/- {np.std(fold_f1s):.4f}")
    print("=" * 60)


if __name__ == "__main__":
    run_kfold()


import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from multimodal_dataset import OASISMultimodalDataset
from early_fusion_model import EarlyFusionModel

# ── Configurazione ────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.normpath(os.path.join(BASE_DIR, ".."))
CKPT_DIR    = os.path.join(PROJECT_DIR, "checkpoints")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
HP_FILE     = os.path.join(RESULTS_DIR, "best_hyperparametri.json")

# Tuning
N_TRIALS    = 30
EPOCHS_HP   = 15
PATIENCE_HP = 5

# Training finale
EPOCHS_FINAL   = 50
PATIENCE_FINAL = 10

LABEL_NAMES = ["CN", "MCI", "AD"]


# ── Dispositivo ───────────────────────────────────────────────────────────────
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── DataLoaders ───────────────────────────────────────────────────────────────
def get_dataloaders(batch_size):
    train_ds = OASISMultimodalDataset(split="train")
    val_ds   = OASISMultimodalDataset(split="val")
    test_ds  = OASISMultimodalDataset(split="test")

    class_weights  = train_ds.get_class_weights()
    sample_weights = class_weights[train_ds.labels]
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights, num_samples=len(train_ds), replacement=True
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        sampler=sampler, num_workers=0, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=0
    )
    return train_loader, val_loader, test_loader


# ── Training epoch ────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for x_tab, x_2d, x_3d, y in loader:
        x_tab, x_2d, x_3d, y = (
            x_tab.to(device), x_2d.to(device),
            x_3d.to(device), y.to(device)
        )
        optimizer.zero_grad()
        logits = model(x_tab, x_2d, x_3d)
        loss   = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / len(loader)
    f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    return avg_loss, acc, f1


# ── Eval epoch ────────────────────────────────────────────────────────────────
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x_tab, x_2d, x_3d, y in loader:
            x_tab, x_2d, x_3d, y = (
                x_tab.to(device), x_2d.to(device),
                x_3d.to(device), y.to(device)
            )
            logits = model(x_tab, x_2d, x_3d)
            loss   = criterion(logits, y)

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / len(loader)
    f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    return avg_loss, acc, f1, all_preds, all_labels


# ══════════════════════════════════════════════════════════════════════════════
# FASE 1 — Hyperparameter Tuning
# ══════════════════════════════════════════════════════════════════════════════

def objective(trial):
    device = get_device()

    lr           = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    dropout      = trial.suggest_float("dropout", 0.1, 0.5)
    emb_dim      = trial.suggest_categorical("emb_dim", [64, 128, 256])
    batch_size   = trial.suggest_categorical("batch_size", [4, 8, 16])
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    w_cn         = trial.suggest_float("w_cn",  0.5, 2.0)
    w_mci        = trial.suggest_float("w_mci", 1.0, 3.0)
    w_ad         = trial.suggest_float("w_ad",  2.0, 6.0)

    train_loader, val_loader, _ = get_dataloaders(batch_size)

    model = EarlyFusionModel(
        emb_dim=emb_dim, n_classes=3, dropout=dropout
    ).to(device)

    class_weights = torch.tensor([w_cn, w_mci, w_ad]).to(device)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)
    optimizer     = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler     = ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)

    best_val_f1      = 0.0
    patience_counter = 0

    for epoch in range(EPOCHS_HP):
        train_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_acc, va_f1, _, _ = eval_epoch(
            model, val_loader, criterion, device
        )
        scheduler.step(va_loss)

        if va_f1 > best_val_f1:
            best_val_f1      = va_f1
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE_HP:
                break

        trial.report(va_f1, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_val_f1


def run_tuning():
    print("=" * 60)
    print("FASE 1 — Hyperparameter Tuning con Optuna")
    print(f"         {N_TRIALS} trial × {EPOCHS_HP} epoche ciascuno")
    print("=" * 60)

    sampler = optuna.samplers.TPESampler(seed=42)
    pruner  = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    study   = optuna.create_study(
        direction="maximize", sampler=sampler,
        pruner=pruner, study_name="early_fusion_hp"
    )

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    print(f"\nTrial completati : {len(study.trials)}")
    print(f"Miglior Val F1   : {study.best_value:.4f}")
    print(f"Migliori params  :")
    for k, v in study.best_params.items():
        print(f"  {k:20s}: {v}")

    # Salva tutti i trial
    all_trials = [
        {"trial": t.number, "value": t.value, "params": t.params}
        for t in study.trials if t.value is not None
    ]
    with open(os.path.join(RESULTS_DIR, "hp_tuning_results.json"), "w") as f:
        json.dump(all_trials, f, indent=2)

    # Salva migliori HP con descrizione
    bp = study.best_params
    best_hp = {
        "architettura":     "Early Fusion",
        "metodo_tuning":    "Optuna TPE Sampler",
        "n_trial_eseguiti": len(all_trials),
        "miglior_val_f1":   round(study.best_value, 4),
        "iperparametri": {
            "lr":           {"valore": bp["lr"],           "descrizione": "Learning rate Adam",               "range": "[1e-5, 1e-3]"},
            "dropout":      {"valore": bp["dropout"],      "descrizione": "Dropout rate regolarizzazione",    "range": "[0.1, 0.5]"},
            "emb_dim":      {"valore": bp["emb_dim"],      "descrizione": "Dimensione embedding modalità",    "range": "[64, 128, 256]"},
            "batch_size":   {"valore": bp["batch_size"],   "descrizione": "Dimensione batch training",        "range": "[4, 8, 16]"},
            "weight_decay": {"valore": bp["weight_decay"], "descrizione": "L2 regularization Adam",          "range": "[1e-5, 1e-3]"},
            "w_cn":         {"valore": bp["w_cn"],         "descrizione": "Peso classe CN nella loss",        "range": "[0.5, 2.0]"},
            "w_mci":        {"valore": bp["w_mci"],        "descrizione": "Peso classe MCI nella loss",       "range": "[1.0, 3.0]"},
            "w_ad":         {"valore": bp["w_ad"],         "descrizione": "Peso classe AD nella loss (boost)","range": "[2.0, 6.0]"},
        }
    }

    with open(HP_FILE, "w") as f:
        json.dump(best_hp, f, indent=2)
    print(f"\n  Migliori HP salvati: {HP_FILE}")

    # Grafico tuning
    plot_tuning(study)

    return study.best_params, study.best_value


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2 — Training Finale
# ══════════════════════════════════════════════════════════════════════════════

def run_training(bp, best_val_f1_tuning):
    print("\n" + "=" * 60)
    print("FASE 2 — Training Finale con Migliori Iperparametri")
    print("=" * 60)

    print(f"\nIperparametri usati (da {HP_FILE}):")
    print(f"  lr           : {bp['lr']:.6f}")
    print(f"  dropout      : {bp['dropout']:.4f}")
    print(f"  emb_dim      : {bp['emb_dim']}")
    print(f"  batch_size   : {bp['batch_size']}")
    print(f"  weight_decay : {bp['weight_decay']:.6f}")
    print(f"  w_cn         : {bp['w_cn']:.4f}")
    print(f"  w_mci        : {bp['w_mci']:.4f}")
    print(f"  w_ad         : {bp['w_ad']:.4f}")

    device = get_device()
    print(f"\nDispositivo: {device}")

    train_loader, val_loader, test_loader = get_dataloaders(bp["batch_size"])

    model = EarlyFusionModel(
        emb_dim=bp["emb_dim"], n_classes=3, dropout=bp["dropout"]
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parametri modello: {n_params:,}")

    class_weights = torch.tensor([bp["w_cn"], bp["w_mci"], bp["w_ad"]]).to(device)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)
    optimizer     = optim.Adam(model.parameters(), lr=bp["lr"],
                               weight_decay=bp["weight_decay"])
    scheduler     = ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

    best_val_loss    = float("inf")
    patience_counter = 0
    history          = []

    print(f"\n{'Epoch':>5} {'TrLoss':>8} {'TrAcc':>7} {'TrF1':>7} "
          f"{'VaLoss':>8} {'VaAcc':>7} {'VaF1':>7} {'LR':>10}")
    print("-" * 70)

    for epoch in range(1, EPOCHS_FINAL + 1):
        tr_loss, tr_acc, tr_f1 = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        va_loss, va_acc, va_f1, _, _ = eval_epoch(
            model, val_loader, criterion, device
        )
        scheduler.step(va_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"{epoch:>5} {tr_loss:>8.4f} {tr_acc:>7.3f} {tr_f1:>7.3f} "
              f"{va_loss:>8.4f} {va_acc:>7.3f} {va_f1:>7.3f} {current_lr:>10.2e}")

        history.append({
            "epoch": epoch,
            "train_loss": tr_loss, "train_acc": tr_acc, "train_f1": tr_f1,
            "val_loss":   va_loss, "val_acc":   va_acc, "val_f1":   va_f1,
        })

        if va_loss < best_val_loss:
            best_val_loss    = va_loss
            patience_counter = 0
            torch.save(model.state_dict(),
                       os.path.join(CKPT_DIR, "early_fusion_best.pth"))
            print(f"        ✅ Miglior modello salvato (val_loss={va_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE_FINAL:
                print(f"\nEarly stopping a epoch {epoch}")
                break

    torch.save(model.state_dict(),
               os.path.join(CKPT_DIR, "early_fusion_last.pth"))

    # Test finale
    print("\n" + "=" * 60)
    print("Valutazione su Test Set")
    print("=" * 60)

    model.load_state_dict(
        torch.load(os.path.join(CKPT_DIR, "early_fusion_best.pth"),
                   map_location=device, weights_only=True)
    )

    te_loss, te_acc, te_f1, te_preds, te_labels = eval_epoch(
        model, test_loader, criterion, device
    )

    print(f"\nTest Loss : {te_loss:.4f}")
    print(f"Test Acc  : {te_acc:.3f}")
    print(f"Test F1   : {te_f1:.3f}")
    print(f"\nReport dettagliato:")
    print(classification_report(te_labels, te_preds,
                                target_names=LABEL_NAMES, zero_division=0))

    # Salva risultati
    results = {
        "model":                  "early_fusion_optuna",
        "best_hp":                bp,
        "miglior_val_f1_tuning":  best_val_f1_tuning,
        "test_loss":              te_loss,
        "test_acc":               te_acc,
        "test_f1":                te_f1,
        "history":                history,
    }
    with open(os.path.join(RESULTS_DIR, "early_fusion_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Grafici
    print("\nGenerazione grafici...")
    plot_curves(history)
    plot_confusion(te_labels, te_preds)

    return te_f1


# ══════════════════════════════════════════════════════════════════════════════
# Grafici
# ══════════════════════════════════════════════════════════════════════════════

def plot_tuning(study):
    trials     = [t for t in study.trials if t.value is not None]
    trial_nums = [t.number for t in trials]
    values     = [t.value  for t in trials]

    best_so_far  = []
    current_best = 0
    for v in values:
        current_best = max(current_best, v)
        best_so_far.append(current_best)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Optuna — Hyperparameter Tuning Early Fusion",
                 fontsize=14, fontweight="bold")

    axes[0].scatter(trial_nums, values, alpha=0.6, color="steelblue", s=40)
    axes[0].plot(trial_nums, best_so_far, color="red", linewidth=2, label="Best so far")
    axes[0].set_title("Val F1 Macro per Trial")
    axes[0].set_xlabel("Trial")
    axes[0].set_ylabel("Val F1 Macro")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    try:
        importances = optuna.importance.get_param_importances(study)
        params = list(importances.keys())[:8]
        vals   = [importances[p] for p in params]
        axes[1].barh(params, vals, color="steelblue", alpha=0.7)
        axes[1].set_title("Importanza Iperparametri")
        axes[1].set_xlabel("Importanza relativa")
        axes[1].grid(True, alpha=0.3, axis="x")
    except Exception:
        axes[1].text(0.5, 0.5, "Non disponibile",
                     ha="center", va="center", transform=axes[1].transAxes)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "hp_tuning_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Grafico tuning salvato: {path}")


def plot_curves(history):
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Early Fusion (Best HP) — Training Curves",
                 fontsize=14, fontweight="bold")

    for ax, (k_tr, k_va), title, ylim in zip(
        axes,
        [("train_loss","val_loss"), ("train_acc","val_acc"), ("train_f1","val_f1")],
        ["Loss", "Accuracy", "F1 Macro"],
        [None, (0,1), (0,1)]
    ):
        ax.plot(epochs, [h[k_tr] for h in history], label="Train", marker="o", markersize=3)
        ax.plot(epochs, [h[k_va] for h in history], label="Val",   marker="o", markersize=3)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)
        if ylim:
            ax.set_ylim(*ylim)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "early_fusion_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Curve training salvate: {path}")


def plot_confusion(te_labels, te_preds):
    cm = confusion_matrix(te_labels, te_preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=ax)
    ax.set_title("Early Fusion (Best HP) — Confusion Matrix (Test)",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "early_fusion_confusion.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix salvata: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("Early Fusion — Pipeline Completa")
    print("  FASE 1: HP Tuning con Optuna")
    print("  FASE 2: Training Finale")
    print("=" * 60)

    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # FASE 1
    best_params, best_val_f1 = run_tuning()

    # FASE 2
    test_f1 = run_training(best_params, best_val_f1)

    print("\n" + "=" * 60)
    print("Pipeline completata!")
    print(f"  Miglior Val F1 (tuning)  : {best_val_f1:.4f}")
    print(f"  Test F1 finale           : {test_f1:.4f}")
    print(f"  HP salvati in            : {HP_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()

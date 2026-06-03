"""
evaluation/evaluation.py
=========================
Valutazione completa del modello AlzheimerMultimodalNet su:
  - Val set (OASIS)
  - Test set esterno (100 casi ospedalieri)

Metriche:
  - Accuracy, F1 macro, AUC one-vs-rest
  - Confusion matrix
  - Report per classe (precision, recall, F1)
  - Calibration (ECE – Expected Calibration Error)

Output:
  - outputs/evaluation/{model_name}_report.json
  - outputs/evaluation/{model_name}_confusion.png

Uso:
  python evaluation/evaluation.py --model_cfg 4
  python evaluation/evaluation.py --model_cfg 4 --split test
"""

import os
import sys
import json
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import config as cfg
from dataset.dataset import build_dataloaders
from models.model    import build_model

EVAL_DIR = os.path.join(ROOT, "outputs", "evaluation")
os.makedirs(EVAL_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# UTILITÀ
# ══════════════════════════════════════════════════════════════════════════════

def _move_batch(batch: dict, device: torch.device) -> dict:
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


def _ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """
    Expected Calibration Error.
    Misura quanto le confidence del modello corrispondono
    alla accuracy reale — importante in ambito clinico.
    """
    confidences = probs.max(axis=1)
    preds       = probs.argmax(axis=1)
    correct     = (preds == labels).astype(float)

    bins     = np.linspace(0, 1, n_bins + 1)
    ece      = 0.0
    n        = len(labels)

    for i in range(n_bins):
        mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc  = correct[mask].mean()
        bin_conf = confidences[mask].mean()
        ece     += mask.sum() / n * abs(bin_acc - bin_conf)

    return float(ece)


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_inference(
    model:  nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict:
    """
    Esegue inference su un DataLoader.

    Ritorna
    -------
    dict con:
        labels   : (N,)  ground truth
        preds    : (N,)  predizioni
        probs    : (N, 3) probabilità per classe
        mri_ids  : lista mri_id
        has_flags: dict con has_tabular, has_2d, has_3d per ogni sample
    """
    model.eval()

    all_labels   = []
    all_preds    = []
    all_probs    = []
    all_mri_ids  = []
    all_has      = {"tabular": [], "2d": [], "3d": []}

    for batch in loader:
        batch  = _move_batch(batch, device)
        labels = batch["label"]

        logits = model(batch)
        probs  = torch.softmax(logits, dim=-1)
        preds  = probs.argmax(dim=-1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_mri_ids.extend(batch["mri_id"])
        all_has["tabular"].extend(batch["has_tabular"].cpu().numpy())
        all_has["2d"].extend(batch["has_2d"].cpu().numpy())
        all_has["3d"].extend(batch["has_3d"].cpu().numpy())

    return {
        "labels":   np.array(all_labels),
        "preds":    np.array(all_preds),
        "probs":    np.array(all_probs),
        "mri_ids":  all_mri_ids,
        "has_flags": {k: np.array(v) for k, v in all_has.items()},
    }


# ══════════════════════════════════════════════════════════════════════════════
# METRICHE
# ══════════════════════════════════════════════════════════════════════════════

def compute_full_metrics(results: dict) -> dict:
    """
    Calcola tutte le metriche di valutazione.
    """
    labels = results["labels"]
    preds  = results["preds"]
    probs  = results["probs"]

    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, average="macro", zero_division=0)
    f1_per_class = f1_score(labels, preds, average=None, zero_division=0).tolist()

    try:
        auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
        auc_per_class = roc_auc_score(
            labels, probs, multi_class="ovr", average=None
        ).tolist()
    except ValueError:
        auc = 0.0
        auc_per_class = [0.0] * cfg.N_CLASSES

    ece = _ece(probs, labels)

    report = classification_report(
        labels, preds,
        target_names=cfg.CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(labels, preds).tolist()

    # metriche per modalità disponibile
    modal_metrics = {}
    for mod, flags in results["has_flags"].items():
        mask = flags.astype(bool)
        if mask.sum() == 0:
            continue
        m_acc = accuracy_score(labels[mask], preds[mask])
        m_f1  = f1_score(labels[mask], preds[mask], average="macro", zero_division=0)
        modal_metrics[f"only_{mod}_available"] = {
            "n": int(mask.sum()), "acc": round(m_acc, 4), "f1": round(m_f1, 4)
        }

    return {
        "accuracy":       round(acc, 4),
        "f1_macro":       round(f1, 4),
        "f1_per_class":   {cfg.CLASS_NAMES[i]: round(v, 4) for i, v in enumerate(f1_per_class)},
        "auc_macro":      round(auc, 4),
        "auc_per_class":  {cfg.CLASS_NAMES[i]: round(v, 4) for i, v in enumerate(auc_per_class)},
        "ece":            round(ece, 4),
        "n_samples":      int(len(labels)),
        "confusion_matrix": cm,
        "classification_report": report,
        "modal_metrics":  modal_metrics,
    }


# ══════════════════════════════════════════════════════════════════════════════
# PLOT
# ══════════════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(
    cm:         list,
    model_name: str,
    split:      str,
    save_path:  str,
):
    """Plotta e salva la confusion matrix normalizzata."""
    cm_arr  = np.array(cm)
    cm_norm = cm_arr.astype(float) / cm_arr.sum(axis=1, keepdims=True).clip(min=1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # assoluta
    sns.heatmap(
        cm_arr, annot=True, fmt="d", cmap="Blues",
        xticklabels=cfg.CLASS_NAMES, yticklabels=cfg.CLASS_NAMES,
        ax=axes[0],
    )
    axes[0].set_title(f"Confusion Matrix (count)\n{model_name} – {split}")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    # normalizzata
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=cfg.CLASS_NAMES, yticklabels=cfg.CLASS_NAMES,
        ax=axes[1], vmin=0, vmax=1,
    )
    axes[1].set_title(f"Confusion Matrix (normalized)\n{model_name} – {split}")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix → {save_path}")


def plot_calibration(
    probs:      np.ndarray,
    labels:     np.ndarray,
    model_name: str,
    split:      str,
    save_path:  str,
    n_bins:     int = 10,
):
    """Reliability diagram per la calibrazione del modello."""
    confidences = probs.max(axis=1)
    preds       = probs.argmax(axis=1)
    correct     = (preds == labels).astype(float)

    bins     = np.linspace(0, 1, n_bins + 1)
    bin_acc  = []
    bin_conf = []
    bin_size = []

    for i in range(n_bins):
        mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
        if mask.sum() == 0:
            bin_acc.append(0)
            bin_conf.append((bins[i] + bins[i + 1]) / 2)
            bin_size.append(0)
        else:
            bin_acc.append(correct[mask].mean())
            bin_conf.append(confidences[mask].mean())
            bin_size.append(mask.sum())

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.bar(
        [(bins[i] + bins[i+1]) / 2 for i in range(n_bins)],
        bin_acc, width=0.08, alpha=0.6, label="Accuracy per bin",
    )
    ax.plot(bin_conf, bin_acc, "ro-", label="Model")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Calibration – {model_name} ({split})")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Calibration plot → {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATE
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(
    model_cfg:  str  = "4",
    split:      str  = "test",   # "val" | "test"
    batch_size: int  = cfg.BATCH_SIZE,
    seed:       int  = cfg.SEED,
):
    """
    Carica il checkpoint migliore e valuta sul split richiesto.

    Parametri
    ---------
    model_cfg : chiave MODEL_CONFIGS
    split     : "val" o "test"
    """
    mcfg       = cfg.MODEL_CONFIGS[model_cfg]
    model_name = mcfg["name"]
    ckpt_path  = mcfg["checkpoint"]

    device = torch.device(
        cfg.DEVICE if (cfg.DEVICE != "cuda" or torch.cuda.is_available()) else "cpu"
    )

    print("=" * 56)
    print(f"  Evaluation: {model_name}  |  split: {split}")
    print(f"  Device    : {device}")
    print(f"  Checkpoint: {ckpt_path}")
    print("=" * 56)

    # ── carica modello ────────────────────────────────────────────────────
    if not os.path.exists(ckpt_path):
        print(f"\n  [ERR] Checkpoint non trovato: {ckpt_path}")
        print("  Esegui prima il training con python training/train.py --all")
        return None

    model = build_model(model_cfg=model_cfg, pretrained_2d=False, device=str(device))
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    print(f"\n  Checkpoint caricato OK")

    # ── dataloader ────────────────────────────────────────────────────────
    loaders = build_dataloaders(
        model_cfg=model_cfg,
        batch_size=batch_size,
        seed=seed,
    )
    loader = loaders[split]
    print(f"  Campioni nel split '{split}': {len(loader.dataset)}")

    # ── inference ─────────────────────────────────────────────────────────
    results = run_inference(model, loader, device)

    # ── metriche ──────────────────────────────────────────────────────────
    metrics = compute_full_metrics(results)

    print(f"\n  {'Metrica':<20} {'Valore':>8}")
    print("  " + "-" * 30)
    print(f"  {'Accuracy':<20} {metrics['accuracy']:>8.4f}")
    print(f"  {'F1 macro':<20} {metrics['f1_macro']:>8.4f}")
    print(f"  {'AUC macro':<20} {metrics['auc_macro']:>8.4f}")
    print(f"  {'ECE':<20} {metrics['ece']:>8.4f}")
    print()
    print("  Per classe:")
    for cls in cfg.CLASS_NAMES:
        print(f"    {cls:<6}  F1={metrics['f1_per_class'][cls]:.4f}"
              f"  AUC={metrics['auc_per_class'][cls]:.4f}")

    if metrics["modal_metrics"]:
        print("\n  Per modalità disponibile:")
        for k, v in metrics["modal_metrics"].items():
            print(f"    {k:<30} n={v['n']:>3}  acc={v['acc']:.4f}  f1={v['f1']:.4f}")

    # ── salva report JSON ─────────────────────────────────────────────────
    report_path = os.path.join(EVAL_DIR, f"{model_name}_{split}_report.json")
    with open(report_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Report JSON → {report_path}")

    # ── plot confusion matrix ─────────────────────────────────────────────
    cm_path = os.path.join(EVAL_DIR, f"{model_name}_{split}_confusion.png")
    plot_confusion_matrix(metrics["confusion_matrix"], model_name, split, cm_path)

    # ── plot calibration ──────────────────────────────────────────────────
    cal_path = os.path.join(EVAL_DIR, f"{model_name}_{split}_calibration.png")
    plot_calibration(results["probs"], results["labels"], model_name, split, cal_path)

    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# CONFRONTO CONFIGURAZIONI
# ══════════════════════════════════════════════════════════════════════════════

def compare_configs(split: str = "test"):
    """
    Valuta tutte le configurazioni disponibili e produce una tabella comparativa.
    Utile per ablation study: confronto Full vs Tab+3D vs Tab+2D vs 2D+3D.
    """
    print("\n" + "=" * 56)
    print("  Ablation study – confronto configurazioni")
    print("=" * 56)

    results_all = {}
    for key, mcfg in cfg.MODEL_CONFIGS.items():
        if not os.path.exists(mcfg["checkpoint"]):
            print(f"  [{key}] {mcfg['name']:<20} → checkpoint mancante, skip")
            continue
        print(f"\n  Valuto [{key}] {mcfg['name']}...")
        m = evaluate(model_cfg=key, split=split)
        if m:
            results_all[mcfg["name"]] = m

    if not results_all:
        print("  Nessun checkpoint trovato.")
        return

    print("\n" + "=" * 70)
    print(f"  {'Config':<22} {'Acc':>6} {'F1':>6} {'AUC':>6} {'ECE':>6}")
    print("  " + "-" * 50)
    for name, m in results_all.items():
        print(f"  {name:<22} {m['accuracy']:>6.4f} {m['f1_macro']:>6.4f}"
              f" {m['auc_macro']:>6.4f} {m['ece']:>6.4f}")

    # salva tabella comparativa
    compare_path = os.path.join(EVAL_DIR, f"ablation_{split}.json")
    with open(compare_path, "w") as f:
        json.dump(results_all, f, indent=2)
    print(f"\n  Ablation table → {compare_path}")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_cfg",  type=str, default="4")
    parser.add_argument("--split",      type=str, default="test",
                        choices=["val", "test"])
    parser.add_argument("--batch_size", type=int, default=cfg.BATCH_SIZE)
    parser.add_argument("--compare",    action="store_true",
                        help="Confronta tutte le configurazioni (ablation study)")
    args = parser.parse_args()

    if args.compare:
        compare_configs(split=args.split)
    else:
        evaluate(
            model_cfg=args.model_cfg,
            split=args.split,
            batch_size=args.batch_size,
        )

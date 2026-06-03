"""
training/train.py
=================
Training pipeline completa per AlzheimerMultimodalNet:

  Fase 1 — Pre-training encoder imaging (enc_2d + enc_3d)
  Fase 2 — Pre-training encoder tabellare (enc_tab)
  Fase 3 — Fine-tuning end-to-end con differential LR

  K-Fold cross validation (--kfold)
  Grid Search iperparametri (--grid_search)

Uso:
  python training/train.py --phase 1
  python training/train.py --phase 2
  python training/train.py --phase 3 --model_cfg 4
  python training/train.py --all
  python training/train.py --kfold --n_splits 5
  python training/train.py --kfold --n_splits 5 --epochs 30 --patience 5
  python training/train.py --grid_search --n_splits 3 --fast
"""

import os
import sys
import time
import argparse
import json
import itertools
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import config as cfg
from dataset.dataset import OASISDataset, build_dataloaders
from models.model    import build_model, AlzheimerMultimodalNet

# ── path ──────────────────────────────────────────────────────────────────────
CKPT_PRETRAIN_IMAGING = os.path.join(cfg.CHECKPOINT_DIR, "pretrain_imaging.pt")
CKPT_PRETRAIN_TABULAR = os.path.join(cfg.CHECKPOINT_DIR, "pretrain_tabular.pt")
LOG_DIR = os.path.join(ROOT, "outputs", "logs")
KFOLD_DIR = os.path.join(ROOT, "outputs", "kfold")
PROCESSED_DIR= os.path.join(ROOT, "data", "processed")
TABULAR_CSV = os.path.join(PROCESSED_DIR, "tabular.csv")
VOLUMETRIC_CSV = os.path.join(PROCESSED_DIR, "volumetric.csv")
MANIFEST_CSV = os.path.join(PROCESSED_DIR, "manifest.csv")

for d in [LOG_DIR, KFOLD_DIR]:
    os.makedirs(d, exist_ok=True)

# ── grid search ───────────────────────────────────────────────────────────────
GRID = {
    "embed_dim":   [128, 256],
    "num_heads":   [4, 8],
    "dropout":     [0.2, 0.3],
    "lr_encoders": [1e-5, 5e-6],
    "lr_fusion":   [1e-4, 5e-5],
    "batch_size":  [2, 4],
}
GRID_FAST = {
    "embed_dim":   [256],
    "num_heads":   [8],
    "dropout":     [0.2, 0.3],
    "lr_encoders": [1e-5],
    "lr_fusion":   [1e-4, 5e-5],
    "batch_size":  [2],
}


# ══════════════════════════════════════════════════════════════════════════════
# UTILITÀ CONDIVISE
# ══════════════════════════════════════════════════════════════════════════════

def _set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()}


def _compute_metrics(all_labels, all_preds, all_probs) -> dict:
    labels = np.array(all_labels)
    preds  = np.array(all_preds)
    probs  = np.array(all_probs)
    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, average="macro", zero_division=0)
    try:
        auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
    except ValueError:
        auc = 0.0
    return {"acc": acc, "f1": f1, "auc": auc}


def _print_header():
    print(f"  {'Epoch':>5} {'LR':>8} | {'Loss':>6} {'Acc':>6} {'F1':>6} {'AUC':>6} |"
          f" {'vLoss':>6} {'vAcc':>6} {'vF1':>6} {'vAUC':>6} | Best", flush=True)
    print("  " + "-" * 88, flush=True)


def _print_epoch(epoch, lr, t_m, v_m, is_best):
    print(
        f"  {epoch:>5} {lr:>8.2e} |"
        f" {t_m['loss']:>6.3f} {t_m['acc']:>6.3f} {t_m['f1']:>6.3f} {t_m['auc']:>6.3f} |"
        f" {v_m['loss']:>6.3f} {v_m['acc']:>6.3f} {v_m['f1']:>6.3f} {v_m['auc']:>6.3f} |"
        f" {'★' if is_best else ' '}",
        flush=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# SCHEDULER E EARLY STOPPING
# ══════════════════════════════════════════════════════════════════════════════

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self.current_epoch = 0

    def step(self):
        self.current_epoch += 1
        e = self.current_epoch
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            if e <= self.warmup_epochs:
                pg["lr"] = base_lr * e / max(self.warmup_epochs, 1)
            else:
                p = (e - self.warmup_epochs) / max(
                    self.total_epochs - self.warmup_epochs, 1)
                pg["lr"] = self.min_lr + 0.5 * (base_lr - self.min_lr) * (
                    1 + np.cos(np.pi * p))

    def get_lr(self):
        return self.optimizer.param_groups[0]["lr"]


class EarlyStopping:
    def __init__(self, patience, checkpoint_path):
        self.patience = patience
        self.checkpoint_path = checkpoint_path
        self.best_score = -float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, score, model) -> bool:
        if score > self.best_score:
            self.best_score = score
            self.counter = 0
            torch.save(model.state_dict(), self.checkpoint_path)
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False


# ══════════════════════════════════════════════════════════════════════════════
# TRAIN / EVAL STEP
# ══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, criterion, device, clip_grad=1.0):
    model.train()
    total_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []
    for batch in loader:
        batch = _move_batch(batch, device)
        labels = batch["label"]
        optimizer.zero_grad()
        logits = model(batch)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        probs = torch.softmax(logits, dim=-1)
        total_loss += loss.item()
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(probs.argmax(dim=-1).cpu().numpy())
        all_probs.extend(probs.detach().cpu().numpy())
    m = _compute_metrics(all_labels, all_preds, all_probs)
    m["loss"] = total_loss / max(len(loader), 1)
    return m


@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []
    for batch in loader:
        batch = _move_batch(batch, device)
        labels = batch["label"]
        logits = model(batch)
        loss = criterion(logits, labels)
        probs = torch.softmax(logits, dim=-1)
        total_loss += loss.item()
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(probs.argmax(dim=-1).cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
    m = _compute_metrics(all_labels, all_preds, all_probs)
    m["loss"] = total_loss / max(len(loader), 1)
    return m


def _run_loop(model, train_loader, val_loader, optimizer, criterion,
              scheduler, stopper, device, epochs, log_path, phase_name,
              verbose=True):
    history = {"train": [], "val": []}
    best_epoch = 0
    t_start = time.time()

    if verbose:
        print(f"\n  Checkpoint → {stopper.checkpoint_path}", flush=True)
        print(f"  Log       → {log_path}\n", flush=True)
        _print_header()

    for epoch in range(1, epochs + 1):
        scheduler.step()
        t_m = train_one_epoch(model, train_loader, optimizer, criterion, device)
        v_m = eval_one_epoch(model, val_loader, criterion, device)

        is_best = stopper.step(v_m["f1"], model)
        if is_best:
            best_epoch = epoch

        history["train"].append(t_m)
        history["val"].append(v_m)
        with open(log_path, "w") as f:
            json.dump(history, f, indent=2)

        if verbose:
            _print_epoch(epoch, scheduler.get_lr(), t_m, v_m, is_best)

        if stopper.should_stop:
            if verbose:
                print(f"\n  [EARLY STOP] Patience esaurita dopo {epoch} epoch.",
                      flush=True)
            break

    elapsed = time.time() - t_start
    if verbose:
        print(f"\n  [{phase_name}] Completato in {elapsed/60:.1f} min", flush=True)
        print(f"  Best val F1 : {stopper.best_score:.4f} @ epoch {best_epoch}",
              flush=True)
    return history, stopper.best_score


# ══════════════════════════════════════════════════════════════════════════════
# FASE 1 — PRE-TRAINING ENCODER IMAGING
# ══════════════════════════════════════════════════════════════════════════════

class ImagingOnlyModel(nn.Module):
    def __init__(self, full_model: AlzheimerMultimodalNet, embed_dim=cfg.EMB_DIM):
        super().__init__()
        self.enc_2d = full_model.enc_2d
        self.enc_3d = full_model.enc_3d
        self.head   = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(cfg.DROPOUT),
            nn.Linear(embed_dim, cfg.N_CLASSES),
        )

    def forward(self, batch):
        has_2d = batch["has_2d"].float().unsqueeze(1)
        has_3d = batch["has_3d"].float().unsqueeze(1)
        f2 = self.enc_2d(batch["slice_2d"]).mean(dim=1)  * has_2d
        f3 = self.enc_3d(batch["volume_3d"]).mean(dim=1) * has_3d
        return self.head(torch.cat([f2, f3], dim=-1))


def pretrain_imaging(epochs=cfg.N_EPOCHS, batch_size=cfg.BATCH_SIZE,
                     lr=cfg.LR, patience=cfg.PATIENCE, seed=cfg.SEED,
                     pretrained_2d=True):
    print("\n" + "=" * 56, flush=True)
    print("  FASE 1 — Pre-training encoder imaging (2D + 3D)", flush=True)
    print("=" * 56, flush=True)
    _set_seed(seed)
    device = torch.device(
        cfg.DEVICE if (cfg.DEVICE != "cuda" or torch.cuda.is_available()) else "cpu")
    loaders = build_dataloaders(model_cfg="1", batch_size=batch_size, seed=seed)
    class_weights = loaders["class_weights"].to(device)
    full_model= build_model(model_cfg="1", pretrained_2d=pretrained_2d,
                                device=str(device))
    model = ImagingOnlyModel(full_model, cfg.EMB_DIM).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = WarmupCosineScheduler(optimizer, 5, epochs)
    stopper = EarlyStopping(patience, CKPT_PRETRAIN_IMAGING)
    log_path = os.path.join(LOG_DIR, "pretrain_imaging.json")
    _, best_f1 = _run_loop(model, loaders["train"], loaders["val"],
                               optimizer, criterion, scheduler, stopper,
                               device, epochs, log_path, "Fase 1 – Imaging")
    print(f"\n  Pesi encoder imaging salvati → {CKPT_PRETRAIN_IMAGING}", flush=True)
    return best_f1


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2 — PRE-TRAINING ENCODER TABELLARE
# ══════════════════════════════════════════════════════════════════════════════

class TabularOnlyModel(nn.Module):
    def __init__(self, full_model: AlzheimerMultimodalNet, embed_dim=cfg.EMB_DIM):
        super().__init__()
        self.enc_tab = full_model.enc_tab
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(cfg.DROPOUT),
            nn.Linear(embed_dim // 2, cfg.N_CLASSES),
        )

    def forward(self, batch):
        feat = self.enc_tab(batch["tabular"], batch["volumetric"]).mean(dim=1)
        return self.head(feat)


def pretrain_tabular(epochs=cfg.N_EPOCHS, batch_size=cfg.BATCH_SIZE,
                     lr=cfg.LR, patience=cfg.PATIENCE, seed=cfg.SEED):
    print("\n" + "=" * 56, flush=True)
    print("  FASE 2 — Pre-training encoder tabellare", flush=True)
    print("=" * 56, flush=True)
    _set_seed(seed)
    device = torch.device(
        cfg.DEVICE if (cfg.DEVICE != "cuda" or torch.cuda.is_available()) else "cpu")
    loaders = build_dataloaders(model_cfg="3", batch_size=batch_size, seed=seed)
    class_weights = loaders["class_weights"].to(device)
    full_model = build_model(model_cfg="3", pretrained_2d=False, device=str(device))
    model = TabularOnlyModel(full_model, cfg.EMB_DIM).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = WarmupCosineScheduler(optimizer, 5, epochs)
    stopper = EarlyStopping(patience, CKPT_PRETRAIN_TABULAR)
    log_path = os.path.join(LOG_DIR, "pretrain_tabular.json")
    _, best_f1 = _run_loop(model, loaders["train"], loaders["val"],
                               optimizer, criterion, scheduler, stopper,
                               device, epochs, log_path, "Fase 2 – Tabellare")
    print(f"\n  Pesi encoder tabellare salvati → {CKPT_PRETRAIN_TABULAR}", flush=True)
    return best_f1


# ══════════════════════════════════════════════════════════════════════════════
# FASE 3 — FINE-TUNING END-TO-END
# ══════════════════════════════════════════════════════════════════════════════

def _load_pretrained_weights(model):
    current = model.state_dict()

    if os.path.exists(CKPT_PRETRAIN_IMAGING):
        ckpt = torch.load(CKPT_PRETRAIN_IMAGING, map_location="cpu")
        w = {}
        for k, v in ckpt.items():
            if not (k.startswith("enc_2d.") or k.startswith("enc_3d.")):
                continue
            if k not in current or v.shape != current[k].shape:
                continue
            w[k] = v
        model.load_state_dict(w, strict=False)
        print(f"  [FASE 3] Pesi imaging caricati : {len(w)} tensori", flush=True)
    else:
        print("  [WARN] Checkpoint imaging non trovato — skip", flush=True)

    if os.path.exists(CKPT_PRETRAIN_TABULAR):
        ckpt = torch.load(CKPT_PRETRAIN_TABULAR, map_location="cpu")
        w = {}
        for k, v in ckpt.items():
            if not k.startswith("enc_tab."):
                continue
            if k not in current or v.shape != current[k].shape:
                continue
            w[k] = v
        model.load_state_dict(w, strict=False)
        print(f"  [FASE 3] Pesi tabellare caricati: {len(w)} tensori", flush=True)
    else:
        print("  [WARN] Checkpoint tabellare non trovato — skip", flush=True)

    return model


def finetune(model_cfg="4", epochs=cfg.N_EPOCHS, batch_size=cfg.BATCH_SIZE,
             lr_encoders=1e-5, lr_fusion=1e-4, weight_decay=cfg.WEIGHT_DECAY,
             patience=cfg.PATIENCE, seed=cfg.SEED, pretrained_2d=True,
             embed_dim=None, num_heads=None, dropout=None, verbose=True):
    print("\n" + "=" * 56, flush=True)
    print("  FASE 3 — Fine-tuning end-to-end (differential LR)", flush=True)
    print(f"  LR encoders : {lr_encoders}  |  LR fusion+head: {lr_fusion}", flush=True)
    print("=" * 56, flush=True)
    _set_seed(seed)
    device = torch.device(
        cfg.DEVICE if (cfg.DEVICE != "cuda" or torch.cuda.is_available()) else "cpu")
    mcfg    = cfg.MODEL_CONFIGS[model_cfg]
    loaders = build_dataloaders(model_cfg=model_cfg, batch_size=batch_size, seed=seed)
    class_weights = loaders["class_weights"].to(device)

    model = build_model(model_cfg=model_cfg, pretrained_2d=pretrained_2d,
                        device=str(device), embed_dim=embed_dim,
                        num_heads=num_heads, dropout=dropout)
    model = _load_pretrained_weights(model)

    enc_params = (list(model.enc_tab.parameters()) +
                    list(model.enc_2d.parameters()) +
                    list(model.enc_3d.parameters()))
    other_params = (list(model.router.parameters()) +
                    list(model.fusion.parameters()) +
                    list(model.head.parameters()))

    optimizer = AdamW([
        {"params": enc_params,   "lr": lr_encoders},
        {"params": other_params, "lr": lr_fusion},
    ], weight_decay=weight_decay)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    scheduler = WarmupCosineScheduler(optimizer, 5, epochs)
    stopper = EarlyStopping(patience, mcfg["checkpoint"])
    log_path = os.path.join(LOG_DIR, f"finetune_{mcfg['name']}.json")

    if verbose:
        print(f"\n  Encoder  : {sum(p.numel() for p in enc_params)/1e6:.2f}M"
              f"  @ LR={lr_encoders}", flush=True)
        print(f"  Fusion   : {sum(p.numel() for p in other_params)/1e6:.2f}M"
              f"  @ LR={lr_fusion}", flush=True)

    return _run_loop(model, loaders["train"], loaders["val"],
                     optimizer, criterion, scheduler, stopper,
                     device, epochs, log_path,
                     "Fase 3 – Fine-tuning", verbose=verbose)


# ══════════════════════════════════════════════════════════════════════════════
# K-FOLD
# ══════════════════════════════════════════════════════════════════════════════

def _train_fold(fold_idx, man_train, man_val, df_tab, df_vol,
                hparams, device, epochs, patience, tag):
    """Addestra un singolo fold con output verbose in tempo reale."""
    _set_seed(cfg.SEED + fold_idx)

    embed_dim = hparams.get("embed_dim",   cfg.EMB_DIM)
    num_heads = hparams.get("num_heads",   cfg.N_HEADS)
    dropout = hparams.get("dropout",     cfg.DROPOUT)
    lr_encoders = hparams.get("lr_encoders", 1e-5)
    lr_fusion = hparams.get("lr_fusion",   1e-4)
    batch_size = hparams.get("batch_size",  cfg.BATCH_SIZE)

    train_labels = man_train["label"].values
    class_counts = np.bincount(train_labels, minlength=cfg.N_CLASSES).astype(float)
    class_w = 1.0 / (class_counts + 1e-8)
    class_w = class_w / class_w.sum() * cfg.N_CLASSES
    class_w_t = torch.tensor(class_w, dtype=torch.float32).to(device)

    ds_train = OASISDataset(man_train, df_tab, df_vol, split="train", model_cfg="4")
    ds_val = OASISDataset(man_val,   df_tab, df_vol, split="val",   model_cfg="4")

    sampler = WeightedRandomSampler(
        torch.from_numpy(class_w[train_labels]).float(),
        num_samples=len(train_labels), replacement=True,
    )
    loader_tr = DataLoader(ds_train, batch_size=batch_size,
                           sampler=sampler, drop_last=True, num_workers=0)
    loader_va = DataLoader(ds_val,   batch_size=batch_size,
                           shuffle=False, num_workers=0)

    criterion = nn.CrossEntropyLoss(weight=class_w_t, label_smoothing=0.05)

    # ── pretrain tabellare ────────────────────────────────────────────────
    print(f"    [Fold {fold_idx+1}] Pretrain tabellare...", flush=True)
    full = build_model("4", pretrained_2d=False, device=str(device),
                        embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
    tab_m = TabularOnlyModel(full, embed_dim).to(device)
    opt_p = AdamW(tab_m.parameters(), lr=lr_fusion, weight_decay=cfg.WEIGHT_DECAY)
    sch_p = WarmupCosineScheduler(opt_p, 3, min(25, epochs))
    best_p = -1.0; pat_p = 0
    pre_ckpt = os.path.join(KFOLD_DIR, f"{tag}_fold{fold_idx}_pre.pt")

    for ep in range(min(25, epochs)):
        sch_p.step()
        train_one_epoch(tab_m, loader_tr, opt_p, criterion, device)
        vm = eval_one_epoch(tab_m, loader_va, criterion, device)
        if vm["f1"] > best_p:
            best_p = vm["f1"]; pat_p = 0
            torch.save(tab_m.state_dict(), pre_ckpt)
        else:
            pat_p += 1
            if pat_p >= 8:
                break
        if (ep + 1) % 5 == 0:
            print(f"      pretrain ep {ep+1:2d} | val F1={vm['f1']:.3f}", flush=True)

    print(f"    [Fold {fold_idx+1}] Pretrain tab → best F1: {best_p:.4f}", flush=True)

    # trasferisci pesi enc_tab
    state = torch.load(pre_ckpt, map_location=device)
    enc_w = {k: v for k, v in state.items() if k.startswith("enc_tab.")}
    full.load_state_dict(enc_w, strict=False)
    if os.path.exists(pre_ckpt):
        os.remove(pre_ckpt)

    # ── fine-tuning ───────────────────────────────────────────────────────
    print(f"    [Fold {fold_idx+1}] Fine-tuning...", flush=True)
    enc_p = (list(full.enc_tab.parameters()) +
               list(full.enc_2d.parameters())  +
               list(full.enc_3d.parameters()))
    other_p = (list(full.router.parameters()) +
               list(full.fusion.parameters())  +
               list(full.head.parameters()))
    opt_ft  = AdamW([{"params": enc_p,   "lr": lr_encoders},
                     {"params": other_p, "lr": lr_fusion}],
                    weight_decay=cfg.WEIGHT_DECAY)
    sch_ft  = WarmupCosineScheduler(opt_ft, 5, epochs)
    best_f1 = -1.0; pat_ft = 0; best_m = {}
    fold_ckpt = os.path.join(KFOLD_DIR, f"{tag}_fold{fold_idx}_best.pt")

    for ep in range(epochs):
        sch_ft.step()
        t_m = train_one_epoch(full, loader_tr, opt_ft, criterion, device)
        v_m = eval_one_epoch(full, loader_va, criterion, device)

        if v_m["f1"] > best_f1:
            best_f1 = v_m["f1"]; best_m = v_m; pat_ft = 0
            torch.save(full.state_dict(), fold_ckpt)
            print(f"      ep {ep+1:3d} | train F1={t_m['f1']:.3f} | "
                  f"val F1={v_m['f1']:.3f} AUC={v_m['auc']:.3f} ★", flush=True)
        else:
            pat_ft += 1
            if (ep + 1) % 10 == 0:
                print(f"      ep {ep+1:3d} | train F1={t_m['f1']:.3f} | "
                      f"val F1={v_m['f1']:.3f} AUC={v_m['auc']:.3f} "
                      f"(patience {pat_ft}/{patience})", flush=True)
            if pat_ft >= patience:
                print(f"      [early stop] fold {fold_idx+1}", flush=True)
                break

    best_m["fold"] = fold_idx
    print(f"    [Fold {fold_idx+1}] Best val F1: {best_f1:.4f}", flush=True)
    return best_m


def run_kfold(n_splits=cfg.N_SPLITS, hparams=None, epochs=cfg.N_EPOCHS,
              patience=cfg.PATIENCE, tag="default") -> dict:
    if hparams is None:
        hparams = {"embed_dim": cfg.EMB_DIM, "num_heads": cfg.N_HEADS,
                   "dropout": cfg.DROPOUT, "lr_encoders": 1e-5,
                   "lr_fusion": 1e-4, "batch_size": cfg.BATCH_SIZE}

    device = torch.device(
        cfg.DEVICE if (cfg.DEVICE != "cuda" or torch.cuda.is_available()) else "cpu")

    manifest = pd.read_csv(MANIFEST_CSV)
    df_tabular = pd.read_csv(TABULAR_CSV)    if os.path.exists(TABULAR_CSV)    else None
    df_volumetric = pd.read_csv(VOLUMETRIC_CSV) if os.path.exists(VOLUMETRIC_CSV) else None

    subj_labels = (manifest.groupby("subject_id")["label"]
                   .agg(lambda x: x.value_counts().idxmax()).reset_index())
    subj_ids = subj_labels["subject_id"].values
    labels = subj_labels["label"].values
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg.SEED)

    print(f"\n{'='*60}", flush=True)
    print(f"  K-Fold CV  n_splits={n_splits}  tag={tag}", flush=True)
    print(f"  hparams: {hparams}", flush=True)
    print(f"{'='*60}", flush=True)

    fold_results = []
    t_start = time.time()

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(subj_ids, labels)):
        tr_subjs = set(subj_ids[tr_idx])
        va_subjs = set(subj_ids[va_idx])
        man_tr = manifest[manifest["subject_id"].isin(tr_subjs)]
        man_va = manifest[manifest["subject_id"].isin(va_subjs)]
        dist = " | ".join(
            f"{cfg.CLASS_NAMES[i]}={man_tr['label'].value_counts().get(i,0)}"
            for i in range(cfg.N_CLASSES))
        print(f"\n  Fold {fold_idx+1}/{n_splits} | "
              f"train={len(man_tr)} val={len(man_va)} | {dist}", flush=True)

        m = _train_fold(fold_idx, man_tr, man_va, df_tabular, df_volumetric,
                        hparams, device, epochs, patience, tag)
        fold_results.append(m)
        print(f"  → F1={m['f1']:.4f}  AUC={m['auc']:.4f}  Acc={m['acc']:.4f}",
              flush=True)

    f1s  = [r["f1"]  for r in fold_results]
    aucs = [r["auc"] for r in fold_results]
    accs = [r["acc"] for r in fold_results]

    summary = {
        "tag": tag, "hparams": hparams, "n_splits": n_splits,
        "fold_results": fold_results,
        "f1_mean":  round(float(np.mean(f1s)),  4),
        "f1_std":   round(float(np.std(f1s)),   4),
        "auc_mean": round(float(np.mean(aucs)), 4),
        "auc_std":  round(float(np.std(aucs)),  4),
        "acc_mean": round(float(np.mean(accs)), 4),
        "acc_std":  round(float(np.std(accs)),  4),
        "elapsed_min": round((time.time() - t_start) / 60, 1),
    }

    print(f"\n{'─'*50}", flush=True)
    print(f"  K-Fold [{tag}]", flush=True)
    print(f"  F1 : {summary['f1_mean']:.4f} ± {summary['f1_std']:.4f}", flush=True)
    print(f"  AUC: {summary['auc_mean']:.4f} ± {summary['auc_std']:.4f}", flush=True)
    print(f"  Acc: {summary['acc_mean']:.4f} ± {summary['acc_std']:.4f}", flush=True)
    print(f"  Tempo: {summary['elapsed_min']} min", flush=True)

    out = os.path.join(KFOLD_DIR, f"{tag}_kfold.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Salvato → {out}", flush=True)
    return summary


# ══════════════════════════════════════════════════════════════════════════════
# GRID SEARCH
# ══════════════════════════════════════════════════════════════════════════════

def run_grid_search(n_splits=3, epochs=cfg.N_EPOCHS, patience=cfg.PATIENCE,
                    fast=False) -> dict:
    grid = GRID_FAST if fast else GRID
    keys = list(grid.keys())
    combos = list(itertools.product(*grid.values()))
    valid = [c for c in combos
              if dict(zip(keys, c))["embed_dim"] % dict(zip(keys, c))["num_heads"] == 0]

    print(f"\n{'='*60}", flush=True)
    print(f"  Grid Search  combo={len(valid)}  splits={n_splits}"
          f"  {'FAST' if fast else 'FULL'}", flush=True)
    print(f"{'='*60}", flush=True)

    all_results = []; best_f1 = -1.0; best_params = None

    for i, combo in enumerate(valid):
        hp  = dict(zip(keys, combo))
        tag = f"gs_{i:03d}"
        print(f"\n[{i+1}/{len(valid)}] {hp}", flush=True)
        try:
            s = run_kfold(n_splits=n_splits, hparams=hp,
                          epochs=epochs, patience=patience, tag=tag)
            all_results.append(s)
            if s["f1_mean"] > best_f1:
                best_f1 = s["f1_mean"]; best_params = hp
                print(f"  ★ Nuovo best F1: {best_f1:.4f}", flush=True)
        except Exception as e:
            print(f"  [ERR] {e}", flush=True)

    all_results.sort(key=lambda x: x["f1_mean"], reverse=True)
    print(f"\n{'='*60}\n  Top 5\n{'─'*60}", flush=True)
    for i, r in enumerate(all_results[:5]):
        print(f"  {i+1}. F1={r['f1_mean']:.4f}±{r['f1_std']:.4f}"
              f"  AUC={r['auc_mean']:.4f} | {r['hparams']}", flush=True)

    report = {"best_params": best_params, "best_f1": best_f1,
              "all_results": all_results}
    rp = os.path.join(KFOLD_DIR, "grid_search_report.json")
    with open(rp, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report → {rp}", flush=True)
    return report


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase",         type=int,   default=None)
    parser.add_argument("--all",           action="store_true")
    parser.add_argument("--kfold",         action="store_true")
    parser.add_argument("--grid_search",   action="store_true")
    parser.add_argument("--fast",          action="store_true")
    parser.add_argument("--model_cfg",     type=str,   default="4")
    parser.add_argument("--epochs",        type=int,   default=cfg.N_EPOCHS)
    parser.add_argument("--batch_size",    type=int,   default=cfg.BATCH_SIZE)
    parser.add_argument("--lr",            type=float, default=cfg.LR)
    parser.add_argument("--lr_encoders",   type=float, default=1e-5)
    parser.add_argument("--lr_fusion",     type=float, default=1e-4)
    parser.add_argument("--patience",      type=int,   default=cfg.PATIENCE)
    parser.add_argument("--n_splits",      type=int,   default=cfg.N_SPLITS)
    parser.add_argument("--no_pretrained", action="store_true")
    args = parser.parse_args()

    pretrained_2d = not args.no_pretrained

    if args.grid_search:
        run_grid_search(n_splits=args.n_splits, epochs=args.epochs,
                        patience=args.patience, fast=args.fast)
    elif args.kfold:
        run_kfold(n_splits=args.n_splits, epochs=args.epochs,
                  patience=args.patience)
    else:
        if args.all or args.phase == 1:
            pretrain_imaging(epochs=args.epochs, batch_size=args.batch_size,
                             lr=args.lr, patience=args.patience,
                             pretrained_2d=pretrained_2d)
        if args.all or args.phase == 2:
            pretrain_tabular(epochs=args.epochs, batch_size=args.batch_size,
                             lr=args.lr, patience=args.patience)
        if args.all or args.phase == 3:
            finetune(model_cfg=args.model_cfg, epochs=args.epochs,
                     batch_size=args.batch_size, lr_encoders=args.lr_encoders,
                     lr_fusion=args.lr_fusion, patience=args.patience,
                     pretrained_2d=pretrained_2d)
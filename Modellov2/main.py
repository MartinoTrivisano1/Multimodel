"""
main.py
=======
Entry point interattivo per AlzheimerMultimodalNet.

Orchestrazione completa della pipeline:
  1.  Preprocessing
  2.  Training completo (Fase 2 + 3)
  3.  Solo Fase 2 — Pre-training tabellare
  4.  Solo Fase 3 — Fine-tuning
  5.  K-Fold cross validation (valutazione affidabile)
  6.  Grid Search → riaddestra con best params → evaluation
  7.  Evaluation
  8.  XAI — GradCAM + Attention
  9.  XAI — SHAP
  10. Quick Run (tutto in sequenza ottimale)
  0.  Esci

Uso:
  python main.py
"""

import os
import sys
import json
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import config as cfg

# ── colori ANSI ───────────────────────────────────────────────────────────────
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


# ══════════════════════════════════════════════════════════════════════════════
# UTILITÀ UI
# ══════════════════════════════════════════════════════════════════════════════

def _header():
    os.system("clear")
    print(f"{BOLD}{CYAN}")
    print("╔══════════════════════════════════════════════════════╗")
    print("║      AlzheimerMultimodalNet — Pipeline Manager      ║")
    print("║      OASIS-2 · MRI + Tabellare · CN/MCI/AD          ║")
    print("╚══════════════════════════════════════════════════════╝")
    print(RESET)


def _section(title: str):
    print(f"\n{BOLD}{YELLOW}{'─'*54}{RESET}")
    print(f"{BOLD}{YELLOW}  {title}{RESET}")
    print(f"{BOLD}{YELLOW}{'─'*54}{RESET}")


def _ok(msg: str):
    print(f"{GREEN}  ✓ {msg}{RESET}", flush=True)


def _warn(msg: str):
    print(f"{YELLOW}  ⚠ {msg}{RESET}", flush=True)


def _err(msg: str):
    print(f"{RED}  ✗ {msg}{RESET}", flush=True)


def _ask(prompt: str, default: str = "") -> str:
    val = input(f"\n{BOLD}  {prompt}{RESET} [{default}]: ").strip()
    return val if val else default


def _confirm(msg: str) -> bool:
    val = input(f"\n{BOLD}  {msg} (s/n): {RESET}").strip().lower()
    return val in ("s", "si", "y", "yes", "1")


def _run(cmd: list, cwd: str = ROOT) -> int:
    """Esegue un comando mostrando output in tempo reale."""
    print(f"\n{CYAN}  $ {' '.join(cmd)}{RESET}\n", flush=True)
    result = subprocess.run(cmd, cwd=cwd)
    return result.returncode


def _check_ckpt(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 0


def _status_pipeline() -> dict:
    manifest     = os.path.join(ROOT, "data", "processed", "manifest.csv")
    gs_report    = os.path.join(ROOT, "outputs", "kfold", "grid_search_report.json")
    kfold_result = os.path.join(ROOT, "outputs", "kfold", "default_kfold.json")
    eval_result  = os.path.join(ROOT, "outputs", "evaluation",
                                 "FULL_MULTIMODAL_test_report.json")
    xai_dir      = os.path.join(ROOT, "outputs", "xai")
    xai_done     = any(f.endswith("_gradcam_slices.png")
                       for f in os.listdir(xai_dir)) if os.path.exists(xai_dir) else False

    return {
        "preprocessing":    os.path.exists(manifest),
        "pretrain_tabular": _check_ckpt(CKPT_TAB),
        "pretrain_imaging": _check_ckpt(CKPT_IMG),
        "finetune":         _check_ckpt(cfg.MODEL_CONFIGS["4"]["checkpoint"]),
        "kfold":            os.path.exists(kfold_result),
        "grid_search":      os.path.exists(gs_report),
        "evaluation":       os.path.exists(eval_result),
        "xai":              xai_done,
    }


def _load_best_params() -> dict | None:
    """Legge i migliori iperparametri dal report del grid search."""
    gs_report = os.path.join(ROOT, "outputs", "kfold", "grid_search_report.json")
    if not os.path.exists(gs_report):
        return None
    with open(gs_report) as f:
        report = json.load(f)
    return report.get("best_params")


# ── path checkpoint ────────────────────────────────────────────────────────────
CKPT_IMG = os.path.join(cfg.CHECKPOINT_DIR, "pretrain_imaging.pt")
CKPT_TAB = os.path.join(cfg.CHECKPOINT_DIR, "pretrain_tabular.pt")


# ══════════════════════════════════════════════════════════════════════════════
# MENU PRINCIPALE
# ══════════════════════════════════════════════════════════════════════════════

def _print_main_menu(status: dict):
    _header()
    _section("Stato pipeline")

    steps = [
        ("preprocessing",    "Preprocessing"),
        ("pretrain_tabular", "Pre-training tabellare (Fase 2)"),
        ("pretrain_imaging", "Pre-training imaging (Fase 1)"),
        ("finetune",         "Fine-tuning end-to-end (Fase 3)"),
        ("kfold",            "K-Fold cross validation"),
        ("grid_search",      "Grid Search iperparametri"),
        ("evaluation",       "Evaluation sul test set"),
        ("xai",              "XAI (GradCAM + Attention)"),
    ]
    for key, label in steps:
        icon = f"{GREEN}✓{RESET}" if status.get(key) else f"{RED}○{RESET}"
        print(f"     {icon}  {label}")

    # mostra best params se disponibili
    best = _load_best_params()
    if best:
        print(f"\n  {CYAN}Best params (Grid Search):{RESET}")
        for k, v in best.items():
            print(f"     {k}: {v}")

    print(f"\n{BOLD}  Azioni:{RESET}")
    print("     [1]   Preprocessing")
    print("     [2]   Training (Fase 2 + 3)")
    print("     [3]   Solo Fase 2 — Pre-training tabellare")
    print("     [4]   Solo Fase 3 — Fine-tuning")
    print("     [5]   K-Fold (valutazione affidabile)")
    print("     [6]   Grid Search → riaddestra con best params")
    print("     [7]   Evaluation")
    print("     [8]   XAI — GradCAM + Attention")
    print("     [9]   XAI — SHAP feature importance")
    print("     [10]  Quick Run (pipeline ottimale completa)")
    print("     [0]   Esci")


# ══════════════════════════════════════════════════════════════════════════════
# AZIONI
# ══════════════════════════════════════════════════════════════════════════════

def do_preprocessing():
    _section("Preprocessing")
    force = _confirm("Forzare rielaborazione anche se i file esistono?")
    cmd = [sys.executable, "preprocessing/preprocessing.py"]
    if force:
        cmd.append("--force")
    rc = _run(cmd)
    if rc == 0:
        _ok("Preprocessing completato")
    else:
        _err("Preprocessing fallito")


def do_training():
    _section("Training (Fase 2 + 3)")
    epochs   = _ask("Numero epoch", str(cfg.N_EPOCHS))
    patience = _ask("Patience", str(cfg.PATIENCE))
    bs       = _ask("Batch size", str(cfg.BATCH_SIZE))
    lr_enc   = _ask("LR encoders", "1e-5")
    lr_fus   = _ask("LR fusion+head", "1e-4")

    # controlla se usare best params da grid search
    best = _load_best_params()
    if best:
        _ok(f"Trovati best params dal Grid Search: {best}")
        use_best = _confirm("Usare questi parametri per il training?")
        if use_best:
            lr_enc   = str(best.get("lr_encoders", lr_enc))
            lr_fus   = str(best.get("lr_fusion",   lr_fus))
            bs       = str(best.get("batch_size",  bs))

    # Fase 2
    _section("Fase 2 — Pre-training tabellare")
    rc = _run([sys.executable, "training/train.py", "--phase", "2",
               "--epochs", epochs, "--patience", patience, "--batch_size", bs])
    if rc != 0:
        _err("Fase 2 fallita")
        if not _confirm("Continuare con Fase 3?"):
            return

    # Fase 3
    _section("Fase 3 — Fine-tuning")
    rc = _run([sys.executable, "training/train.py", "--phase", "3",
               "--epochs", epochs, "--patience", patience,
               "--batch_size", bs,
               "--lr_encoders", lr_enc, "--lr_fusion", lr_fus])
    if rc == 0:
        _ok("Training completato")
    else:
        _err("Fase 3 fallita")


def do_pretrain_tabular():
    _section("Solo Fase 2 — Pre-training tabellare")
    epochs   = _ask("Numero epoch", str(cfg.N_EPOCHS))
    patience = _ask("Patience", str(cfg.PATIENCE))
    rc = _run([sys.executable, "training/train.py", "--phase", "2",
               "--epochs", epochs, "--patience", patience])
    if rc == 0:
        _ok("Pre-training tabellare completato")


def do_finetune():
    _section("Solo Fase 3 — Fine-tuning")

    # controlla best params
    best = _load_best_params()
    if best:
        _ok(f"Best params disponibili: {best}")
        use_best = _confirm("Usare best params dal Grid Search?")
    else:
        use_best = False

    epochs   = _ask("Numero epoch", str(cfg.N_EPOCHS))
    patience = _ask("Patience", str(cfg.PATIENCE))

    if use_best and best:
        lr_enc = str(best.get("lr_encoders", "1e-5"))
        lr_fus = str(best.get("lr_fusion",   "1e-4"))
        bs     = str(best.get("batch_size",  cfg.BATCH_SIZE))
        print(f"  Usando: lr_enc={lr_enc}  lr_fus={lr_fus}  bs={bs}")
    else:
        lr_enc = _ask("LR encoders", "1e-5")
        lr_fus = _ask("LR fusion+head", "1e-4")
        bs     = _ask("Batch size", str(cfg.BATCH_SIZE))

    rc = _run([sys.executable, "training/train.py", "--phase", "3",
               "--epochs", epochs, "--patience", patience,
               "--batch_size", bs,
               "--lr_encoders", lr_enc, "--lr_fusion", lr_fus])
    if rc == 0:
        _ok("Fine-tuning completato")


def do_kfold():
    _section("K-Fold cross validation")
    _warn("Il K-Fold valuta le metriche ma NON produce il checkpoint finale.")
    _warn("Usa Grid Search (opzione 6) per ottimizzare e riaddestrare.")

    n_splits = _ask("Numero fold", str(cfg.N_SPLITS))
    epochs   = _ask("Epoch per fold", "30")
    patience = _ask("Patience per fold", "5")

    rc = _run([sys.executable, "training/train.py", "--kfold",
               "--n_splits", n_splits, "--epochs", epochs,
               "--patience", patience])

    if rc == 0:
        # leggi e mostra risultati
        kfold_path = os.path.join(ROOT, "outputs", "kfold", "default_kfold.json")
        if os.path.exists(kfold_path):
            with open(kfold_path) as f:
                res = json.load(f)
            _ok(f"K-Fold completato:")
            print(f"     F1 : {res['f1_mean']:.4f} ± {res['f1_std']:.4f}")
            print(f"     AUC: {res['auc_mean']:.4f} ± {res['auc_std']:.4f}")
            print(f"     Acc: {res['acc_mean']:.4f} ± {res['acc_std']:.4f}")


def do_grid_search_and_retrain():
    """
    Pipeline completa Grid Search:
    1. Esegue grid search su tutte le combinazioni con K-Fold
    2. Legge i best params dal report
    3. Riaddestra Fase 2 + Fase 3 con i best params
    4. Esegue evaluation automaticamente
    """
    _section("Grid Search → Riaddestra → Evaluation")

    print(f"\n  {BOLD}Questo processo:{RESET}")
    print("   1. Esegue Grid Search (K-Fold per ogni combinazione HP)")
    print("   2. Identifica i migliori iperparametri")
    print("   3. Riaddestra Fase 2 + Fase 3 con i best params")
    print("   4. Valuta sul test set")

    fast     = _confirm("Modalità FAST (4 combo × 3 fold, ~45 min)?")
    n_splits = _ask("Numero fold per Grid Search", "3")
    epochs_gs= _ask("Epoch per fold (Grid Search)", "20")
    pat_gs   = _ask("Patience (Grid Search)", "5")
    epochs_ft= _ask("Epoch per riaddestramento finale", str(cfg.N_EPOCHS))
    pat_ft   = _ask("Patience per riaddestramento finale", str(cfg.PATIENCE))

    # ── Step 1: Grid Search ───────────────────────────────────────────────
    _section("Step 1/3 — Grid Search")
    cmd = [sys.executable, "training/train.py", "--grid_search",
           "--n_splits", n_splits, "--epochs", epochs_gs, "--patience", pat_gs]
    if fast:
        cmd.append("--fast")

    rc = _run(cmd)
    if rc != 0:
        _err("Grid Search fallita")
        return

    # ── Step 2: Leggi best params ─────────────────────────────────────────
    best = _load_best_params()
    if not best:
        _err("Impossibile leggere best params dal report")
        return

    _ok("Best params trovati:")
    for k, v in best.items():
        print(f"     {k}: {v}")

    if not _confirm("Procedere con il riaddestramento usando questi params?"):
        return

    # ── Step 3: Riaddestra Fase 2 + 3 con best params ────────────────────
    _section("Step 2/3 — Riaddestramento con best params")

    lr_enc = str(best.get("lr_encoders", "1e-5"))
    lr_fus = str(best.get("lr_fusion",   "1e-4"))
    bs     = str(best.get("batch_size",  cfg.BATCH_SIZE))

    # Fase 2
    print(f"\n  {BOLD}Fase 2 — Pre-training tabellare{RESET}", flush=True)
    rc = _run([sys.executable, "training/train.py", "--phase", "2",
               "--epochs", epochs_ft, "--patience", pat_ft, "--batch_size", bs])
    if rc != 0:
        _err("Fase 2 fallita")
        if not _confirm("Continuare con Fase 3?"):
            return

    # Fase 3
    print(f"\n  {BOLD}Fase 3 — Fine-tuning end-to-end{RESET}", flush=True)
    rc = _run([sys.executable, "training/train.py", "--phase", "3",
               "--epochs", epochs_ft, "--patience", pat_ft,
               "--batch_size", bs, "--lr_encoders", lr_enc, "--lr_fusion", lr_fus])
    if rc != 0:
        _err("Fase 3 fallita")
        return

    _ok("Riaddestramento completato con best params")

    # ── Step 4: Evaluation ────────────────────────────────────────────────
    _section("Step 3/3 — Evaluation")
    rc = _run([sys.executable, "evaluation/evaluation.py",
               "--model_cfg", "4", "--split", "test"])
    if rc == 0:
        # mostra risultati
        eval_path = os.path.join(ROOT, "outputs", "evaluation",
                                  "FULL_MULTIMODAL_test_report.json")
        if os.path.exists(eval_path):
            with open(eval_path) as f:
                res = json.load(f)
            _ok("Evaluation completata:")
            print(f"     F1 macro : {res['f1_macro']:.4f}")
            print(f"     AUC macro: {res['auc_macro']:.4f}")
            print(f"     Accuracy : {res['accuracy']:.4f}")
            print(f"     ECE      : {res['ece']:.4f}")
    else:
        _err("Evaluation fallita")


def do_evaluation():
    _section("Evaluation")
    print(f"\n{BOLD}  Split:{RESET}")
    print("     [1] Test set")
    print("     [2] Val set")
    print("     [3] Ablation study (confronta tutte le config)")
    choice = _ask("Scelta", "1")

    if choice == "3":
        split = _ask("Split", "test")
        rc = _run([sys.executable, "evaluation/evaluation.py",
                   "--compare", "--split", split])
    else:
        split = "test" if choice == "1" else "val"
        rc = _run([sys.executable, "evaluation/evaluation.py",
                   "--model_cfg", "4", "--split", split])

    if rc == 0:
        eval_path = os.path.join(ROOT, "outputs", "evaluation",
                                  f"FULL_MULTIMODAL_{split}_report.json")
        if os.path.exists(eval_path):
            with open(eval_path) as f:
                res = json.load(f)
            _ok(f"Evaluation {split}:")
            print(f"     F1 macro : {res['f1_macro']:.4f}")
            print(f"     AUC macro: {res['auc_macro']:.4f}")
            print(f"     Accuracy : {res['accuracy']:.4f}")
            print(f"     ECE      : {res['ece']:.4f}")


def do_xai_gradcam():
    _section("XAI — GradCAM 3D + Attention weights")

    # mostra mri_id disponibili nel test set
    try:
        import pandas as pd
        from sklearn.model_selection import train_test_split

        manifest = pd.read_csv(
            os.path.join(ROOT, "data", "processed", "manifest.csv"))
        subj_labels = (manifest.groupby("subject_id")["label"]
                       .agg(lambda x: x.value_counts().idxmax())
                       .reset_index())
        subj_ids = subj_labels["subject_id"].values
        labels   = subj_labels["label"].values

        _, subj_temp, _, lab_temp = train_test_split(
            subj_ids, labels, test_size=0.30,
            stratify=labels, random_state=cfg.SEED)
        _, subj_test = train_test_split(
            subj_temp, test_size=0.50,
            stratify=lab_temp, random_state=cfg.SEED)

        test_m = manifest[
            manifest["subject_id"].isin(subj_test) & manifest["has_3d"]]
        class_map = {0: "CN", 1: "MCI", 2: "AD"}

        print(f"\n  {BOLD}Sample nel test set con volume 3D:{RESET}")
        for _, row in test_m.iterrows():
            label_name = class_map.get(row["label"], "?")
            print(f"     {row['mri_id']:<22} ({label_name})")
    except Exception as e:
        _warn(f"Impossibile caricare lista: {e}")

    print()
    all_samples = _confirm("Eseguire XAI su tutti i sample del test set?")

    if all_samples:
        rc = _run([sys.executable, "explainability/xai.py",
                   "--model_cfg", "4", "--all"])
    else:
        mri_id = _ask("mri_id da analizzare", "OAS2_0004_MR1")
        rc = _run([sys.executable, "explainability/xai.py",
                   "--model_cfg", "4", "--mri_id", mri_id])

    if rc == 0:
        _ok("XAI completata → outputs/xai/")
        xai_dir = os.path.join(ROOT, "outputs", "xai")
        if _confirm("Aprire la cartella outputs/xai/?"):
            subprocess.run(["open", xai_dir])


def do_xai_shap():
    _section("XAI — SHAP feature importance")
    _warn("Richiede: pip install shap")
    _warn("Può richiedere 10-15 minuti")
    if _confirm("Procedere?"):
        rc = _run([sys.executable, "explainability/xai.py",
                   "--model_cfg", "4", "--shap_only"])
        if rc == 0:
            _ok("SHAP completato → outputs/xai/")


def do_quick_run():
    """
    Pipeline ottimale completa:
    1. Preprocessing
    2. Fase 2 — Pretrain tabellare
    3. Fase 3 — Fine-tuning
    4. K-Fold (valutazione)
    5. Evaluation
    6. XAI sui sample più interessanti
    """
    _section("Quick Run — Pipeline ottimale completa")

    print(f"\n  {BOLD}La pipeline eseguirà:{RESET}")
    print("   1. Preprocessing (se non già fatto)")
    print("   2. Fase 2 — Pre-training tabellare")
    print("   3. Fase 3 — Fine-tuning end-to-end")
    print("   4. K-Fold (5 fold, metriche affidabili)")
    print("   5. Evaluation sul test set")
    print("   6. XAI su campioni CN e MCI")

    if not _confirm("Confermi?"):
        return

    epochs   = _ask("Epoch", str(cfg.N_EPOCHS))
    patience = _ask("Patience", str(cfg.PATIENCE))
    kf_ep    = _ask("Epoch per fold K-Fold", "30")
    kf_pat   = _ask("Patience K-Fold", "5")

    steps = [
        # preprocessing solo se manifest non esiste
        ("preprocessing", lambda: _run(
            [sys.executable, "preprocessing/preprocessing.py"])
            if not os.path.exists(
                os.path.join(ROOT, "data", "processed", "manifest.csv"))
            else 0),

        ("Fase 2 — Pre-training tabellare", lambda: _run(
            [sys.executable, "training/train.py", "--phase", "2",
             "--epochs", epochs, "--patience", patience])),

        ("Fase 3 — Fine-tuning", lambda: _run(
            [sys.executable, "training/train.py", "--phase", "3",
             "--epochs", epochs, "--patience", patience])),

        ("K-Fold (5 fold)", lambda: _run(
            [sys.executable, "training/train.py", "--kfold",
             "--n_splits", "5", "--epochs", kf_ep, "--patience", kf_pat])),

        ("Evaluation test set", lambda: _run(
            [sys.executable, "evaluation/evaluation.py",
             "--model_cfg", "4", "--split", "test"])),

        ("XAI — OAS2_0004_MR1 (CN)", lambda: _run(
            [sys.executable, "explainability/xai.py",
             "--model_cfg", "4", "--mri_id", "OAS2_0004_MR1"])),

        ("XAI — OAS2_0023_MR1 (MCI→AD)", lambda: _run(
            [sys.executable, "explainability/xai.py",
             "--model_cfg", "4", "--mri_id", "OAS2_0023_MR1"])),
    ]

    for label, func in steps:
        _section(label)
        try:
            rc = func()
            if rc != 0:
                _err(f"'{label}' terminato con errore (rc={rc})")
                if not _confirm("Continuare comunque?"):
                    return
            else:
                _ok(f"{label} completato")
        except Exception as e:
            _err(f"Errore in '{label}': {e}")
            if not _confirm("Continuare comunque?"):
                return

    # riepilogo finale
    _section("Riepilogo finale")
    eval_path = os.path.join(ROOT, "outputs", "evaluation",
                              "FULL_MULTIMODAL_test_report.json")
    kfold_path = os.path.join(ROOT, "outputs", "kfold", "default_kfold.json")

    if os.path.exists(eval_path):
        with open(eval_path) as f:
            res = json.load(f)
        print(f"\n  {BOLD}Test set:{RESET}")
        print(f"     F1 macro : {res['f1_macro']:.4f}")
        print(f"     AUC macro: {res['auc_macro']:.4f}")
        print(f"     Accuracy : {res['accuracy']:.4f}")

    if os.path.exists(kfold_path):
        with open(kfold_path) as f:
            res = json.load(f)
        print(f"\n  {BOLD}K-Fold (5 fold):{RESET}")
        print(f"     F1  : {res['f1_mean']:.4f} ± {res['f1_std']:.4f}")
        print(f"     AUC : {res['auc_mean']:.4f} ± {res['auc_std']:.4f}")

    _ok("Pipeline completa!")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════

def main():
    actions = {
        "1":  do_preprocessing,
        "2":  do_training,
        "3":  do_pretrain_tabular,
        "4":  do_finetune,
        "5":  do_kfold,
        "6":  do_grid_search_and_retrain,
        "7":  do_evaluation,
        "8":  do_xai_gradcam,
        "9":  do_xai_shap,
        "10": do_quick_run,
    }

    while True:
        status = _status_pipeline()
        _print_main_menu(status)

        choice = input(f"\n{BOLD}  Scelta: {RESET}").strip()

        if choice == "0":
            print(f"\n{CYAN}  Arrivederci!{RESET}\n")
            break
        elif choice in actions:
            try:
                actions[choice]()
            except KeyboardInterrupt:
                _warn("\nInterrotto dall'utente.")
            input(f"\n{BOLD}  Premi INVIO per tornare al menu...{RESET}")
        else:
            _warn("Scelta non valida.")


if __name__ == "__main__":
    main()
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================================================
# DEVICE
# =========================================================

if torch.cuda.is_available():
    DEVICE = "cuda"
    print("\n[DEVICE] CUDA disponibile")
elif torch.backends.mps.is_available():
    DEVICE = "mps"
    print("\n[DEVICE] Apple Silicon MPS disponibile")
else:
    DEVICE = "cpu"
    print("\n[DEVICE] Uso CPU")

# =========================================================
# GENERAL
# =========================================================

SEED = 42
N_CLASSES = 3

# =========================================================
# LABEL MAPPING
# Target: CDR → classe
# 0.0  → 0  (CN  — Cognitivamente Normale)
# 0.5  → 1  (MCI — Mild Cognitive Impairment)
# 1.0  → 2  (AD  — Alzheimer's Disease)
# 2.0  → 2  (AD  — merge con 1.0, solo 3 casi in OASIS-2)
# =========================================================

CDR_TO_CLASS = {0.0: 0, 0.5: 1, 1.0: 2, 2.0: 2}
CLASS_NAMES = ["CN", "MCI", "AD"]

# =========================================================
# PATHS – RAW DATA
# =========================================================

RAW_MRI_DIR = os.path.join(BASE_DIR, "data", "raw", "mri")
RAW_TABULAR_FILE = os.path.join(BASE_DIR, "data", "raw", "Tabellare.xlsx")

# =========================================================
# PATHS – PROCESSED DATA
# =========================================================

PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
SLICES_2D_DIR = os.path.join(PROCESSED_DIR, "slices_2d")
VOLUMES_3D_DIR = os.path.join(PROCESSED_DIR, "volumes_3d")
TABULAR_CSV = os.path.join(PROCESSED_DIR, "tabular.csv")
SURVIVAL_CSV = os.path.join(PROCESSED_DIR, "survival_labels.csv")
MANIFEST_CSV = os.path.join(PROCESSED_DIR, "manifest.csv")

# Colonne obbligatorie nel manifest.csv
# Il manifest collega ogni paziente ai file disponibili
MANIFEST_SCHEMA = {
    "required": ["subject_id", "label"],
    "optional": ["slice_2d_path", "volume_3d_path", "tabular_available"],
}

# =========================================================
# PATHS – ARTIFACTS & CHECKPOINTS
# =========================================================

ARTIFACTS_DIR = os.path.join(BASE_DIR, "data", "artifacts")
TABULAR_SCALER_PATH = os.path.join(ARTIFACTS_DIR, "tabular_scaler.pkl")

CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# =========================================================
# MRI PREPROCESSING
# =========================================================

SIZE_2D = (224, 224)    # H x W per le slice 2D
SIZE_3D = (96, 96, 96) # D x H x W — standard Med3D/ADNI
SLICE_PCT_2D = 0.45          # percentuale lungo ogni asse per slice centrale

# =========================================================
# TABULAR FEATURES
# NOTA: CDR rimossa — è il target, tenerla causerebbe data leakage.
# MMSE è una misurazione cognitiva indipendente dal CDR → legittima come feature.
# =========================================================

FEATURE_COLUMNS = ["Age", "EDUC", "SES", "MMSE", "eTIV", "nWBV", "ASF"]
N_TABULAR_FEATURES = len(FEATURE_COLUMNS)

# =========================================================
# SURVIVAL ANALYSIS
# Intervalli temporali discreti in giorni
# =========================================================

SURVIVAL_TIME_BINS = [0, 180, 365, 730, 1095, 1825, 2555]
N_SURVIVAL_BINS = len(SURVIVAL_TIME_BINS) - 1

# =========================================================
# MODEL ARCHITECTURE
# =========================================================

EMB_DIM = 256
N_HEADS = 8
N_TRANSFORMER_LAYERS = 2
DROPOUT = 0.3

# MoE
N_EXPERTS = 4
MOE_TOP_K = 2

# =========================================================
# CROSS-MODAL ATTENTION – coppie dinamiche per config
#
# Ogni config del MODEL_REGISTRY dichiara quali coppie
# bidirezionali attivare. Il fusion module legge questa
# lista e salta le coppie con modalità assenti.
#
# Formato: lista di tuple ("mod_a", "mod_b")
# Modalità disponibili: "tabular", "2d", "3d"
# =========================================================

CROSS_ATTN_PAIRS_ALL = [("tabular", "2d"), ("tabular", "3d"), ("2d", "3d")]
CROSS_ATTN_PAIRS_TAB2D = [("tabular", "2d")]
CROSS_ATTN_PAIRS_TAB3D = [("tabular", "3d")]
CROSS_ATTN_PAIRS_2D3D = [("2d", "3d")]

# =========================================================
# TRAINING
# =========================================================

BATCH_SIZE = 2
LR = 1e-4
WEIGHT_DECAY = 1e-5
N_EPOCHS = 100
PATIENCE = 10

# Split
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15   # i 100 ospedalieri usati come test esterno

# Weighted cross-entropy per sbilanciamento CN/MCI/AD
USE_CLASS_WEIGHTS = True

# Loss weights per multi-task learning
LAMBDA_CLASS = 1.0
LAMBDA_SURV  = 1.0
LAMBDA_REGR  = 0.5

# K-FOLD
N_SPLITS = 5

# MODALITY DROPOUT – simula modalità mancanti durante training
# per rendere il modello robusto anche a input parziali
MODALITY_DROPOUT = 0.2

# =========================================================
# MODEL REGISTRY
# Ogni entry definisce quali modalità sono attive e
# quali coppie cross-attention vengono usate.
# =========================================================

MODEL_2D_3D = os.path.join(CHECKPOINT_DIR, "model_2d_3d.pt")
MODEL_TABULAR_2D = os.path.join(CHECKPOINT_DIR, "model_tabular_2d.pt")
MODEL_TABULAR_3D = os.path.join(CHECKPOINT_DIR, "model_tabular_3d.pt")
MODEL_FULL = os.path.join(CHECKPOINT_DIR, "model_full.pt")

MODEL_CONFIGS = {
    "1": {
        "name":             "MRI_2D_3D",
        "checkpoint":       MODEL_2D_3D,
        "tabular":          False,
        "2d":               True,
        "3d":               True,
        "cross_attn_pairs": CROSS_ATTN_PAIRS_2D3D,
    },
    "2": {
        "name":             "TABULAR_2D",
        "checkpoint":       MODEL_TABULAR_2D,
        "tabular":          True,
        "2d":               True,
        "3d":               False,
        "cross_attn_pairs": CROSS_ATTN_PAIRS_TAB2D,
    },
    "3": {
        "name":             "TABULAR_3D",
        "checkpoint":       MODEL_TABULAR_3D,
        "tabular":          True,
        "2d":               False,
        "3d":               True,
        "cross_attn_pairs": CROSS_ATTN_PAIRS_TAB3D,
    },
    "4": {
        "name":             "FULL_MULTIMODAL",
        "checkpoint":       MODEL_FULL,
        "tabular":          True,
        "2d":               True,
        "3d":               True,
        "cross_attn_pairs": CROSS_ATTN_PAIRS_ALL,
    },
}


# =========================================================
# UTILITY – stampa riepilogo config
# =========================================================

def print_summary():
    print("=" * 52)
    print("  Alzheimer Multimodal – Config")
    print("=" * 52)
    print(f"  Device          : {DEVICE}")
    print(f"  Embed dim       : {EMB_DIM}  |  Heads: {N_HEADS}")
    print(f"  Classi          : {N_CLASSES} ({' / '.join(CLASS_NAMES)})")
    print(f"  Size 2D         : {SIZE_2D}")
    print(f"  Size 3D         : {SIZE_3D}")
    print(f"  Tabular feat.   : {N_TABULAR_FEATURES} ({', '.join(FEATURE_COLUMNS)})")
    print(f"  Epochs          : {N_EPOCHS}  |  Batch: {BATCH_SIZE}  |  LR: {LR}")
    print(f"  Modality dropout: {MODALITY_DROPOUT}")
    print(f"  OASIS MRI dir   : {RAW_MRI_DIR}")
    print("=" * 52)
    print("\n  Model configs disponibili:")
    for k, v in MODEL_CONFIGS.items():
        modalities = " + ".join(
            m for m in ["tabular", "2d", "3d"] if v[m]
        )
        pairs = ", ".join(f"{a}↔{b}" for a, b in v["cross_attn_pairs"])
        print(f"  [{k}] {v['name']:<20} modalità: {modalities}")
        print(f"       cross-attn pairs : {pairs}")
    print("=" * 52)


if __name__ == "__main__":
    print_summary()
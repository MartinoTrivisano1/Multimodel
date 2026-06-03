"""
explainability/xai.py
=====================
Modulo XAI completo per AlzheimerMultimodalNet con backbone MONAI.

Fix: GradCAM3D aggiornato per MONAI ResNet18 (layer4[-1].conv2)
     con fallback automatico alla CNN leggera.

Tre livelli di spiegabilità:
  1. GradCAM 3D  → heatmap volumetrica sulle regioni cerebrali
  2. Attention weights → importanza inter-modale (XAI intrinseca MADDi)
  3. SHAP → feature importance sui dati tabellari e volumetrici
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import config as cfg
from dataset.dataset import build_dataloaders
from models.model    import build_model, AlzheimerMultimodalNet

XAI_DIR = os.path.join(ROOT, "outputs", "xai")
os.makedirs(XAI_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# UTILITÀ
# ══════════════════════════════════════════════════════════════════════════════

def _move_batch(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()}


def _load_model(model_cfg: str, device: torch.device) -> AlzheimerMultimodalNet:
    mcfg      = cfg.MODEL_CONFIGS[model_cfg]
    ckpt_path = mcfg["checkpoint"]
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint non trovato: {ckpt_path}")
    model = build_model(model_cfg=model_cfg, pretrained_2d=False, device=str(device))
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    print(f"  Checkpoint caricato: {ckpt_path}")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# 1. GRADCAM 3D
# ══════════════════════════════════════════════════════════════════════════════

class GradCAM3D:
    """
    GradCAM applicato all'encoder 3D.

    Supporta due backbone:
      - MONAI ResNet18: aggancia a layer4[-1].conv2
      - CNN leggera (fallback): aggancia a backbone[3].conv[0]

    Produce heatmap volumetrica (D, H, W) normalizzata in [0,1].
    """

    def __init__(self, model: AlzheimerMultimodalNet):
        self.model         = model
        self.activations   = None
        self.gradients     = None
        self._hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        backbone = self.model.enc_3d.backbone

        # MONAI ResNet18: backbone è Sequential con layer4 come attributo
        if hasattr(backbone, 'layer4'):
            # MONAI ResNet18 — ultimo BasicBlock, ultimo conv
            target_layer = backbone.layer4[-1].conv2
            print("  [GradCAM] Hook su MONAI ResNet18 layer4[-1].conv2")
        elif hasattr(backbone[-1], 'conv'):
            # CNN leggera — ultimo ConvBlock3D
            target_layer = backbone[-1].conv[0]
            print("  [GradCAM] Hook su CNN leggera backbone[-1].conv[0]")
        else:
            # fallback generico — cerca il primo Conv3d nel backbone
            target_layer = None
            for m in backbone.modules():
                if isinstance(m, torch.nn.Conv3d):
                    target_layer = m
            if target_layer is None:
                raise RuntimeError("Nessun layer Conv3d trovato nel backbone 3D")
            print(f"  [GradCAM] Hook su Conv3d generico: {target_layer}")

        def fwd_hook(module, input, output):
            self.activations = output.detach()

        def bwd_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self._hook_handles.append(target_layer.register_forward_hook(fwd_hook))
        self._hook_handles.append(target_layer.register_full_backward_hook(bwd_hook))

    def remove_hooks(self):
        for h in self._hook_handles:
            h.remove()

    def compute(self, batch: dict, target_class: int = None) -> np.ndarray:
        self.model.eval()
        self.model.zero_grad()

        batch["volume_3d"].requires_grad_(True)
        logits = self.model(batch)

        if target_class is None:
            target_class = logits.argmax(dim=-1).item()

        logits[0, target_class].backward()

        # GradCAM: media gradienti sui canali → pesi
        # activations: (1, C, D, H, W)
        weights = self.gradients.mean(dim=list(range(2, self.gradients.dim())),
                                      keepdim=True)
        cam     = (weights * self.activations).sum(dim=1)
        cam     = torch.relu(cam).squeeze(0).cpu().numpy()

        # normalizza in [0, 1]
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)

        # interpola alla dimensione originale
        from torch.nn.functional import interpolate
        cam_t = torch.from_numpy(cam).unsqueeze(0).unsqueeze(0)
        cam_t = interpolate(cam_t, size=cfg.SIZE_3D,
                            mode="trilinear", align_corners=False)
        return cam_t.squeeze().numpy().astype(np.float32)


def run_gradcam(
    model:        AlzheimerMultimodalNet,
    batch:        dict,
    mri_id:       str,
    device:       torch.device,
    target_class: int = None,
) -> np.ndarray:
    batch   = _move_batch(batch, device)
    gcam    = GradCAM3D(model)
    heatmap = gcam.compute(batch, target_class=target_class)
    gcam.remove_hooks()

    # salva heatmap raw
    np.save(os.path.join(XAI_DIR, f"{mri_id}_gradcam.npy"), heatmap)
    print(f"  GradCAM 3D → {XAI_DIR}/{mri_id}_gradcam.npy")

    # visualizzazione
    vol     = batch["volume_3d"].squeeze().detach().cpu().numpy()
    vol_vis = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)
    D, H, W = vol_vis.shape

    slices = {
        "Assiale":   (vol_vis[D//2],       heatmap[D//2]),
        "Coronale":  (vol_vis[:,H//2,:],   heatmap[:,H//2,:]),
        "Sagittale": (vol_vis[:,:,W//2],   heatmap[:,:,W//2]),
    }

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    with torch.no_grad():
        pred_name = cfg.CLASS_NAMES[
            model(batch).argmax(dim=-1).item()
            if target_class is None else target_class
        ]

    for col, (title, (sl_vol, sl_cam)) in enumerate(slices.items()):
        axes[0, col].imshow(sl_vol, cmap="gray", origin="lower")
        axes[0, col].set_title(f"{title} – MRI")
        axes[0, col].axis("off")
        axes[1, col].imshow(sl_vol, cmap="gray", origin="lower")
        axes[1, col].imshow(sl_cam, cmap="jet", alpha=0.45,
                            origin="lower", vmin=0, vmax=1)
        axes[1, col].set_title(f"{title} – GradCAM")
        axes[1, col].axis("off")

    fig.suptitle(f"GradCAM 3D | {mri_id} | Predizione: {pred_name}", fontsize=13)
    sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(0, 1))
    plt.colorbar(sm, ax=axes[1,:], orientation="horizontal",
                 fraction=0.03, pad=0.04, label="Importanza")
    plt.tight_layout()

    fig_path = os.path.join(XAI_DIR, f"{mri_id}_gradcam_slices.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  GradCAM slices → {fig_path}")
    return heatmap


# ══════════════════════════════════════════════════════════════════════════════
# 2. ATTENTION WEIGHTS
# ══════════════════════════════════════════════════════════════════════════════

def run_attention_xai(
    model:  AlzheimerMultimodalNet,
    batch:  dict,
    mri_id: str,
    device: torch.device,
) -> dict:
    batch = _move_batch(batch, device)
    with torch.no_grad():
        logits, weights = model(batch, return_weights=True)

    pred_class = logits.argmax(dim=-1).item()
    pred_name  = cfg.CLASS_NAMES[pred_class]
    pred_prob  = torch.softmax(logits, dim=-1)[0, pred_class].item()

    report = {
        "mri_id":      mri_id,
        "prediction":  pred_name,
        "confidence":  round(pred_prob, 4),
        "probabilities": {
            cfg.CLASS_NAMES[i]: round(
                torch.softmax(logits, dim=-1)[0, i].item(), 4)
            for i in range(cfg.N_CLASSES)
        },
        "attention_weights": {},
        "attention_shapes":  {},
    }

    for pair_name, w_tuple in weights.items():
        if w_tuple is None:
            continue
        w_ab, w_ba = w_tuple
        # media scalare per il grafico
        report["attention_weights"][pair_name] = {
            "A_to_B": round(float(w_ab.mean().cpu()), 6),
            "B_to_A": round(float(w_ba.mean().cpu()), 6),
        }
        # shape per riferimento (es. tab_3d: (1, 29, 216))
        report["attention_shapes"][pair_name] = {
            "A_to_B": list(w_ab.shape),
            "B_to_A": list(w_ba.shape),
        }

    json_path = os.path.join(XAI_DIR, f"{mri_id}_attention.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Attention weights → {json_path}")

    # grafico a barre
    if report["attention_weights"]:
        pairs = list(report["attention_weights"].keys())
        a2b   = [report["attention_weights"][p]["A_to_B"] for p in pairs]
        b2a   = [report["attention_weights"][p]["B_to_A"] for p in pairs]
        x     = np.arange(len(pairs))

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(x - 0.175, a2b, 0.35, label="A→B", color="#4C72B0")
        ax.bar(x + 0.175, b2a, 0.35, label="B→A", color="#DD8452")
        ax.set_xticks(x)
        ax.set_xticklabels(pairs)
        ax.set_ylabel("Peso attention (media)")
        ax.set_title(
            f"Cross-Modal Attention | {mri_id} | "
            f"Pred: {pred_name} ({pred_prob:.2%})"
        )
        # aggiungi shape come annotazione
        for i, p in enumerate(pairs):
            sh = report["attention_shapes"][p]["A_to_B"]
            ax.annotate(f"{sh}", (x[i]-0.175, max(a2b[i], 0.01)),
                        ha='center', va='bottom', fontsize=7, color='gray')
        ax.legend()
        plt.tight_layout()
        fig_path = os.path.join(XAI_DIR, f"{mri_id}_attention.png")
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Attention plot  → {fig_path}")

    return report


# ══════════════════════════════════════════════════════════════════════════════
# 3. SHAP
# ══════════════════════════════════════════════════════════════════════════════

class TabularWrapper(nn.Module):
    def __init__(self, model, n_tabular, n_volumetric, device):
        super().__init__()
        self.model        = model
        self.n_tabular    = n_tabular
        self.n_volumetric = n_volumetric
        self.device       = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tab  = x[:, :self.n_tabular]
        vol  = x[:, self.n_tabular:]
        # enc_tab ora restituisce (B, N, D) → mean pool → head
        feat = self.model.enc_tab(tab, vol).mean(dim=1)
        return self.model.head(feat)


def run_shap(model, loader, device, n_background=50, n_explain=100,
             feature_names=None):
    try:
        import shap
    except ImportError:
        print("  [WARN] pip install shap")
        return None

    if feature_names is None:
        feature_names = cfg.FEATURE_COLUMNS + ["sex_enc"] + \
                        [f"vol_{i}" for i in range(21)]

    n_tab = len(cfg.FEATURE_COLUMNS) + 1
    n_vol = 21

    all_x = []
    for batch in loader:
        tab = batch["tabular"].cpu().numpy()
        vol = batch["volumetric"].cpu().numpy()
        all_x.append(np.concatenate([tab, vol], axis=1))
        if sum(x.shape[0] for x in all_x) >= n_background + n_explain:
            break

    all_x  = np.concatenate(all_x, axis=0)
    bg     = all_x[:n_background]
    exp_x  = all_x[n_background:n_background + n_explain]

    wrapper = TabularWrapper(model, n_tab, n_vol, device).to(device)
    wrapper.eval()

    def predict_fn(x):
        t = torch.from_numpy(x).float().to(device)
        with torch.no_grad():
            return torch.softmax(wrapper(t), dim=-1).cpu().numpy()

    print(f"  [SHAP] bg={n_background}  explain={len(exp_x)}")
    explainer   = shap.KernelExplainer(predict_fn, bg)
    shap_values = explainer.shap_values(exp_x, nsamples=200)

    np.save(os.path.join(XAI_DIR, "shap_values.npy"), np.array(shap_values))

    for cls_idx, cls_name in enumerate(cfg.CLASS_NAMES):
        fig, _ = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values[cls_idx], exp_x,
                          feature_names=feature_names,
                          plot_type="bar", show=False, max_display=20)
        plt.title(f"SHAP – Classe {cls_name}")
        plt.tight_layout()
        p = os.path.join(XAI_DIR, f"shap_summary_{cls_name}.png")
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  SHAP {cls_name} → {p}")

    shap_mean = np.mean([np.abs(sv) for sv in shap_values], axis=0)
    fi        = shap_mean.mean(axis=0)
    idx       = fi.argsort()[::-1][:20]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh([feature_names[i] for i in idx[::-1]],
            fi[idx[::-1]], color="#4C72B0")
    ax.set_xlabel("Mean |SHAP|")
    ax.set_title("SHAP – Top 20 global")
    plt.tight_layout()
    p = os.path.join(XAI_DIR, "shap_summary_global.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  SHAP global → {p}")
    return shap_values


# ══════════════════════════════════════════════════════════════════════════════
# RUN PER SAMPLE
# ══════════════════════════════════════════════════════════════════════════════

def run_xai_for_sample(model, loader, mri_id, device):
    target_batch = None
    for batch in loader:
        ids = list(batch["mri_id"])
        if mri_id in ids:
            idx = ids.index(mri_id)
            target_batch = {
                k: v[idx:idx+1] if isinstance(v, torch.Tensor) else [v[idx]]
                for k, v in batch.items()
            }
            break

    if target_batch is None:
        print(f"  [WARN] '{mri_id}' non trovato nel loader")
        return

    print(f"\n  Sample: {mri_id}")
    print(f"  has_2d={target_batch['has_2d'].item()}  "
          f"has_3d={target_batch['has_3d'].item()}  "
          f"has_tabular={target_batch['has_tabular'].item()}")

    if target_batch["has_3d"].item():
        print("\n  [1/2] GradCAM 3D...")
        run_gradcam(model, target_batch, mri_id, device)
    else:
        print("  [1/2] GradCAM 3D → skip")

    print("\n  [2/2] Attention weights...")
    run_attention_xai(model, target_batch, mri_id, device)


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_cfg",  type=str, default="4")
    parser.add_argument("--mri_id",     type=str, default=None)
    parser.add_argument("--shap_only",  action="store_true")
    parser.add_argument("--all",        action="store_true")
    parser.add_argument("--batch_size", type=int, default=cfg.BATCH_SIZE)
    args = parser.parse_args()

    device = torch.device(
        cfg.DEVICE if (cfg.DEVICE != "cuda" or torch.cuda.is_available()) else "cpu"
    )

    print("=" * 56)
    print(f"  XAI – {cfg.MODEL_CONFIGS[args.model_cfg]['name']}")
    print(f"  Device: {device}")
    print("=" * 56)

    model   = _load_model(args.model_cfg, device)
    loaders = build_dataloaders(model_cfg=args.model_cfg,
                                batch_size=args.batch_size)
    test_loader = loaders["test"]

    if args.shap_only:
        run_shap(model, test_loader, device)
    elif args.mri_id:
        run_xai_for_sample(model, test_loader, args.mri_id, device)
    elif args.all:
        run_shap(model, test_loader, device)
        for batch in test_loader:
            for i, mid in enumerate(list(batch["mri_id"])):
                single = {k: v[i:i+1] if isinstance(v, torch.Tensor)
                           else [v[i]] for k, v in batch.items()}
                run_xai_for_sample(model, [single], mid, device)
    else:
        print("  Usa --mri_id, --shap_only o --all")
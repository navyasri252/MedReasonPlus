"""
src/run_gradcam_local.py
=========================
Generate Grad-CAM++ visualisations locally on CPU.

Uses the saved ConvNeXt-Tiny checkpoint for ConvNeXt GradCAM++.
Saves PNG files to reports/gradcam/.

For the hybrid model, downloads the checkpoint from Kaggle if available,
otherwise clearly states it is not available locally.
"""
import json
import random
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import confusion_matrix
from torchvision import transforms
from torchvision.models import convnext_tiny

sys.path.insert(0, str(Path(__file__).resolve().parent))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "archive (7)" / "chest_xray"
CX_CKPT      = PROJECT_ROOT / "processed" / "convnext" / "best_model.pt"
GRADCAM_OUT  = PROJECT_ROOT / "reports" / "gradcam"
GRADCAM_OUT.mkdir(parents=True, exist_ok=True)

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
C2I  = {"NORMAL": 0, "PNEUMONIA": 1}

# ---------------------------------------------------------------------------
# Check availability
# ---------------------------------------------------------------------------
print("=" * 55)
print("  Grad-CAM Local Runner — MedReason++")
print("=" * 55)
print(f"  Dataset   : {DATASET_ROOT}")
print(f"  Checkpoint: {CX_CKPT}")
print(f"  Output    : {GRADCAM_OUT}")
print(f"  Device    : CPU")
print()

if not CX_CKPT.exists():
    print("ERROR: ConvNeXt checkpoint not found. Exiting.")
    sys.exit(1)
if not (DATASET_ROOT / "test").is_dir():
    print("ERROR: Dataset test folder not found. Exiting.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def collect(folder):
    recs = []
    for cls, idx in C2I.items():
        cls_dir = folder / cls
        if cls_dir.is_dir():
            for p in sorted(cls_dir.glob("*.jpeg")):
                recs.append({"path": p, "label": idx, "class_name": cls})
    return recs

eval_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

def denorm(t):
    t = t.clone()
    for c, m, s in zip(range(3), MEAN, STD):
        t[c] = t[c] * s + m
    return t.permute(1, 2, 0).clamp(0, 1).numpy()

ts = collect(DATASET_ROOT / "test")
print(f"Test images: {len(ts)}  (NORMAL={sum(1 for r in ts if r['label']==0)}  "
      f"PNEUMONIA={sum(1 for r in ts if r['label']==1)})")

# ---------------------------------------------------------------------------
# Load ConvNeXt-Tiny
# ---------------------------------------------------------------------------
print("\nLoading ConvNeXt-Tiny checkpoint...")
device = torch.device("cpu")
cx = convnext_tiny(weights=None)
cx.classifier[-1] = nn.Linear(cx.classifier[-1].in_features, 2)
cx.load_state_dict(torch.load(CX_CKPT, map_location=device, weights_only=True))
cx = cx.to(device).eval()
print(f"  Loaded OK — {sum(p.numel() for p in cx.parameters())/1e6:.1f}M parameters")

# Target: last depthwise Conv2d in last ConvNeXt block
cx_target = [cx.features[7][-1].block[0]]
print(f"  Target layer: features[7][-1].block[0]  {cx_target[0].weight.shape}")

# ---------------------------------------------------------------------------
# Predictions on test set
# ---------------------------------------------------------------------------
print("\nRunning predictions on full test set...")
sfx = nn.Softmax(dim=1)

all_probs, all_preds, all_labels = [], [], []
batch = 8
for i in range(0, len(ts), batch):
    b    = ts[i:i+batch]
    imgs = torch.stack([eval_tf(Image.open(r["path"]).convert("RGB")) for r in b])
    with torch.no_grad():
        out = cx(imgs)
    all_probs += sfx(out)[:, 1].tolist()
    all_preds += out.argmax(1).tolist()
    all_labels += [r["label"] for r in b]

cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
tn, fp, fn, tp_val = cm.ravel()
acc = (tp_val + tn) / len(all_labels)
print(f"  TP={tp_val}  TN={tn}  FP={fp}  FN={fn}  Accuracy={acc:.4f}")

# Select images for visualisation (small sample — fast on CPU)
random.seed(42)
pick = lambda lst, n: random.sample(lst, min(n, len(lst)))

TP_i = [i for i, (l, p) in enumerate(zip(all_labels, all_preds)) if l == 1 and p == 1]
TN_i = [i for i, (l, p) in enumerate(zip(all_labels, all_preds)) if l == 0 and p == 0]
FP_i = [i for i, (l, p) in enumerate(zip(all_labels, all_preds)) if l == 0 and p == 1]
FN_i = [i for i, (l, p) in enumerate(zip(all_labels, all_preds)) if l == 1 and p == 0]

# Use 3 images per category to keep it fast
N_PER_CAT = 3
selected = {
    "True_Positive":  pick(TP_i, N_PER_CAT),
    "True_Negative":  pick(TN_i, N_PER_CAT),
    "False_Positive": pick(FP_i, N_PER_CAT),
    "False_Negative": pick(FN_i, min(N_PER_CAT, len(FN_i))),
}
print(f"  Selected for GradCAM: "
      + "  ".join(f"{k}={len(v)}" for k, v in selected.items()))

# ---------------------------------------------------------------------------
# GradCAM++ and EigenCAM
# ---------------------------------------------------------------------------
try:
    from pytorch_grad_cam import EigenCAM, GradCAMPlusPlus
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    _GRADCAM_OK = True
    print("\ngrad-cam library: OK")
except ImportError:
    _GRADCAM_OK = False
    print("\nERROR: grad-cam not installed. Run: pip install grad-cam")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Grid visualisation function
# ---------------------------------------------------------------------------
def make_grid(model_fn, target_layers, idxs, title, save_path,
              cam_class=GradCAMPlusPlus, label="GradCAM++"):
    if not idxs:
        print(f"  Skipping '{title}' — no images in category"); return

    cam = cam_class(model=model_fn, target_layers=target_layers)
    n   = len(idxs)
    fig, axes = plt.subplots(n, 3, figsize=(12, 3.5 * n))
    if n == 1:
        axes = axes[None, :]
    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.01)

    for row, idx in enumerate(idxs):
        rec       = ts[idx]
        img       = Image.open(rec["path"]).convert("RGB")
        t         = eval_tf(img).unsqueeze(0)
        img_np    = denorm(t.squeeze(0))
        pred      = all_preds[idx]
        conf      = all_probs[idx] if pred == 1 else 1 - all_probs[idx]
        true_str  = "PNEUMONIA" if rec["label"] == 1 else "NORMAL"
        pred_str  = "PNEUMONIA" if pred == 1 else "NORMAL"

        with torch.enable_grad():
            hmap = cam(
                input_tensor=t,
                targets=[ClassifierOutputTarget(pred)]
            )[0]

        overlay = show_cam_on_image(img_np, hmap, use_rgb=True)

        axes[row, 0].imshow(img_np)
        axes[row, 0].axis("off")
        axes[row, 0].set_title(f"Original\nTrue: {true_str}", fontsize=9)

        axes[row, 1].imshow(hmap, cmap="jet")
        axes[row, 1].axis("off")
        axes[row, 1].set_title(f"{label} heatmap\n(red = most activated)", fontsize=9)

        axes[row, 2].imshow(overlay)
        axes[row, 2].axis("off")
        axes[row, 2].set_title(f"Overlay\nPred: {pred_str} ({conf:.2f})", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path.name}  ({save_path.stat().st_size // 1024} KB)")

# ---------------------------------------------------------------------------
# Generate ConvNeXt standalone Grad-CAM++ plots
# ---------------------------------------------------------------------------
print("\n=== ConvNeXt-Tiny — GradCAM++ ===")
for cat, idxs in selected.items():
    make_grid(
        cx, cx_target, idxs,
        title=f"ConvNeXt-Tiny GradCAM++ — {cat.replace('_', ' ')}",
        save_path=GRADCAM_OUT / f"convnext_{cat.lower()}.png",
        cam_class=GradCAMPlusPlus,
        label="GradCAM++",
    )

# ---------------------------------------------------------------------------
# Comparison grid: correct vs incorrect predictions side by side
# ---------------------------------------------------------------------------
print("\n=== Comparison grid: TP vs FP vs TN vs FN ===")
compare_cats = {
    "TP (Pneumonia detected)":     pick(TP_i, 2),
    "FP (Normal → Pneumonia)":     pick(FP_i, 2),
    "TN (Normal correct)":         pick(TN_i, 2),
    "FN (Pneumonia missed)":       pick(FN_i, min(2, len(FN_i))),
}

all_compare_idxs = [idx for idxs in compare_cats.values() for idx in idxs]
cam_cx = GradCAMPlusPlus(model=cx, target_layers=cx_target)

n_rows = len(all_compare_idxs)
fig, axes = plt.subplots(n_rows, 3, figsize=(12, 3.5 * n_rows))
if n_rows == 1:
    axes = axes[None, :]

fig.suptitle(
    "ConvNeXt-Tiny GradCAM++ — All Prediction Categories",
    fontsize=13, fontweight="bold"
)

row = 0
for cat_label, idxs in compare_cats.items():
    for idx in idxs:
        rec      = ts[idx]
        t        = eval_tf(Image.open(rec["path"]).convert("RGB")).unsqueeze(0)
        img_np   = denorm(t.squeeze(0))
        pred     = all_preds[idx]
        conf     = all_probs[idx] if pred == 1 else 1 - all_probs[idx]
        true_str = "PNEUMONIA" if rec["label"] == 1 else "NORMAL"
        pred_str = "PNEUMONIA" if pred == 1 else "NORMAL"
        correct  = "✓" if pred == rec["label"] else "✗"

        with torch.enable_grad():
            hmap = cam_cx(input_tensor=t, targets=[ClassifierOutputTarget(pred)])[0]

        overlay = show_cam_on_image(img_np, hmap, use_rgb=True)

        axes[row, 0].imshow(img_np); axes[row, 0].axis("off")
        axes[row, 0].set_title(f"{cat_label}\nTrue: {true_str}", fontsize=8)

        axes[row, 1].imshow(hmap, cmap="jet"); axes[row, 1].axis("off")
        axes[row, 1].set_title("GradCAM++ heatmap", fontsize=8)

        axes[row, 2].imshow(overlay); axes[row, 2].axis("off")
        axes[row, 2].set_title(f"Pred: {pred_str} {correct} ({conf:.2f})", fontsize=8)

        row += 1

plt.tight_layout()
comparison_path = GRADCAM_OUT / "convnext_all_categories_comparison.png"
plt.savefig(comparison_path, dpi=110, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {comparison_path.name}  ({comparison_path.stat().st_size // 1024} KB)")

# ---------------------------------------------------------------------------
# Save metadata
# ---------------------------------------------------------------------------
metadata = {
    "model": "ConvNeXt-Tiny",
    "checkpoint": str(CX_CKPT),
    "cam_method": "GradCAM++",
    "target_layer": "features[7][-1].block[0]",
    "test_set_size": len(ts),
    "confusion_matrix": {"TP": int(tp_val), "TN": int(tn), "FP": int(fp), "FN": int(fn)},
    "accuracy": round(acc, 4),
    "images_per_category": N_PER_CAT,
    "output_dir": str(GRADCAM_OUT),
    "note": (
        "Hybrid GradCAM requires best_hybrid_model.pth — "
        "download from Kaggle working/hybrid_results/ and place in "
        "processed/hybrid_convnext_cxformer/ to enable hybrid Grad-CAM."
    ),
}
with (GRADCAM_OUT / "gradcam_metadata.json").open("w") as f:
    json.dump(metadata, f, indent=2)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"\n{'='*55}")
print("  Grad-CAM++ complete")
print(f"{'='*55}")
for f in sorted(GRADCAM_OUT.iterdir()):
    print(f"  {f.name}  ({f.stat().st_size // 1024} KB)")
print(f"\n  Open PNG files in: {GRADCAM_OUT}")

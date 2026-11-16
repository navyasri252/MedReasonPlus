"""
src/shap_explainer.py
======================
SHAP Explainability for MedReason++ (ConvNeXt-Tiny)

Uses shap.GradientExplainer — works via backpropagation, not sampling.
Much faster than PartitionExplainer or KernelExplainer on CPU.

Strategy:
  - Background set: 20 randomly sampled images from the test set
  - Explain: P(PNEUMONIA) for the target image
  - Output: pixel-level SHAP values → aggregated to RGB channel → heatmap

Usage (standalone):
    python src/shap_explainer.py --image path/to/xray.jpeg

Usage (programmatic):
    from shap_explainer import SHAPExplainer
    exp = SHAPExplainer()
    result = exp.explain("path/to/xray.jpeg")
    # result["shap_heatmap_b64"]  → base64 PNG
    # result["shap_overlay_b64"]  → base64 overlay PNG
    # result["top_region"]        → "lower-left lung zone"

⚠ CLINICAL DISCLAIMER: Research prototype. NOT for clinical use.
"""

import base64
import io
import random
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import convnext_tiny

_SRC = Path(__file__).resolve().parent
_ROOT = _SRC.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

CX_CKPT   = _ROOT / "processed" / "convnext" / "best_model.pt"
DATA_ROOT = _ROOT / "archive (7)" / "chest_xray" / "test"

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

eval_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


# ---------------------------------------------------------------------------
# Model wrapper for SHAP: takes a tensor batch, returns P(PNEUMONIA) only
# ---------------------------------------------------------------------------

class ConvNeXtPneumoniaWrapper(nn.Module):
    """
    Wraps ConvNeXt-Tiny to output P(PNEUMONIA) as a single scalar per image.
    SHAP GradientExplainer works best with single-output models.
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x)                     # [B, 2]
        probs  = torch.softmax(logits, dim=1)       # [B, 2]
        return probs[:, 1:2]                        # [B, 1] — P(PNEUMONIA)


# ---------------------------------------------------------------------------
# Background dataset builder
# ---------------------------------------------------------------------------

def build_background(
    data_root: Path = DATA_ROOT,
    n_samples: int = 20,
    seed: int = 42,
) -> torch.Tensor:
    """
    Build a background tensor from the test set for SHAP GradientExplainer.

    Uses a small set of images (default 20) — enough to estimate expected
    SHAP values without being computationally prohibitive on CPU.

    Returns
    -------
    torch.Tensor  shape (n_samples, 3, 224, 224)
    """
    all_images: list[Path] = []
    for cls in ["NORMAL", "PNEUMONIA"]:
        cls_dir = data_root / cls
        if cls_dir.is_dir():
            all_images.extend(sorted(cls_dir.glob("*.jpeg")))

    if not all_images:
        # Fallback: use random noise if dataset not available
        print("[SHAPExplainer] Warning: test images not found, using random background.")
        torch.manual_seed(seed)
        return torch.randn(n_samples, 3, 224, 224) * 0.1

    rng = random.Random(seed)
    sampled = rng.sample(all_images, min(n_samples, len(all_images)))

    tensors = []
    for p in sampled:
        img = Image.open(p).convert("RGB")
        tensors.append(eval_tf(img))

    background = torch.stack(tensors)   # [N, 3, 224, 224]
    print(f"[SHAPExplainer] Background: {background.shape} from {data_root}")
    return background


# ---------------------------------------------------------------------------
# SHAP result → base64 helpers
# ---------------------------------------------------------------------------

def _shap_to_heatmap_b64(shap_values: np.ndarray) -> str:
    """
    Convert SHAP values (3, 224, 224) → base64 PNG heatmap.

    Aggregates over RGB channels using L2 norm (preserves both
    positive and negative contributions' magnitude).
    """
    # Aggregate 3 channels → 1 spatial map
    agg = np.linalg.norm(shap_values, axis=0)   # (224, 224)

    # Clip extreme values and normalise to [0, 1]
    p_low, p_high = np.percentile(agg, [1, 99])
    agg_clipped   = np.clip(agg, p_low, p_high)
    agg_norm      = (agg_clipped - agg_clipped.min()) / (agg_clipped.max() - agg_clipped.min() + 1e-8)

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(agg_norm, cmap="hot", vmin=0, vmax=1)
    ax.axis("off")
    fig.tight_layout(pad=0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _shap_overlay_b64(
    img_np: np.ndarray,       # (224, 224, 3) float [0,1]
    shap_values: np.ndarray,  # (3, 224, 224)
    alpha: float = 0.5,
) -> str:
    """Overlay SHAP heatmap on the original image."""
    agg = np.linalg.norm(shap_values, axis=0)
    p_low, p_high = np.percentile(agg, [1, 99])
    agg_clipped   = np.clip(agg, p_low, p_high)
    agg_norm      = (agg_clipped - agg_clipped.min()) / (agg_clipped.max() - agg_clipped.min() + 1e-8)

    colormap  = cm.get_cmap("hot")
    heatmap   = colormap(agg_norm)[:, :, :3]   # (224, 224, 3)
    overlay   = (1 - alpha) * img_np + alpha * heatmap
    overlay   = np.clip(overlay, 0, 1)

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(overlay)
    ax.axis("off")
    fig.tight_layout(pad=0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _describe_region(shap_values: np.ndarray) -> str:
    """Describe which image zone has the highest SHAP magnitude."""
    agg = np.linalg.norm(shap_values, axis=0)
    h, w = agg.shape
    thirds = [agg[:, :w//3].mean(), agg[:, w//3:2*w//3].mean(), agg[:, 2*w//3:].mean()]
    halves = [agg[:h//2, :].mean(), agg[h//2:, :].mean()]
    lr  = ["left", "central", "right"][int(np.argmax(thirds))]
    ud  = "upper" if halves[0] > halves[1] else "lower"
    return f"{ud}-{lr} lung zone"


def _denorm(t: torch.Tensor) -> np.ndarray:
    t = t.clone()
    for c, m, s in zip(range(3), MEAN, STD):
        t[c] = t[c] * s + m
    return t.permute(1, 2, 0).clamp(0, 1).numpy()


# ---------------------------------------------------------------------------
# SHAPExplainer class
# ---------------------------------------------------------------------------

class SHAPExplainer:
    """
    Generates SHAP explanations for ConvNeXt-Tiny predictions.

    Loads the model once and caches the background tensor.
    Subsequent calls to explain() reuse both.
    """

    def __init__(
        self,
        ckpt_path: Optional[Path] = None,
        data_root: Optional[Path] = None,
        n_background: int = 20,
        seed: int = 42,
        device: Optional[str] = None,
    ):
        self.device = torch.device(device or "cpu")
        self.ckpt_path  = ckpt_path or CX_CKPT
        self.data_root  = data_root or DATA_ROOT
        self.n_bg       = n_background
        self.seed       = seed
        self._model     = None
        self._wrapper   = None
        self._explainer = None
        self._background= None
        print(f"[SHAPExplainer] Ready (device={self.device}). Loads on first call.")

    def _load(self):
        if self._explainer is not None:
            return

        import shap

        print("[SHAPExplainer] Loading ConvNeXt-Tiny checkpoint…")
        base = convnext_tiny(weights=None)
        base.classifier[-1] = nn.Linear(base.classifier[-1].in_features, 2)
        base.load_state_dict(
            torch.load(self.ckpt_path, map_location=self.device, weights_only=True)
        )
        base = base.to(self.device).eval()
        self._model   = base
        self._wrapper = ConvNeXtPneumoniaWrapper(base).to(self.device).eval()

        print("[SHAPExplainer] Building background tensor…")
        self._background = build_background(self.data_root, self.n_bg, self.seed)
        self._background = self._background.to(self.device)

        print("[SHAPExplainer] Initialising GradientExplainer…")
        self._explainer = shap.GradientExplainer(self._wrapper, self._background)
        print("[SHAPExplainer] Ready.")

    def explain(self, image_path: str) -> dict:
        """
        Compute SHAP values for a single chest X-ray image.

        Parameters
        ----------
        image_path : str  path to JPEG chest X-ray

        Returns
        -------
        dict with keys:
            prediction          NORMAL / PNEUMONIA
            confidence          float 0–1
            shap_heatmap_b64    base64 PNG (hot colormap)
            shap_overlay_b64    base64 PNG (original + overlay)
            top_region          str e.g. "lower-left lung zone"
            shap_max            float (max SHAP magnitude)
            shap_mean           float (mean SHAP magnitude)
        """
        self._load()   # no-op after first call

        img  = Image.open(image_path).convert("RGB")
        t    = eval_tf(img).unsqueeze(0).to(self.device)   # [1, 3, 224, 224]
        img_np = _denorm(t.squeeze(0).cpu())               # (224, 224, 3) for overlay

        print(f"[SHAPExplainer] Computing SHAP values for {Path(image_path).name}…")

        # Get prediction first
        with torch.no_grad():
            logits = self._model(t)
            probs  = torch.softmax(logits, dim=1)[0]
            pred_idx = int(probs.argmax().item())
            confidence = float(probs[pred_idx].item())
        label = "PNEUMONIA" if pred_idx == 1 else "NORMAL"

        # Compute SHAP values
        # shap_values shape: list of one array [1, 3, 224, 224]
        shap_vals = self._explainer.shap_values(t, nsamples=100)

        if isinstance(shap_vals, list):
            sv = shap_vals[0]   # first (and only) output neuron
        else:
            sv = shap_vals

        # sv shape: (1, 3, 224, 224) — squeeze to (3, 224, 224)
        sv_squeezed = sv[0]   # (3, 224, 224)

        heatmap_b64 = _shap_to_heatmap_b64(sv_squeezed)
        overlay_b64 = _shap_overlay_b64(img_np, sv_squeezed)
        top_region  = _describe_region(sv_squeezed)

        agg = np.linalg.norm(sv_squeezed, axis=0)

        print(f"[SHAPExplainer] Done. Prediction: {label} ({confidence*100:.1f}%)")

        return {
            "prediction":        label,
            "confidence":        round(confidence, 4),
            "shap_heatmap_b64":  heatmap_b64,
            "shap_overlay_b64":  overlay_b64,
            "top_region":        top_region,
            "shap_max":          round(float(agg.max()), 4),
            "shap_mean":         round(float(agg.mean()), 4),
        }


# ---------------------------------------------------------------------------
# Batch SHAP for the GradCAM gallery (all 4 categories)
# ---------------------------------------------------------------------------

def run_shap_gallery(
    output_dir: Optional[Path] = None,
    n_per_category: int = 3,
    seed: int = 42,
) -> Path:
    """
    Generate SHAP grid images for all 4 prediction categories
    (TP, TN, FP, FN) and save to reports/shap/.

    Uses the same test images as the GradCAM gallery for direct comparison.
    """
    out_dir = output_dir or (_ROOT / "reports" / "shap")
    out_dir.mkdir(parents=True, exist_ok=True)

    explainer = SHAPExplainer()
    explainer._load()

    # Collect test images and get predictions
    all_records = []
    for cls, lbl in [("NORMAL", 0), ("PNEUMONIA", 1)]:
        cls_dir = DATA_ROOT / cls
        if cls_dir.is_dir():
            for p in sorted(cls_dir.glob("*.jpeg")):
                all_records.append({"path": p, "true_label": lbl, "class_name": cls})

    if not all_records:
        print("No test images found. Skipping gallery.")
        return out_dir

    # Get predictions for all records
    print(f"[SHAP Gallery] Getting predictions for {len(all_records)} images…")
    softmax = nn.Softmax(dim=1)
    preds = []
    batch_size = 32
    for i in range(0, len(all_records), batch_size):
        batch = all_records[i:i+batch_size]
        imgs  = torch.stack([eval_tf(Image.open(r["path"]).convert("RGB")) for r in batch])
        with torch.no_grad():
            probs = softmax(explainer._model(imgs))[:, 1].tolist()
        for r, p in zip(batch, probs):
            r["prob"] = p
            r["pred"] = 1 if p >= 0.5 else 0
        preds.extend(batch)

    # Categorise
    rng = random.Random(seed)
    cats = {
        "true_positive":  [r for r in preds if r["true_label"]==1 and r["pred"]==1],
        "true_negative":  [r for r in preds if r["true_label"]==0 and r["pred"]==0],
        "false_positive": [r for r in preds if r["true_label"]==0 and r["pred"]==1],
        "false_negative": [r for r in preds if r["true_label"]==1 and r["pred"]==0],
    }

    titles = {
        "true_positive":  "SHAP — True Positives (Pneumonia correctly detected)",
        "true_negative":  "SHAP — True Negatives (Normal correctly identified)",
        "false_positive": "SHAP — False Positives (Normal misclassified)",
        "false_negative": "SHAP — False Negatives (Pneumonia missed)",
    }

    for cat, records in cats.items():
        selected = rng.sample(records, min(n_per_category, len(records)))
        if not selected:
            print(f"  Skipping {cat} — no images.")
            continue

        n = len(selected)
        fig, axes = plt.subplots(n, 3, figsize=(11, 3.5 * n), facecolor="#06090f")
        if n == 1:
            axes = axes[None, :]
        fig.suptitle(titles[cat], fontsize=12, fontweight="bold", color="white", y=1.01)

        for row, rec in enumerate(selected):
            img = Image.open(rec["path"]).convert("RGB")
            t   = eval_tf(img).unsqueeze(0)
            img_np = _denorm(t.squeeze(0))

            # SHAP values
            sv    = explainer._explainer.shap_values(t, nsamples=100)
            sv    = sv[0][0] if isinstance(sv, list) else sv[0]  # (3,224,224)
            agg   = np.linalg.norm(sv, axis=0)
            p_lo, p_hi = np.percentile(agg, [1, 99])
            agg_n = np.clip((agg - p_lo) / (p_hi - p_lo + 1e-8), 0, 1)

            overlay = np.clip((1-.5)*img_np + .5*cm.get_cmap("hot")(agg_n)[:,:,:3], 0, 1)

            true_s = rec["class_name"]
            pred_s = "PNEUMONIA" if rec["pred"]==1 else "NORMAL"
            conf   = rec["prob"] if rec["pred"]==1 else 1-rec["prob"]

            for ax in axes[row]:
                ax.set_facecolor("#06090f")

            axes[row,0].imshow(img_np);              axes[row,0].axis("off")
            axes[row,0].set_title(f"Original\nTrue: {true_s}", fontsize=8.5, color="white")

            axes[row,1].imshow(agg_n, cmap="hot");   axes[row,1].axis("off")
            axes[row,1].set_title("SHAP Heatmap\n(high=red, low=black)", fontsize=8.5, color="white")

            axes[row,2].imshow(overlay);             axes[row,2].axis("off")
            axes[row,2].set_title(f"Overlay\nPred: {pred_s} ({conf:.0%})", fontsize=8.5, color="white")

        fig.patch.set_facecolor("#06090f")
        plt.tight_layout()
        save_path = out_dir / f"shap_{cat}.png"
        fig.savefig(save_path, dpi=120, bbox_inches="tight", facecolor="#06090f")
        plt.close(fig)
        print(f"  Saved: {save_path.name}  ({save_path.stat().st_size//1024} KB)")

    # Comparison with GradCAM
    print("[SHAP Gallery] Generating SHAP vs GradCAM comparison…")
    _make_comparison(explainer, preds, out_dir, rng)

    print(f"\n[SHAP Gallery] Complete. Files saved to: {out_dir}")
    return out_dir


def _make_comparison(explainer, preds, out_dir, rng):
    """Side-by-side SHAP vs GradCAM++ for TP images."""
    try:
        from pytorch_grad_cam import GradCAMPlusPlus
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    except ImportError:
        print("  Skipping comparison — grad-cam not installed.")
        return

    tps = [r for r in preds if r["true_label"]==1 and r["pred"]==1]
    selected = rng.sample(tps, min(3, len(tps)))
    if not selected:
        return

    cam = GradCAMPlusPlus(
        model=explainer._model,
        target_layers=[explainer._model.features[7][-1].block[0]]
    )

    n = len(selected)
    fig, axes = plt.subplots(n, 4, figsize=(15, 4*n), facecolor="#06090f")
    if n == 1: axes = axes[None, :]
    fig.suptitle("GradCAM++ vs SHAP — True Positives (Pneumonia)", fontsize=12, fontweight="bold", color="white")

    col_titles = ["Original", "GradCAM++", "SHAP Heatmap", "SHAP Overlay"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=9.5, fontweight="bold", color="#60a5fa", pad=6)

    for row, rec in enumerate(selected):
        img    = Image.open(rec["path"]).convert("RGB")
        t      = eval_tf(img).unsqueeze(0)
        img_np = _denorm(t.squeeze(0))

        # GradCAM++
        with torch.enable_grad():
            gcam = cam(input_tensor=t, targets=[ClassifierOutputTarget(1)])[0]
        gcam_overlay = show_cam_on_image(img_np, gcam, use_rgb=True)

        # SHAP
        sv  = explainer._explainer.shap_values(t, nsamples=100)
        sv  = sv[0][0] if isinstance(sv, list) else sv[0]
        agg = np.linalg.norm(sv, axis=0)
        p_lo, p_hi = np.percentile(agg, [1, 99])
        agg_n = np.clip((agg - p_lo) / (p_hi - p_lo + 1e-8), 0, 1)
        shap_ov = np.clip((1-.5)*img_np + .5*cm.get_cmap("hot")(agg_n)[:,:,:3], 0, 1)

        for ax in axes[row]: ax.set_facecolor("#06090f"); ax.axis("off")

        axes[row,0].imshow(img_np)
        axes[row,1].imshow(gcam_overlay)
        axes[row,2].imshow(agg_n, cmap="hot")
        axes[row,3].imshow(shap_ov)

    fig.patch.set_facecolor("#06090f")
    plt.tight_layout()
    cmp_path = out_dir / "shap_vs_gradcam_comparison.png"
    fig.savefig(cmp_path, dpi=120, bbox_inches="tight", facecolor="#06090f")
    plt.close(fig)
    print(f"  Saved: {cmp_path.name}  ({cmp_path.stat().st_size//1024} KB)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(description="SHAP explainability for MedReason++")
    parser.add_argument("--image",   default=None, help="Single image to explain")
    parser.add_argument("--gallery", action="store_true", help="Generate full SHAP gallery")
    args = parser.parse_args()

    if args.gallery:
        print("Generating full SHAP gallery (this takes ~10–20 min on CPU)…")
        run_shap_gallery()
    elif args.image:
        exp = SHAPExplainer()
        result = exp.explain(args.image)
        print(f"\nPrediction : {result['prediction']} ({result['confidence']*100:.1f}%)")
        print(f"Top region : {result['top_region']}")
        print(f"SHAP max   : {result['shap_max']}")
        print(f"SHAP mean  : {result['shap_mean']}")
        # Save images
        for key, fname in [("shap_heatmap_b64","shap_heatmap.png"), ("shap_overlay_b64","shap_overlay.png")]:
            with open(fname, "wb") as f:
                f.write(base64.b64decode(result[key]))
            print(f"Saved: {fname}")
    else:
        print("Usage: python src/shap_explainer.py --image path/to/xray.jpeg")
        print("       python src/shap_explainer.py --gallery")

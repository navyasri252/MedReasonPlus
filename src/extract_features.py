"""
src/extract_features.py
========================
Feature extraction pipeline for MedReason++ agentic system.

Extracts from a single chest X-ray image:
  1. ConvNeXt-Tiny 768-dim features + GradCAM++ heatmap
  2. CXformer-S 384-dim chest-domain features
  3. Hybrid model prediction + confidence

Returns a SharedContext dict that is passed to the multi-agent system.

Usage (standalone):
    python src/extract_features.py --image path/to/xray.jpeg
"""
import sys
import json
import base64
import io
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import convnext_tiny
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# ---------------------------------------------------------------------------
# Paths — all relative to project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = _SRC.parent
CX_CKPT      = PROJECT_ROOT / "processed" / "convnext"  / "best_model.pt"
CXF_CKPT     = PROJECT_ROOT / "processed" / "cxformer"  / "best_model.pt"
HYB_CKPT     = PROJECT_ROOT / "processed" / "hybrid_convnext_cxformer" / "best_hybrid_model.pth"

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
PROJ = 256

# ---------------------------------------------------------------------------
# Transform helpers
# ---------------------------------------------------------------------------
eval_tf_224 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

def _denorm(t: torch.Tensor) -> np.ndarray:
    t = t.clone()
    for c, m, s in zip(range(3), MEAN, STD):
        t[c] = t[c] * s + m
    return t.permute(1, 2, 0).clamp(0, 1).numpy()


# ---------------------------------------------------------------------------
# Model definitions (must match training code exactly)
# ---------------------------------------------------------------------------

def _build_convnext() -> nn.Module:
    m = convnext_tiny(weights=None)
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, 2)
    return m


def _build_convnext_extractor(base: nn.Module) -> nn.Module:
    class _Ext(nn.Module):
        def __init__(self, b):
            super().__init__()
            self.features = b.features
            self.avgpool  = b.avgpool
        def forward(self, x):
            return self.avgpool(self.features(x)).flatten(1)
    return _Ext(base)


def _build_cxformer_extractor():
    from transformers import AutoModel
    bb = AutoModel.from_pretrained("m42-health/CXformer-small")
    class _Ext(nn.Module):
        def __init__(self, b): super().__init__(); self.bb = b
        def forward(self, x): return self.bb(pixel_values=x).last_hidden_state[:, 0, :]
    return _Ext(bb)


class _HybridFusion(nn.Module):
    def __init__(self, cx_enc, cxf_enc):
        super().__init__()
        self.cx_enc   = cx_enc
        self.cxf_enc  = cxf_enc
        self.proj_cx  = nn.Sequential(nn.Linear(768, PROJ), nn.LayerNorm(PROJ), nn.GELU())
        self.proj_cxf = nn.Sequential(nn.Linear(384, PROJ), nn.LayerNorm(PROJ), nn.GELU())
        self.classifier = nn.Sequential(
            nn.Linear(PROJ * 2, 256), nn.GELU(), nn.Dropout(0.3), nn.Linear(256, 2))
    def forward(self, cx_img, cxf_img):
        with torch.no_grad():
            f_cx  = self.cx_enc(cx_img)
            f_cxf = self.cxf_enc(cxf_img)
        return self.classifier(torch.cat([self.proj_cx(f_cx), self.proj_cxf(f_cxf)], dim=1))


# ---------------------------------------------------------------------------
# FeatureExtractor — loads all models once, reuses for multiple images
# ---------------------------------------------------------------------------

class FeatureExtractor:
    """
    Loads ConvNeXt, CXformer, and Hybrid models once.
    Call extract(image_path) for each new image.
    """

    def __init__(self, device: Optional[str] = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"[FeatureExtractor] Device: {self.device}")
        self._load_models()

    def _load_models(self):
        print("[FeatureExtractor] Loading ConvNeXt-Tiny...")
        self.cx_model = _build_convnext()
        self.cx_model.load_state_dict(
            torch.load(CX_CKPT, map_location=self.device, weights_only=True))
        self.cx_model = self.cx_model.to(self.device).eval()

        print("[FeatureExtractor] Loading CXformer-S backbone...")
        from transformers import AutoImageProcessor
        self.cxf_proc = AutoImageProcessor.from_pretrained(
            "m42-health/CXformer-small", trust_remote_code=True)
        cx_base2 = _build_convnext()
        self.cx_enc  = _build_convnext_extractor(cx_base2)
        for p in self.cx_enc.parameters(): p.requires_grad = False
        self.cxf_enc = _build_cxformer_extractor()
        for p in self.cxf_enc.parameters(): p.requires_grad = False

        print("[FeatureExtractor] Loading Hybrid model...")
        self.hybrid = _HybridFusion(self.cx_enc, self.cxf_enc)
        self.hybrid.load_state_dict(
            torch.load(HYB_CKPT, map_location=self.device, weights_only=True), strict=False)
        self.hybrid = self.hybrid.to(self.device).eval()

        # GradCAM++ target layer
        self.cam = GradCAMPlusPlus(
            model=self.cx_model,
            target_layers=[self.cx_model.features[7][-1].block[0]]
        )
        print("[FeatureExtractor] All models loaded.")

    def extract(self, image_path: str) -> dict:
        """
        Extract all features and GradCAM for a single image.

        Returns
        -------
        dict (SharedContext) with keys:
            image_path, prediction, confidence, label_str,
            gradcam_heatmap_b64,  gradcam_overlay_b64,
            convnext_features,    cxformer_features,
            convnext_logits,      hybrid_logits
        """
        img_path = Path(image_path)
        img      = Image.open(img_path).convert("RGB")

        # ── ConvNeXt preprocessing ────────────────────────────────────
        t224     = eval_tf_224(img).unsqueeze(0).to(self.device)
        img_np   = _denorm(t224.squeeze(0).cpu())

        # ── CXformer preprocessing ────────────────────────────────────
        t518 = self.cxf_proc(img, return_tensors="pt")["pixel_values"].to(self.device)

        # ── GradCAM++ ─────────────────────────────────────────────────
        with torch.enable_grad():
            hmap = self.cam(
                input_tensor=t224,
                targets=[ClassifierOutputTarget(1)]  # target: PNEUMONIA
            )[0]

        overlay  = show_cam_on_image(img_np, hmap, use_rgb=True)
        hmap_b64 = _arr_to_b64(hmap, cmap="jet")
        ovl_b64  = _arr_to_b64(overlay)

        # ── ConvNeXt features (768-dim) ───────────────────────────────
        with torch.no_grad():
            cx_logits  = self.cx_model(t224)
            cx_feats   = self.hybrid.cx_enc(t224)   # [1, 768]

        # ── CXformer features (384-dim) ───────────────────────────────
        with torch.no_grad():
            cxf_feats  = self.hybrid.cxf_enc(t518)  # [1, 384]

        # ── Hybrid prediction ─────────────────────────────────────────
        with torch.no_grad():
            hyb_logits = self.hybrid(t224, t518)
            hyb_probs  = torch.softmax(hyb_logits, dim=1)[0]
            pred_idx   = int(hyb_probs.argmax().item())
            confidence = float(hyb_probs[pred_idx].item())

        label_str = "PNEUMONIA" if pred_idx == 1 else "NORMAL"

        # ── GradCAM activation summary ────────────────────────────────
        hmap_norm = (hmap - hmap.min()) / (hmap.max() - hmap.min() + 1e-8)
        activation_level = float(hmap_norm.mean())
        top_region        = _describe_activation_region(hmap_norm)

        return {
            # Core prediction
            "image_path":         str(img_path),
            "prediction":         label_str,
            "confidence":         round(confidence, 4),
            "pred_idx":           pred_idx,
            "prob_normal":        round(float(hyb_probs[0].item()), 4),
            "prob_pneumonia":     round(float(hyb_probs[1].item()), 4),

            # Visual evidence
            "gradcam_heatmap_b64": hmap_b64,
            "gradcam_overlay_b64": ovl_b64,
            "activation_level":   round(activation_level, 4),
            "activation_region":  top_region,

            # Feature vectors (as lists for JSON serialisation)
            "convnext_features":  cx_feats[0].tolist(),
            "cxformer_features":  cxf_feats[0].tolist(),

            # Logits for downstream use
            "hybrid_logits":      hyb_logits[0].tolist(),
            "convnext_logits":    cx_logits[0].tolist(),

            # Agent findings — to be filled by agents
            "agent_findings":     [],
            "final_report":       "",
        }


def _arr_to_b64(arr: np.ndarray, cmap: Optional[str] = None) -> str:
    """Convert numpy array to base64-encoded PNG string."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    if cmap:
        plt.imsave(buf, arr, format="png", cmap=cmap)
    else:
        if arr.dtype != np.uint8:
            arr = (arr * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(arr).save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _describe_activation_region(hmap: np.ndarray) -> str:
    """Describe which image region has highest GradCAM activation."""
    h, w   = hmap.shape
    thirds = [hmap[:, :w//3].mean(), hmap[:, w//3:2*w//3].mean(), hmap[:, 2*w//3:].mean()]
    rows   = [hmap[:h//2, :].mean(), hmap[h//2:, :].mean()]
    lr     = ["left", "central", "right"][int(np.argmax(thirds))]
    ud     = "upper" if rows[0] > rows[1] else "lower"
    return f"{ud}-{lr} lung zone"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to chest X-ray JPEG")
    args = parser.parse_args()

    extractor = FeatureExtractor()
    ctx = extractor.extract(args.image)

    print(f"\nPrediction  : {ctx['prediction']}")
    print(f"Confidence  : {ctx['confidence'] * 100:.1f}%")
    print(f"Activation  : {ctx['activation_level']:.3f} in {ctx['activation_region']}")
    print(f"ConvNeXt features dim : {len(ctx['convnext_features'])}")
    print(f"CXformer features dim : {len(ctx['cxformer_features'])}")

    # Save context (without large feature arrays) for inspection
    out = {k: v for k, v in ctx.items()
           if k not in ("convnext_features", "cxformer_features",
                        "gradcam_heatmap_b64", "gradcam_overlay_b64")}
    print("\nContext summary:")
    print(json.dumps(out, indent=2))

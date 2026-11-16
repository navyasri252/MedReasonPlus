"""
src/run_gradcam_all.py
======================
Grad-CAM visualisations for ALL models in MedReason++:
  1. ConvNeXt-Tiny standalone  → GradCAM++
  2. Hybrid (ConvNeXt branch)  → EigenCAM (frozen backbone)
  3. CXformer-S standalone     → EigenCAM (ViT, no spatial feature maps)

Saves PNG files to reports/gradcam/
"""
import json, random, sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch, torch.nn as nn
from PIL import Image
from sklearn.metrics import confusion_matrix
from torchvision import transforms
from torchvision.models import convnext_tiny
from transformers import AutoModel, AutoImageProcessor
from pytorch_grad_cam import GradCAMPlusPlus, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET      = PROJECT_ROOT / "archive (7)" / "chest_xray"
CX_CKPT      = PROJECT_ROOT / "processed" / "convnext"  / "best_model.pt"
CXF_CKPT     = PROJECT_ROOT / "processed" / "cxformer"  / "best_model.pt"
HYB_CKPT     = PROJECT_ROOT / "processed" / "hybrid_convnext_cxformer" / "best_hybrid_model.pth"
OUT          = PROJECT_ROOT / "reports" / "gradcam"
OUT.mkdir(parents=True, exist_ok=True)

MEAN = [0.485, 0.456, 0.406]; STD = [0.229, 0.224, 0.225]
C2I  = {"NORMAL": 0, "PNEUMONIA": 1}
PROJ = 256
device = torch.device("cpu")

print("=" * 60)
print("  MedReason++ — Grad-CAM All Models")
print("=" * 60)
print(f"  ConvNeXt  ckpt : {CX_CKPT.exists()}  {round(CX_CKPT.stat().st_size/1e6,1) if CX_CKPT.exists() else 0} MB")
print(f"  CXformer  ckpt : {CXF_CKPT.exists()}  {round(CXF_CKPT.stat().st_size/1e6,1) if CXF_CKPT.exists() else 0} MB")
print(f"  Hybrid    ckpt : {HYB_CKPT.exists()}  {round(HYB_CKPT.stat().st_size/1e6,1) if HYB_CKPT.exists() else 0} MB")

# ── Data helpers ──────────────────────────────────────────────────────
def collect(folder):
    recs = []
    for cls, idx in C2I.items():
        for p in sorted((folder/cls).glob("*.jpeg")):
            recs.append({"path": p, "label": idx, "class_name": cls})
    return recs

eval_tf_224 = transforms.Compose([
    transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

def denorm(t):
    t = t.clone()
    for c,m,s in zip(range(3),MEAN,STD): t[c]=t[c]*s+m
    return t.permute(1,2,0).clamp(0,1).numpy()

ts = collect(DATASET/"test")
print(f"\n  Test images: {len(ts)}  (NORMAL={sum(1 for r in ts if r['label']==0)}  PNEUMONIA={sum(1 for r in ts if r['label']==1)})")

# ── Prediction helper ─────────────────────────────────────────────────
def get_preds(model_fn, records, tf, batch=8):
    model_fn.eval(); probs=[]; preds=[]; labels=[]
    sfx = nn.Softmax(dim=1)
    for i in range(0, len(records), batch):
        b    = records[i:i+batch]
        imgs = torch.stack([tf(Image.open(r["path"]).convert("RGB")) for r in b])
        with torch.no_grad():
            out = model_fn(imgs)
        probs  += sfx(out)[:,1].tolist()
        preds  += out.argmax(1).tolist()
        labels += [r["label"] for r in b]
    return probs, preds, labels

# ── GradCAM grid ──────────────────────────────────────────────────────
def make_grid(model_fn, target_layers, idxs, all_preds, all_probs,
              title, save_path, cam_class, label):
    if not idxs:
        print(f"  Skipping {save_path.name} — no images"); return
    cam = cam_class(model=model_fn, target_layers=target_layers)
    n   = len(idxs)
    fig, axes = plt.subplots(n, 3, figsize=(12, 3.5*n))
    if n == 1: axes = axes[None,:]
    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.01)

    for row, idx in enumerate(idxs):
        rec   = ts[idx]
        t     = eval_tf_224(Image.open(rec["path"]).convert("RGB")).unsqueeze(0)
        inp   = denorm(t.squeeze(0))
        pred  = all_preds[idx]
        conf  = all_probs[idx] if pred==1 else 1-all_probs[idx]
        ts_   = "PNEUMONIA" if rec["label"]==1 else "NORMAL"
        ps_   = "PNEUMONIA" if pred==1 else "NORMAL"

        with torch.enable_grad():
            hmap = cam(input_tensor=t, targets=[ClassifierOutputTarget(pred)])[0]

        overlay = show_cam_on_image(inp, hmap, use_rgb=True)
        axes[row,0].imshow(inp);             axes[row,0].axis("off"); axes[row,0].set_title(f"Original\nTrue: {ts_}", fontsize=9)
        axes[row,1].imshow(hmap, cmap="jet"); axes[row,1].axis("off"); axes[row,1].set_title(f"{label}\n(red=high activation)", fontsize=9)
        axes[row,2].imshow(overlay);         axes[row,2].axis("off"); axes[row,2].set_title(f"Overlay\nPred: {ps_} ({conf:.2f})", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path.name}  ({save_path.stat().st_size//1024} KB)")

# ── Category selection ────────────────────────────────────────────────
def pick_categories(preds, labels, n=3):
    random.seed(42)
    pick = lambda lst,k: random.sample(lst, min(k,len(lst)))
    TP = [i for i,(l,p) in enumerate(zip(labels,preds)) if l==1 and p==1]
    TN = [i for i,(l,p) in enumerate(zip(labels,preds)) if l==0 and p==0]
    FP = [i for i,(l,p) in enumerate(zip(labels,preds)) if l==0 and p==1]
    FN = [i for i,(l,p) in enumerate(zip(labels,preds)) if l==1 and p==0]
    return {"True_Positive":pick(TP,n), "True_Negative":pick(TN,n),
            "False_Positive":pick(FP,n), "False_Negative":pick(FN,min(n,len(FN)))}

# ════════════════════════════════════════════════════════════════════
# MODEL 1 — ConvNeXt-Tiny  (GradCAM++)
# ════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  MODEL 1: ConvNeXt-Tiny — GradCAM++")
print("="*60)
cx = convnext_tiny(weights=None)
cx.classifier[-1] = nn.Linear(cx.classifier[-1].in_features, 2)
cx.load_state_dict(torch.load(CX_CKPT, map_location=device, weights_only=True))
cx = cx.eval()
cx_target = [cx.features[7][-1].block[0]]
print(f"  Loaded OK | Target: features[7][-1].block[0] {cx_target[0].weight.shape}")

cx_probs, cx_preds, cx_labels = get_preds(cx, ts, eval_tf_224)
cm = confusion_matrix(cx_labels, cx_preds, labels=[0,1]); tn,fp,fn,tp=cm.ravel()
print(f"  TP={tp}  TN={tn}  FP={fp}  FN={fn}  Acc={round((tp+tn)/len(cx_labels),4)}")
cx_cats = pick_categories(cx_preds, cx_labels, n=3)

for cat, idxs in cx_cats.items():
    make_grid(cx, cx_target, idxs, cx_preds, cx_probs,
              title=f"ConvNeXt-Tiny GradCAM++ — {cat.replace('_',' ')}",
              save_path=OUT/f"convnext_{cat.lower()}.png",
              cam_class=GradCAMPlusPlus, label="GradCAM++")

# ════════════════════════════════════════════════════════════════════
# MODEL 2 — Hybrid ConvNeXt branch  (EigenCAM)
# ════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  MODEL 2: Hybrid ConvNeXt branch — EigenCAM")
print("="*60)

_cx_base = convnext_tiny(weights=None)
class ConvNeXtExtractor(nn.Module):
    def __init__(self, b): super().__init__(); self.features=b.features; self.avgpool=b.avgpool
    def forward(self, x): return self.avgpool(self.features(x)).flatten(1)

cx_enc = ConvNeXtExtractor(_cx_base)
for p in cx_enc.parameters(): p.requires_grad = False

_cxf_bb = AutoModel.from_pretrained("m42-health/CXformer-small")
class CXformerExtractor(nn.Module):
    def __init__(self, bb): super().__init__(); self.bb=bb
    def forward(self, x): return self.bb(pixel_values=x).last_hidden_state[:,0,:]
cxf_enc = CXformerExtractor(_cxf_bb)
for p in cxf_enc.parameters(): p.requires_grad = False

class HybridFusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cx_enc=cx_enc; self.cxf_enc=cxf_enc
        self.proj_cx  = nn.Sequential(nn.Linear(768,PROJ), nn.LayerNorm(PROJ), nn.GELU())
        self.proj_cxf = nn.Sequential(nn.Linear(384,PROJ), nn.LayerNorm(PROJ), nn.GELU())
        self.classifier = nn.Sequential(nn.Linear(PROJ*2,256), nn.GELU(), nn.Dropout(0.3), nn.Linear(256,2))
    def forward(self, cx_img, cxf_img):
        with torch.no_grad():
            f_cx=self.cx_enc(cx_img); f_cxf=self.cxf_enc(cxf_img)
        return self.classifier(torch.cat([self.proj_cx(f_cx), self.proj_cxf(f_cxf)], dim=1))

hybrid = HybridFusionModel()
hybrid.load_state_dict(torch.load(HYB_CKPT, map_location=device, weights_only=True), strict=False)
hybrid.eval()

# EigenCAM wrapper — accesses ConvNeXt features directly, no gradients needed
class HybridCXWrapper(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.features   = h.cx_enc.features
        self.avgpool    = h.cx_enc.avgpool
        self.proj_cx    = h.proj_cx
        self.classifier = h.classifier
    def forward(self, x):
        f = self.avgpool(self.features(x)).flatten(1)
        p = self.proj_cx(f)
        z = torch.zeros(x.size(0), PROJ, device=x.device)
        return self.classifier(torch.cat([p, z], dim=1))

hyb_wrapper = HybridCXWrapper(hybrid).eval()
hyb_target  = [hyb_wrapper.features[7][-1].block[0]]
print(f"  Loaded OK | Target: features[7][-1].block[0] {hyb_target[0].weight.shape}")

# Use same image indices as ConvNeXt for fair comparison
hyb_preds_wrap, hyb_probs_wrap, _ = get_preds(hyb_wrapper, ts, eval_tf_224)
hyb_preds_wrap_int = [int(p) for p in hyb_preds_wrap]
cm2 = confusion_matrix(cx_labels, hyb_preds_wrap_int, labels=[0,1]); tn2,fp2,fn2,tp2=cm2.ravel()
print(f"  TP={tp2}  TN={tn2}  FP={fp2}  FN={fn2}  Acc={round((tp2+tn2)/len(cx_labels),4)}")
# Note: wrapper uses zero CXformer branch so predictions differ from full hybrid
# GradCAM shows ConvNeXt branch activation patterns regardless of accuracy
print("  (Wrapper uses ConvNeXt branch only — accuracy reflects branch contribution)")
print(f"  Using same {len(cx_cats['True_Positive'])} TP images as ConvNeXt for comparison")

for cat, idxs in cx_cats.items():  # same image selection as ConvNeXt
    make_grid(hyb_wrapper, hyb_target, idxs, hyb_preds_wrap, hyb_probs_wrap,
              title=f"Hybrid ConvNeXt Branch EigenCAM — {cat.replace('_',' ')}",
              save_path=OUT/f"hybrid_{cat.lower()}.png",
              cam_class=EigenCAM, label="EigenCAM")

# ════════════════════════════════════════════════════════════════════
# MODEL 3 — CXformer-S standalone  (EigenCAM on patch tokens)
# ════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  MODEL 3: CXformer-S standalone — EigenCAM")
print("="*60)

cxf_proc = AutoImageProcessor.from_pretrained("m42-health/CXformer-small", trust_remote_code=True)
cxf_bb2  = AutoModel.from_pretrained("m42-health/CXformer-small")

class CXformerClassifier(nn.Module):
    def __init__(self, bb, hdim=384):
        super().__init__()
        self.bb   = bb
        self.head = nn.Linear(hdim, 2)
    def forward(self, x):
        return self.head(self.bb(pixel_values=x).last_hidden_state[:,0,:])

cxf_model = CXformerClassifier(cxf_bb2)
cxf_model.load_state_dict(torch.load(CXF_CKPT, map_location=device, weights_only=True), strict=False)
cxf_model.eval()
print("  CXformer-S loaded OK")

# For CXformer (ViT/DINOv2), EigenCAM target is norm1 of the last encoder layer
cxf_target = [cxf_model.bb.encoder.layer[-1].norm1]
print(f"  Target: encoder.layer[-1].norm1  ({type(cxf_target[0]).__name__})")

# CXformer needs 518×518 input via its own processor
class CXfDataset(torch.utils.data.Dataset):
    def __init__(self, records, proc, augment=False):
        self.records=records; self.proc=proc
    def __len__(self): return len(self.records)
    def __getitem__(self, i):
        r=self.records[i]
        img=Image.open(r["path"]).convert("RGB")
        return self.proc(img, return_tensors="pt")["pixel_values"].squeeze(0), r["label"]

def get_preds_cxf(model_fn, records, proc, batch=4):
    model_fn.eval(); probs=[]; preds=[]; labels=[]
    sfx=nn.Softmax(dim=1)
    for i in range(0, len(records), batch):
        b    = records[i:i+batch]
        imgs = torch.stack([proc(Image.open(r["path"]).convert("RGB"), return_tensors="pt")["pixel_values"].squeeze(0) for r in b])
        with torch.no_grad():
            out = model_fn(imgs)
        probs  += sfx(out)[:,1].tolist()
        preds  += out.argmax(1).tolist()
        labels += [r["label"] for r in b]
    return probs, preds, labels

print("  Running CXformer-S predictions (518×518, slow on CPU)...")
# Only predict on the small subset needed for GradCAM — not full 624
all_idxs = sorted(set(sum(cx_cats.values(), [])))  # same ~10-14 images
cxf_subset = [ts[i] for i in all_idxs]
cxf_probs_sub, cxf_preds_sub, cxf_labels_sub = get_preds_cxf(cxf_model, cxf_subset, cxf_proc)
# remap to original indices
idx_remap = {orig_i: sub_i for sub_i, orig_i in enumerate(all_idxs)}
cxf_preds_full = {orig_i: cxf_preds_sub[sub_i] for orig_i, sub_i in idx_remap.items()}
cxf_probs_full = {orig_i: cxf_probs_sub[sub_i] for orig_i, sub_i in idx_remap.items()}
print(f"  Predicted {len(cxf_subset)} images for GradCAM")

# CXformer GradCAM uses 518×518 input
def make_grid_cxf(model_fn, target_layers, idxs, preds_dict, probs_dict,
                  title, save_path, cam_class, label):
    if not idxs:
        print(f"  Skipping {save_path.name}"); return
    cam = cam_class(model=model_fn, target_layers=target_layers)
    n   = len(idxs)
    fig, axes = plt.subplots(n, 3, figsize=(12, 3.5*n))
    if n == 1: axes = axes[None,:]
    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.01)

    for row, idx in enumerate(idxs):
        rec  = ts[idx]
        img  = Image.open(rec["path"]).convert("RGB")
        t    = cxf_proc(img, return_tensors="pt")["pixel_values"]  # [1,3,518,518]
        # Display image at 224x224 for readability
        inp  = denorm(eval_tf_224(img).squeeze(0) if False else
                      transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),
                                          transforms.Normalize(MEAN,STD)])(img))
        pred = preds_dict.get(idx, 1)
        conf = probs_dict.get(idx, 0.5)
        if pred == 0: conf = 1 - conf
        ts_  = "PNEUMONIA" if rec["label"]==1 else "NORMAL"
        ps_  = "PNEUMONIA" if pred==1 else "NORMAL"

        with torch.enable_grad():
            hmap = cam(input_tensor=t, targets=[ClassifierOutputTarget(pred)])[0]
        # hmap is 518×518 — resize to 224 for display
        hmap_r = np.array(Image.fromarray((hmap*255).astype(np.uint8)).resize((224,224), Image.BILINEAR)) / 255.0

        overlay = show_cam_on_image(inp, hmap_r, use_rgb=True)
        axes[row,0].imshow(inp);              axes[row,0].axis("off"); axes[row,0].set_title(f"Original\nTrue: {ts_}", fontsize=9)
        axes[row,1].imshow(hmap_r, cmap="jet"); axes[row,1].axis("off"); axes[row,1].set_title(f"{label}\n(red=high activation)", fontsize=9)
        axes[row,2].imshow(overlay);          axes[row,2].axis("off"); axes[row,2].set_title(f"Overlay\nPred: {ps_} ({conf:.2f})", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path.name}  ({save_path.stat().st_size//1024} KB)")

for cat, idxs in cx_cats.items():
    make_grid_cxf(cxf_model, cxf_target, idxs, cxf_preds_full, cxf_probs_full,
                  title=f"CXformer-S EigenCAM — {cat.replace('_',' ')}",
                  save_path=OUT/f"cxformer_{cat.lower()}.png",
                  cam_class=EigenCAM, label="EigenCAM")

# ════════════════════════════════════════════════════════════════════
# COMPARISON: all 3 models on same True Positive images
# ════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  COMPARISON: ConvNeXt vs Hybrid vs CXformer (True Positives)")
print("="*60)

compare_idxs = cx_cats["True_Positive"][:3]
cam_cx  = GradCAMPlusPlus(model=cx,          target_layers=cx_target)
cam_hyb = EigenCAM(       model=hyb_wrapper, target_layers=hyb_target)
cam_cxf = EigenCAM(       model=cxf_model,   target_layers=cxf_target)

n = len(compare_idxs)
fig, axes = plt.subplots(n, 7, figsize=(28, 4*n))
if n==1: axes=axes[None,:]
fig.suptitle("GradCAM Comparison — ConvNeXt GradCAM++ | Hybrid EigenCAM | CXformer EigenCAM\n(True Positives: Pneumonia correctly detected)",
             fontsize=12, fontweight="bold")
col_titles = ["Original","ConvNeXt\nheatmap","ConvNeXt\noverlay","Hybrid\nheatmap","Hybrid\noverlay","CXformer\nheatmap","CXformer\noverlay"]
for col, t in enumerate(col_titles):
    axes[0,col].set_title(t, fontsize=9, fontweight="bold")

for row, idx in enumerate(compare_idxs):
    rec  = ts[idx]
    img  = Image.open(rec["path"]).convert("RGB")
    t224 = eval_tf_224(img).unsqueeze(0)
    t518 = cxf_proc(img, return_tensors="pt")["pixel_values"]
    inp  = denorm(t224.squeeze(0))

    with torch.enable_grad():
        hcx  = cam_cx( input_tensor=t224, targets=[ClassifierOutputTarget(1)])[0]
        hhyb = cam_hyb(input_tensor=t224, targets=[ClassifierOutputTarget(1)])[0]
        hcxf = cam_cxf(input_tensor=t518, targets=[ClassifierOutputTarget(1)])[0]

    hcxf_r = np.array(Image.fromarray((hcxf*255).astype(np.uint8)).resize((224,224), Image.BILINEAR))/255.0

    axes[row,0].imshow(inp);                                    axes[row,0].axis("off")
    axes[row,1].imshow(hcx,   cmap="jet");                      axes[row,1].axis("off")
    axes[row,2].imshow(show_cam_on_image(inp, hcx,   use_rgb=True)); axes[row,2].axis("off")
    axes[row,3].imshow(hhyb,  cmap="jet");                      axes[row,3].axis("off")
    axes[row,4].imshow(show_cam_on_image(inp, hhyb,  use_rgb=True)); axes[row,4].axis("off")
    axes[row,5].imshow(hcxf_r,cmap="jet");                      axes[row,5].axis("off")
    axes[row,6].imshow(show_cam_on_image(inp, hcxf_r,use_rgb=True)); axes[row,6].axis("off")

plt.tight_layout()
comp_path = OUT / "comparison_all_models.png"
plt.savefig(comp_path, dpi=110, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {comp_path.name}  ({comp_path.stat().st_size//1024} KB)")

# ── Save metadata ─────────────────────────────────────────────────────
meta = {
    "models": {
        "convnext":  {"method": "GradCAMPlusPlus", "target": "features[7][-1].block[0]",
                      "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)},
        "hybrid":    {"method": "EigenCAM", "target": "features[7][-1].block[0] (ConvNeXt branch)"},
        "cxformer":  {"method": "EigenCAM", "target": "encoder.layer[-1].layernorm_before"},
    },
    "output_dir": str(OUT),
}
with (OUT/"gradcam_metadata.json").open("w") as f: json.dump(meta, f, indent=2)

# ── Final summary ─────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("  All Grad-CAM files generated:")
print(f"{'='*60}")
for f in sorted(OUT.iterdir()):
    print(f"  {f.name}  ({f.stat().st_size//1024} KB)")
print(f"\n  Location: {OUT}")

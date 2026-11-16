"""
MedReason++ — Kaggle GPU Training Script
=========================================
Run this in a Kaggle Notebook with GPU T4 accelerator.

Trains both ConvNeXt-Tiny and CXformer-S back-to-back.
Total expected time on T4 GPU: ~15–25 minutes for both models.

Dataset path on Kaggle: /kaggle/input/chest-xray-pneumonia/chest_xray/
"""

# ── INSTALL DEPENDENCIES ─────────────────────────────────────────────────
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "transformers", "opencv-python-headless"], check=True)

# ── IMPORTS ───────────────────────────────────────────────────────────────
import os, json, csv, time, random
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from torchvision import transforms
from PIL import Image

from sklearn.metrics import (
    confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score
)
import numpy as np

# ── CONFIG ────────────────────────────────────────────────────────────────
# Auto-detect dataset root — Kaggle path varies by how you added the dataset
def find_dataset_root():
    candidates = [
        Path("/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray"),
        Path("/kaggle/input/chest-xray-pneumonia/chest_xray"),
        Path("/kaggle/input/chest-xray-pneumonia"),
    ]
    # Also search all subdirectories under /kaggle/input
    for top in Path("/kaggle/input").rglob("train"):
        if (top / "NORMAL").is_dir() and (top / "PNEUMONIA").is_dir():
            candidates.insert(0, top.parent)
            break
    for c in candidates:
        if (c / "train" / "NORMAL").is_dir():
            print(f"  Dataset found at: {c}")
            return c
    # Print what IS in /kaggle/input to help debug
    print("ERROR: Could not find dataset. Contents of /kaggle/input:")
    for p in sorted(Path("/kaggle/input").rglob("*"))[:30]:
        print(f"  {p}")
    raise FileNotFoundError("Dataset not found. Check that you added the chest-xray-pneumonia dataset.")

DATASET_ROOT = find_dataset_root()
OUTPUT_DIR    = Path("/kaggle/working/medreason_results")
SEED          = 42
VAL_FRACTION  = 0.15
BATCH_SIZE    = 64       # can go larger on T4
LR_CONVNEXT   = 5e-5
LR_CXFORMER   = 1e-4
MAX_EPOCHS    = 10
PATIENCE      = 3

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── DATA UTILITIES ────────────────────────────────────────────────────────
CLASS_TO_IDX = {"NORMAL": 0, "PNEUMONIA": 1}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def collect_paths(folder):
    records = []
    for cls, idx in CLASS_TO_IDX.items():
        for p in sorted((folder / cls).glob("*.jpeg")):
            records.append({"path": p, "label": idx, "class_name": cls})
    return records

def stratified_split(records, val_frac=0.15, seed=42):
    rng = random.Random(seed)
    by_class = {}
    for r in records:
        by_class.setdefault(r["label"], []).append(r)
    train_recs, val_recs = [], []
    for lbl, recs in sorted(by_class.items()):
        s = recs[:]
        rng.shuffle(s)
        n_val = max(1, round(len(s) * val_frac))
        val_recs.extend(s[:n_val])
        train_recs.extend(s[n_val:])
    rng.shuffle(train_recs); rng.shuffle(val_recs)
    return train_recs, val_recs

def class_weights(labels):
    counts = Counter(labels)
    n = len(labels)
    w = torch.zeros(2)
    for i in range(2):
        w[i] = n / (2 * counts.get(i, 1))
    return w

def compute_metrics(labels, probs, threshold=0.5):
    y    = np.array(labels)
    yhat = np.array([1 if p >= threshold else 0 for p in probs])
    cm   = confusion_matrix(y, yhat, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "accuracy":    round(float((tp+tn)/len(y)),    4),
        "precision":   round(float(precision_score(y, yhat, zero_division=0)), 4),
        "recall":      round(float(recall_score(y,    yhat, zero_division=0)), 4),
        "specificity": round(float(tn/(tn+fp)) if (tn+fp)>0 else 0.0, 4),
        "f1":          round(float(f1_score(y,    yhat, zero_division=0)), 4),
        "roc_auc":     round(float(roc_auc_score(y, probs)), 4),
        "confusion_matrix": {"tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp)},
        "n_samples":   int(len(y)),
        "threshold":   threshold,
    }

# ── STANDARD DATASET (224×224, for ConvNeXt) ─────────────────────────────
class CXRDataset(Dataset):
    def __init__(self, records, transform):
        self.records = records
        self.transform = transform
    def __len__(self): return len(self.records)
    def __getitem__(self, i):
        r = self.records[i]
        img = Image.open(r["path"]).convert("RGB")
        return self.transform(img), r["label"]
    @property
    def labels(self): return [r["label"] for r in self.records]

train_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
eval_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

all_train = collect_paths(DATASET_ROOT / "train")
train_recs, val_recs = stratified_split(all_train, VAL_FRACTION, SEED)
test_recs  = collect_paths(DATASET_ROOT / "test")

print(f"Split — train:{len(train_recs)}  val:{len(val_recs)}  test:{len(test_recs)}")

# ── TRAINING / EVAL LOOPS ─────────────────────────────────────────────────
def train_epoch(model, loader, crit, opt):
    model.train()
    loss_sum = correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        out  = model(x)
        loss = crit(out, y)
        loss.backward(); opt.step()
        loss_sum += loss.item() * x.size(0)
        correct  += (out.argmax(1) == y).sum().item()
        total    += x.size(0)
    return loss_sum/total, correct/total

def eval_epoch(model, loader, crit, get_pixel=False):
    model.eval()
    loss_sum = correct = total = 0
    all_lbl, all_prob = [], []
    softmax = nn.Softmax(dim=1)
    with torch.no_grad():
        for batch in loader:
            x, y = batch[0].to(device), batch[1].to(device)
            out   = model(x)
            loss  = crit(out, y)
            probs = softmax(out)[:,1]
            loss_sum += loss.item() * x.size(0)
            correct  += (out.argmax(1) == y).sum().item()
            total    += x.size(0)
            all_lbl.extend(y.cpu().tolist())
            all_prob.extend(probs.cpu().tolist())
    return loss_sum/total, correct/total, all_lbl, all_prob

def train_model(model, train_loader, val_loader, test_loader, crit, lr,
                label, out_dir, max_epochs=MAX_EPOCHS, patience=PATIENCE):
    out_dir.mkdir(parents=True, exist_ok=True)
    opt  = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs, eta_min=lr/10)

    best_auc  = -1.0
    no_imp    = 0
    best_path = out_dir / "best_model.pt"
    history   = []
    t0        = time.time()

    for ep in range(1, max_epochs+1):
        te = time.time()
        tl, ta = train_epoch(model, train_loader, crit)
        vl, va, vlbl, vprb = eval_epoch(model, val_loader, crit)
        sched.step()

        val_auc = round(float(roc_auc_score(np.array(vlbl), vprb)), 4)
        improved = val_auc > best_auc
        note = "← best" if improved else f"no impr {no_imp+1}/{patience}"

        row = {
            "epoch": ep, "train_loss": round(tl,4), "train_acc": round(ta,4),
            "val_loss": round(vl,4), "val_acc": round(va,4),
            "val_auc": val_auc, "epoch_s": round(time.time()-te,1)
        }
        history.append(row)
        print(f"  [{label}] Ep {ep:2d}/{max_epochs} | "
              f"tr {tl:.4f}/{ta:.4f} | vl {vl:.4f}/{va:.4f} AUC {val_auc:.4f}  {note}")

        if improved:
            best_auc = val_auc; no_imp = 0
            torch.save(model.state_dict(), best_path)
        else:
            no_imp += 1
            if no_imp >= patience:
                print(f"  Early stopping.")
                break

    # ── Final test eval ──────────────────────────────────────────────────
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    _, _, tlbl, tprb = eval_epoch(model, test_loader, crit)
    metrics = compute_metrics(tlbl, tprb)
    metrics["total_time_s"] = round(time.time()-t0, 1)
    metrics["epochs_trained"] = len(history)
    metrics["best_val_auc"]   = best_auc

    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(out_dir / "history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=history[0].keys()); w.writeheader(); w.writerows(history)

    print(f"\n  [{label}] TEST: acc={metrics['accuracy']} rec={metrics['recall']} "
          f"spec={metrics['specificity']} F1={metrics['f1']} AUC={metrics['roc_auc']}")
    print(f"  CM: TN={metrics['confusion_matrix']['tn']} "
          f"FP={metrics['confusion_matrix']['fp']} "
          f"FN={metrics['confusion_matrix']['fn']} "
          f"TP={metrics['confusion_matrix']['tp']}")
    return metrics, history

# ════════════════════════════════════════════════════════════════════════════
# 1.  ConvNeXt-Tiny
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  1/2  ConvNeXt-Tiny")
print("="*60)

w_train = class_weights([r["label"] for r in train_recs]).to(device)
crit_cx  = nn.CrossEntropyLoss(weight=w_train)

cx_train_ds = CXRDataset(train_recs, train_tf)
cx_val_ds   = CXRDataset(val_recs,   eval_tf)
cx_test_ds  = CXRDataset(test_recs,  eval_tf)

cx_train_ld = DataLoader(cx_train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
cx_val_ld   = DataLoader(cx_val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
cx_test_ld  = DataLoader(cx_test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

cx_model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
cx_model.classifier[-1] = nn.Linear(cx_model.classifier[-1].in_features, 2)
cx_model = cx_model.to(device)
print(f"  ConvNeXt-Tiny params: {sum(p.numel() for p in cx_model.parameters())/1e6:.1f}M")

cx_metrics, cx_history = train_model(
    cx_model, cx_train_ld, cx_val_ld, cx_test_ld,
    crit_cx, LR_CONVNEXT, "ConvNeXt", OUTPUT_DIR / "convnext"
)

# ════════════════════════════════════════════════════════════════════════════
# 2.  CXformer-S
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  2/2  CXformer-S  (frozen backbone — linear probing)")
print("  License: CC BY-NC-4.0  |  Research use only")
print("="*60)

from transformers import AutoModel, AutoImageProcessor

cxf_processor = AutoImageProcessor.from_pretrained("m42-health/CXformer-small", trust_remote_code=True)
cxf_backbone  = AutoModel.from_pretrained("m42-health/CXformer-small")
hidden_dim    = cxf_backbone.config.hidden_size   # 384

# Freeze backbone
for p in cxf_backbone.parameters():
    p.requires_grad = False

class CXformerClassifier(nn.Module):
    def __init__(self, backbone, hidden, n_cls=2):
        super().__init__()
        self.backbone = backbone
        self.head     = nn.Linear(hidden, n_cls)
    def forward(self, x):
        out = self.backbone(pixel_values=x)
        return self.head(out.last_hidden_state[:, 0, :])

cxf_model = CXformerClassifier(cxf_backbone, hidden_dim).to(device)
trainable  = sum(p.numel() for p in cxf_model.parameters() if p.requires_grad)/1e6
print(f"  CXformer-S trainable: {trainable:.3f}M (head only)")

# CXformer-specific dataset using its own processor
class CXformerDataset(Dataset):
    def __init__(self, records, processor, augment=False):
        self.records   = records
        self.processor = processor
        self.aug       = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ]) if augment else None
    def __len__(self): return len(self.records)
    def __getitem__(self, i):
        r   = self.records[i]
        img = Image.open(r["path"]).convert("RGB")
        if self.aug: img = self.aug(img)
        pv  = self.processor(img, return_tensors="pt")["pixel_values"].squeeze(0)
        return pv, r["label"]

CXF_BATCH = 32   # smaller batch for 518×518 images
cxf_train_ds = CXformerDataset(train_recs, cxf_processor, augment=True)
cxf_val_ds   = CXformerDataset(val_recs,   cxf_processor, augment=False)
cxf_test_ds  = CXformerDataset(test_recs,  cxf_processor, augment=False)

cxf_train_ld = DataLoader(cxf_train_ds, batch_size=CXF_BATCH, shuffle=True,  num_workers=4, pin_memory=True)
cxf_val_ld   = DataLoader(cxf_val_ds,   batch_size=CXF_BATCH, shuffle=False, num_workers=4)
cxf_test_ld  = DataLoader(cxf_test_ds,  batch_size=CXF_BATCH, shuffle=False, num_workers=4)

crit_cxf = nn.CrossEntropyLoss(weight=w_train)

cxf_metrics, cxf_history = train_model(
    cxf_model, cxf_train_ld, cxf_val_ld, cxf_test_ld,
    crit_cxf, LR_CXFORMER, "CXformer-S", OUTPUT_DIR / "cxformer"
)

# ════════════════════════════════════════════════════════════════════════════
# Final summary
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  FINAL COMPARISON (test set, threshold=0.50)")
print("="*60)
print(f"  {'Model':<15} {'Acc':>6} {'Recall':>8} {'Spec':>8} {'F1':>8} {'AUC':>8}")
print(f"  {'-'*55}")
for name, m in [("ConvNeXt-Tiny", cx_metrics), ("CXformer-S", cxf_metrics)]:
    print(f"  {name:<15} {m['accuracy']:>6.4f} {m['recall']:>8.4f} "
          f"{m['specificity']:>8.4f} {m['f1']:>8.4f} {m['roc_auc']:>8.4f}")
print()
print("  Results saved to: /kaggle/working/medreason_results/")
print("  Download convnext/best_model.pt and cxformer/best_model.pt")
